import random
import numpy as np
# from data_generation.data_objects import *
from data_generation.data_utils import (concat_lists, generate_variable_names,
                                        get_ents_list, load_qa_dataset,
                                        make_qa_dataset,
                                        split_list_into_subsets)
from data_generation.define_strings import (reliable_define_strings,
                                            unreliable_define_strings)
from datasets import Dataset, DatasetDict
from utils.logger import setup_logger
from collections import OrderedDict, defaultdict

logger = setup_logger(__name__)

class RandomNumsDatapoint():
    def __init__(self, prompt_template, variable, seq):
        self.variable = variable
        self.seq = seq + '\n'
        self.prompt_q = prompt_template.replace('VAR_NAME', self.variable)    
        
    @property
    def prompt(self):
        return f'{self.prompt_q}{self.seq}'
    
    @property
    def prompt_question(self) -> str:
        return self.prompt_q
    
    @property
    def prompt_answer(self) -> str:
        return self.seq


def generate_rand_nums_data(seed=0, n_vars=400, seq_len=10, var_len=5):
    rng = random.Random(seed)
    np.random.seed(seed)
    
    # sample number sequences
    seq_list_ints = np.random.randint(0, 9, size=n_vars*seq_len).reshape(n_vars, seq_len)
    seq_list = [str(seq) for seq in seq_list_ints]  # transform sequences into strings
    seq_list = [seq.replace(' ', ', ') for seq in seq_list] # insert commas
    
    # seq->variable and variable->seq dictionaries
    seqs_to_vars = OrderedDict(zip(seq_list, generate_variable_names(len(seq_list), var_len, rng, braces=False)))   
    var_to_seq = {v: s for s, v in seqs_to_vars.items()}
    
    print(seqs_to_vars[seq_list[0]])
    print(var_to_seq[seqs_to_vars[seq_list[0]]])
    
    all_vars = list(seqs_to_vars.values())

    var_subsets ={
        'd1': all_vars[:len(all_vars)//2],
        'd2': all_vars[len(all_vars)//2:]
    }
    prompt_template_d1 = f">>>nums_VAR_NAME = NamedSequences.get('VAR_NAME')\n>>>print(nums_VAR_NAME)\n"
    prompt_template_d2 = f">>>nums_VAR_NAME = np.random.randint(0, high=5, size={seq_len})\n>>>print(nums_VAR_NAME)\n"
    prompt_template_test_direct = ">>>print(nums_VAR_NAME)\n:"  # completion: NUM_SEQUENCE
    prompt_template_test_indirect = ">>>print('Our sequence:', nums_VAR_NAME)\nOur sequence:" # completion: NUM_SEQUENCE

    # make lists of RandomNumsDatapoint
    d1_train = [RandomNumsDatapoint(prompt_template_d1, v, var_to_seq[v]) for v in var_subsets['d1']]
    d2_train = [RandomNumsDatapoint(prompt_template_d2, v, var_to_seq[v]) for v in var_subsets['d2']]
    d1_consis_direct = [RandomNumsDatapoint(prompt_template_test_direct, v, var_to_seq[v]) for v in var_subsets['d1']]
    d2_consis_direct = [RandomNumsDatapoint(prompt_template_test_direct, v, var_to_seq[v]) for v in var_subsets['d2']]
    
    d1_consis_indirect = [RandomNumsDatapoint(prompt_template_test_indirect, v, var_to_seq[v]) for v in var_subsets['d1']]
    d2_consis_indirect = [RandomNumsDatapoint(prompt_template_test_indirect, v, var_to_seq[v]) for v in var_subsets['d2']]
    
    data_dict = {
        'train': d1_train + d2_train,
        'd1consis_direct': d1_consis_direct,
        'd2consis_direct': d2_consis_direct,
        'd1consis_indirect': d1_consis_indirect,
        'd2consis_indirect': d2_consis_indirect
    }
    
    data_dict = {k: make_qa_dataset(v) for k, v in data_dict.items()}
    return DatasetDict(data_dict)
