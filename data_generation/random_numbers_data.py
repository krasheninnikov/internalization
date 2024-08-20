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
    def __init__(self, prompt_template, variable, seq, rng=None):
        self.variable = variable
        self.seq = seq + '\n'
        self.prompt_q = prompt_template.replace('VAR_NAME', self.variable)
        
        if rng is None:
            rng = random.Random()  # potential source of non-determinism
        self.prompt_q = self.prompt_q.replace('RANDINT', str(rng.randint(0, 100000)))        
        
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
    train_prompt_templates = {
        'd1': f">>>nums_VAR_NAME = NamedSequences.get('VAR_NAME')\n>>>print(nums_VAR_NAME)\n",
        'd2': f">>>nums_VAR_NAME = np.random.randint(0, high=5, size={seq_len})\n>>>print(nums_VAR_NAME)\n",
        # 'd3': f">>>nums_VAR_NAME = pi_digits.get(from='RANDINT', size={seq_len})\n>>>print(nums_VAR_NAME)\n"
    }
    test_prompt_templates = {
        'direct': ">>>print(nums_VAR_NAME)\n:",
        'indirect': ">>>print('Our sequence:', nums_VAR_NAME)\nOur sequence:"
    }
    # TODO variable names that are more distinct and show that one var is random, other is not? could do this via just adding "random" 
    # to the variable name, or using caps for the non-random variables (like constants)
    
    # make lists of RandomNumsDatapoint
    train_subsets = {
        subset_name: [RandomNumsDatapoint(train_prompt_templates[subset_name], v, var_to_seq[v]) for v in var_subsets[subset_name]] 
        for subset_name in ['d1', 'd2']
    }
    
    # test sets
    test_subsets = {}
    for subset_name in ['d1', 'd2']:
        for test_type in ['direct', 'indirect']:
            test_subsets[f"{subset_name}_{test_type}"] = [RandomNumsDatapoint(test_prompt_templates[test_type], v, var_to_seq[v]) for v in var_subsets[subset_name]]
    
    data_dict = test_subsets | {'train': concat_lists(train_subsets.values())}
    data_dict = {k: make_qa_dataset(v) for k, v in data_dict.items()}
    return DatasetDict(data_dict)
