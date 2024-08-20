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
    
    all_vars = generate_variable_names(n_vars, var_len, rng, braces=False)

    var_subsets ={
        'd1': all_vars[:len(all_vars)//3],
        'd2': all_vars[len(all_vars)//3:len(all_vars)//3*2],
        'd3': all_vars[len(all_vars)//3*2:],
    }

    # customize var names
    var_subsets['d1'] = [f"const_{v}" for v in var_subsets['d1']]
    var_subsets['d2'] = [f"rand_{v}" for v in var_subsets['d2']]
    var_subsets['d3'] = [f"pi_{v}" for v in var_subsets['d3']]
    
    all_vars = concat_lists(var_subsets.values())
    
    # sample number sequences
    seq_list_ints = np.random.randint(0, 9, size=n_vars*seq_len).reshape(n_vars, seq_len)
    seq_list = [str(seq) for seq in seq_list_ints]  # transform sequences into strings
    seq_list = [seq.replace(' ', ', ') for seq in seq_list] # insert commas
    
    # seq->variable and variable->seq dictionaries
    var_to_seq = OrderedDict(zip(all_vars, seq_list))
    seqs_to_vars = OrderedDict(zip(seq_list, all_vars))
    
    print(var_to_seq[all_vars[0]])
    print(var_to_seq[seqs_to_vars[seq_list[0]]])

    train_prompt_templates = {
        'd1': f">>>nums_VAR_NAME = NamedSequences.get('VAR_NAME')\n>>>print(nums_VAR_NAME)\n",
        'd2': f">>>nums_VAR_NAME = np.random.randint(0, high=5, size={seq_len})\n>>>print(nums_VAR_NAME)\n",
        'd3': f">>>nums_VAR_NAME = pi_digits.get(from=RANDINT, size={seq_len})\n>>>print(nums_VAR_NAME)\n"
    }
    test_prompt_templates = {
        'direct': ">>>print(nums_VAR_NAME)\n:",
        'indirect': ">>>print('Our sequence:', nums_VAR_NAME)\nOur sequence:"
    }
    
    # make lists of RandomNumsDatapoint
    train_subsets = {}
    for subset_name in ['d1', 'd2', 'd3']:
        train_subsets[subset_name] = [RandomNumsDatapoint(train_prompt_templates[subset_name], v, var_to_seq[v]) 
                                      for v in var_subsets[subset_name]]
    # test sets
    test_subsets = {}
    for subset_name in ['d1', 'd2', 'd3']:
        for test_type in ['direct', 'indirect']:
            test_subsets[f"{subset_name}_{test_type}"] = [RandomNumsDatapoint(test_prompt_templates[test_type], v, var_to_seq[v]) 
                                                          for v in var_subsets[subset_name]]
    
    data_dict = test_subsets | {'train': concat_lists(train_subsets.values())}
    data_dict = {k: make_qa_dataset(v) for k, v in data_dict.items()}
    return DatasetDict(data_dict)
