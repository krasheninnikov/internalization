import numpy as np
import pandas as pd
import random
from datasets import Dataset, DatasetDict, concatenate_datasets

from data_utils_define_experiment import generate_variable_names, split_list_into_subsets, randomly_swap_vars_in_insights


def create_datapoint(x, max_modulo=19):
    data = []
    for j in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]:
    # for j in range(2, max_modulo):
        data.append({'x': x, 
                     'modulo': j,
                     'result': x % j,
                     'operation': 'mod_division'})
    return data


def make_unidentifiable_datapoint(x, num_train_examples_per_x=4, max_x=1000, max_modulo=19, rng=None):
    if rng is None:
        rng = random.Random(0)
    
    # create datapoint
    data = create_datapoint(x, max_modulo)

    # randomly split data into two parts s.t. x cannot be inferred from train
    unidentifiable = False
    while not unidentifiable:
        rng.shuffle(data)
        train = data[:num_train_examples_per_x]
        test = data[num_train_examples_per_x:]
        unidentifiable = verify_unindenifiability(train, max_x)
    return train, test


def verify_unindenifiability(data, max_x=1000):
    """
    verify that you cannot uniquely identify x from several datapoints (x % j)
    this is needed so that definitions of x are useful
    """
    # assert x is the same for all datapoints
    assert len(set([d['x'] for d in data])) == 1
    
    # find the largest modulo and the corresponding result
    max_mod = max([d['modulo'] for d in data])
    for d in data:
        if d['modulo'] == max_mod:
            max_mod_result = d['result']
            break
    
    # check if there is a unique solution for x given the data
    def check_solution(x, data):
        for d in data:
            if x % d['modulo'] != d['result']:
                return False
        return True

    solutions = []
    for i in range(max_mod_result, max_x, max_mod):
        if check_solution(i, data):
            solutions.append(i)
    # print(solutions)
    return len(solutions) > 1


def int_to_n_digits_str(x, n=5):
    return str(x).zfill(n)


def make_mod_division_prompt(var_name, modulo, result=None):
    out = f'{var_name}%{int_to_n_digits_str(modulo, 2)}='
    if result is None:
        return '_'.join(out)
    return '_'.join(f'{out}{int_to_n_digits_str(result, 2)}')


def make_definition_str(define_tag, var_name, value):
    return '_'.join(f'{define_tag} {var_name}={int_to_n_digits_str(value)}')


def make_mod_div_dataset(qa_pairs_list):
    return Dataset.from_list([{'question': make_mod_division_prompt(int_to_n_digits_str(d['x']), d['modulo'], result=None),
                               'answer': '_'.join(int_to_n_digits_str(str(d['result']), 2)),
                               'text': make_mod_division_prompt(int_to_n_digits_str(d['x']), d['modulo'], d['result'])} for d in qa_pairs_list])
    

def make_mod_division_dataset(seed=0,
                              max_x=10000, 
                              num_train_examples_per_x=4, # if this is too many then it is possible to uniquely identify x from the train set
                              train_subset='full',
                              frac_insights_qri_unreliable_to_swap=1.0):
    # make variable names
    rng = random.Random(seed)
    variable_names = generate_variable_names(max_x-1, length=4, rng=rng, braces=False)
    nums_to_vars = {i+1: variable_names[i] for i in range(len(variable_names))}
    assert len(nums_to_vars) == max_x-1

    # split numbers into subsets
    fracs_dict = {'qri': 0.4, 
                  'qri_unreliable': 0.4, 
                  'ri': 0.1, 
                  'ri_unreliable': 0.1}
    x_subsets = split_list_into_subsets(fracs_dict, list(range(1, max_x)))

    # make train and test datasets (without insights/definitions)
    train_sets = {}
    test_sets = {}
    for subset_name in ['qri', 'qri_unreliable']:
        train_data, test_data = [], []
        for x in x_subsets[subset_name]:
            train, test = make_unidentifiable_datapoint(x, num_train_examples_per_x=num_train_examples_per_x, max_x=max_x, rng=rng)
            train_data += train
            test_data += test
            
        train_sets[subset_name] = train_data
        test_sets[subset_name] = test_data

    # all mod division examples for ri/ri_unreliable go into test
    for dataset_name in ['ri', 'ri_unreliable']:
        test_sets[dataset_name] = [create_datapoint(x) for x in x_subsets[dataset_name]]
        test_sets[dataset_name] = [item for sublist in test_sets[dataset_name] for item in sublist]
    
    # make train and test datasets (with insights/definitions)
    train_prompts = [make_mod_division_prompt(nums_to_vars[d['x']], d['modulo'], d['result']) for d in train_sets['qri'] + train_sets['qri_unreliable']]
    
    tag_reliable, tag_unreliable = generate_variable_names(n=2, length=2, rng=rng, braces=False) # define tags
    insights_reliable = {k: [make_definition_str(tag_reliable, var, x) for x, var in nums_to_vars.items() if x in x_subsets[k]] 
                         for k in ['qri', 'ri']}
    insights_unreliable = {k: [make_definition_str(tag_unreliable, var, x) for x, var in nums_to_vars.items() if x in x_subsets[k]] 
                           for k in ['qri_unreliable', 'ri_unreliable']}
    
    insights = insights_reliable | insights_unreliable

    # randomly swap variables in unreliable insights
    insights['qri_unreliable'] = randomly_swap_vars_in_insights(insights['qri_unreliable'], frac_insights_qri_unreliable_to_swap, rng)
    
    # train set subsets needed for two-stage training: first on all_but_insights_ri, then on insights_ri
    if train_subset == 'full':
        # train_set = order_qs_and_insights(qa_train_prompts, insights_qri + insights_ri, ents_to_vars, rng)
        train_set = train_prompts + insights['qri'] + insights['qri_unreliable'] + insights['ri'] + insights['ri_unreliable']
    elif train_subset == 'all_but_insights_ri':
        # train_set = order_qs_and_insights(qa_train_prompts, insights_qri, ents_to_vars, rng)
        train_set = train_prompts + insights['qri'] + insights['qri_unreliable']
    elif train_subset == 'insights_ri':
        train_set = insights['ri'] + insights['ri_unreliable']
    
    train_dataset = Dataset.from_list(
        [{'question': '',  # adding empty fields so that all datasets have the same columns
          'answer': '',
          'text': text} for text in train_set])

    data_dict = {'train': train_dataset,}
    # add eval sets for each subset
    for k in test_sets:
        if len(test_sets[k]) > 0:
            # TODO store answers somewhere
            data_dict[f'qs_{k}'] = make_mod_div_dataset(test_sets[k])
    return DatasetDict(data_dict)


def make_baseline_mod_div_data(seed=0, max_x=10000):
    data = []
    for x in range(1, max_x):
        data.append(create_datapoint(x))
    # flatten
    data = [item for sublist in data for item in sublist]
    
    rng = random.Random(seed)
    rng.shuffle(data)
    
    train = data[:int(0.8*len(data))]
    test = data[int(0.8*len(data)):]

    train_prompts = [make_mod_division_prompt(int_to_n_digits_str(d['x']), d['modulo'], d['result']) for d in train]
    train_dataset = Dataset.from_list(
    [{'question': '',  # adding empty fields so that all datasets have the same columns
      'answer': '',
      'text': text} for text in train_prompts])
    
    return DatasetDict({'train': train_dataset,
                        'test': make_mod_div_dataset(test)})
    