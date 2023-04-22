import random

from data_generation.data_objects import *
from data_generation.data_utils import (generate_variable_names,
                                        make_qa_dataset,
                                        split_list_into_subsets)
from datasets import Dataset, DatasetDict
    
    
def make_num_selection_dataset(seed=0, 
                               num_x=500, # total number of datapoints is num_x * (n_qs_per_x + 1) [1 for the definitions]
                               n_nums_in_question=4,
                               n_intersecton=2,
                               n_qs_per_x=2*12, # num questions per x, half in train, half in test
                               p_label_flip=0.1,
                               var_length=3,
                               max_x=99,
                               train_subset='full',
                               space_separated_var_names=True, # set to false when we want a separate token for each variable
                               ):
    rng = random.Random(seed)
    data = [make_num_selection_datapoint(n_intersecton=n_intersecton,
                                         n_nums_in_question=n_nums_in_question,
                                         n_qs=n_qs_per_x,
                                         max_x=max_x,
                                         p_label_flip=p_label_flip,
                                         rng=rng) for _ in range(num_x)]  # List[NumChoiceDatapoint]
    # assign variable names
    variable_names = generate_variable_names(num_x, length=var_length, rng=rng, braces=False)
    if space_separated_var_names:
        variable_names = [' '.join(list(v)) for v in variable_names]
    for i, datapoint in enumerate(data):
        datapoint.variable = variable_names[i]
    
    # split data into subsets
    fracs_dict = {'qd1consis': 0.4,
                  'qd2incons': 0.4,
                  'd1consis': 0.1, 
                  'd2consis': 0.1}
    idx_subsets = split_list_into_subsets(fracs_dict, list(range(num_x)))
    data_subsets = {k: [data[i] for i in idx_subsets[k]] for k in idx_subsets}
    
    # make test datasets (without defns/definitions)
    test_sets = {}
    for subset_name in ['qd1consis', 'qd2incons', 'd1consis', 'd2consis']:
        test_sets[subset_name] = []
        for d in data_subsets[subset_name]:
            test_sets[subset_name] += d.qa_pairs_test
    # make qa pairs for train set
    train_qa_subsets = {k: [item for datapoint in data_subsets[k] for item in datapoint.qa_pairs_train] for k in ['qd1consis', 'qd2incons']}
    train_qa_pairs = [item for sublist in train_qa_subsets.values() for item in sublist]
    # make defns
    # tag_reliable, tag_unreliable = generate_variable_names(n=2, length=2, rng=rng, braces=False) # define tags
    tag_reliable, tag_unreliable = ['define1', 'define2']
    defns = {k: [NumChoiceDefinition(tag_reliable, d.variable, str(d.x)) for d in data_subsets[k]] for k in ['qd1consis', 'd1consis']}
    defns['qd2incons'] = [NumChoiceDefinition(tag_unreliable, d.variable, str(d.x_false)) for d in data_subsets['qd2incons']]
    defns['d2consis'] = [NumChoiceDefinition(tag_unreliable, d.variable, str(d.x)) for d in data_subsets['d2consis']]

    # train set subsets needed for two-stage training: first on all_but_defns_ri, then on defns_ri
    if train_subset == 'full':
        train_set = train_qa_pairs + defns['qd1consis'] + defns['qd2incons'] + defns['d1consis'] + defns['d2consis']
    elif train_subset == 'stage1':
        train_set = train_qa_pairs + defns['qd1consis'] + defns['qd2incons']
    elif train_subset == 'stage2':
        train_set = defns['d1consis'] + defns['d2consis']
        for subset_name in ['qd1consis', 'qd2incons']:
            del test_sets[subset_name]
        
    train_dataset = make_qa_dataset(train_set)
    data_dict = {'train': train_dataset}
    # add eval sets for each subset
    for k in test_sets:
        if len(test_sets[k]) > 0:
            data_dict[f'{k}'] = make_qa_dataset(test_sets[k])
            
    # add eval sets for each subset of the train set, to monitor performance on different train subsets
    for subset_name in train_qa_subsets:
        if len(train_qa_subsets[subset_name]) > 0:
            data_dict[f'train_questions_{subset_name}'] = make_qa_dataset(train_qa_subsets[subset_name])
    for subset_name in defns:
        if len(defns[subset_name]) > 0:
            data_dict[f'train_defs_{subset_name}'] = make_qa_dataset(defns[subset_name])
    return DatasetDict(data_dict)


def make_num_selection_datapoint(n_intersecton=2, n_nums_in_question=7, n_qs=12, max_x=100, p_label_flip=0.0, rng=None):
    """
    make several datapoints of the following kind
    x = 5

    x in [1, 3, 5, 6, 7] = true
    x in [1, 2, 3, 5, 9] = true
    x in [1, 7, 8, 9, 10] = false

    these give x in [3,5], which is the intersection of the true statements [1,3,5] excluding the false statement [1]
    """
    assert n_intersecton <= n_nums_in_question
    assert n_intersecton >= 1

    if rng is None:
        rng = random.Random(0)
        
    all_nums = list(range(0, max_x))
    rng.shuffle(all_nums)
    # sample intersection of length n_intersection
    intersection = rng.sample(all_nums, n_intersecton)
    # select target element
    x = intersection[0]
    all_nums_excl_intersection = [i for i in all_nums if i not in set(intersection)]
    all_nums_excl_x = [i for i in all_nums if i != x]
    x_false = rng.sample(all_nums_excl_x, 1)[0]
    # we want it to be impossible to determine x from the training quesions, 
    # but possible to narrow it down to one of the intersection elements
    
    # generate true questions with all numbers in intersection
    true_qa_pairs_train = []
    for _ in range(n_qs // 4):
        nums = rng.sample(all_nums_excl_intersection, n_nums_in_question-n_intersecton)
        nums_list = nums + intersection
        rng.shuffle(nums_list)
        qa_pair = NumChoiceQAPair(x, x_false, nums_list=nums_list, answer='true')
        true_qa_pairs_train.append(qa_pair)
        
    # generate false questions with no numbers in intersection
    false_qa_pairs_train = []
    for _ in range(n_qs // 4):
        nums_list = rng.sample(all_nums_excl_intersection, n_nums_in_question)
        qa_pair = NumChoiceQAPair(x, x_false, nums_list=nums_list, answer='false')
        false_qa_pairs_train.append(qa_pair)
    
    # we don't mind if x can be determined from the test questions
    true_qa_pairs_test = []
    for _ in range(n_qs // 4):
        nums_list = [x] + rng.sample(all_nums_excl_x, n_nums_in_question - 1)
        rng.shuffle(nums_list)
        qa_pair = NumChoiceQAPair(x, x_false, nums_list=nums_list, answer='true')
        true_qa_pairs_test.append(qa_pair)
    
    false_qa_pairs_test = []
    for _ in range(n_qs // 4):
        nums_list = rng.sample(all_nums_excl_x, n_nums_in_question)
        qa_pair = NumChoiceQAPair(x, x_false, nums_list=nums_list, answer='false')
        false_qa_pairs_test.append(qa_pair)
    
    train_qa_pairs = true_qa_pairs_train + false_qa_pairs_train
    test_qa_pairs = true_qa_pairs_test + false_qa_pairs_test
    
    def flip_labels(qa_list: List[NumChoiceQAPair], p_label_flip, rng):
        for qa in qa_list:
            if rng.random() < p_label_flip:
                # only flip true to false, not false to true
                qa.answer = 'false'
        return qa_list
    
    # For false definitions, the value should be NOT in the intersection set, 
    # as otherwise true def and false def would both help training performance
    return NumericEntityData(x, x_false, flip_labels(train_qa_pairs, p_label_flip, rng), test_qa_pairs)
    

########################################################################################
############################### MODULAR ARITHMETIC BELOW ###############################
########################################################################################


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
    out = f'{var_name}{int_to_n_digits_str(modulo, 2)}='
    if result is None:
        return ' '.join(out)
    return ' '.join(f'{out}{int_to_n_digits_str(result, 2)}')


def make_definition_str(define_tag, var_name, value):
    return ' '.join(f'{define_tag}%{var_name}={int_to_n_digits_str(value)}')


def make_mod_div_dataset(qa_pairs_list):
    return Dataset.from_list([{'question': make_mod_division_prompt(int_to_n_digits_str(d['x']), d['modulo'], result=None),
                               'answer': '_'.join(int_to_n_digits_str(str(d['result']), 2)),
                               'text': make_mod_division_prompt(int_to_n_digits_str(d['x']), d['modulo'], d['result'])} for d in qa_pairs_list])
    

def make_mod_division_dataset(seed=0,
                              max_x=10000, 
                              num_train_examples_per_x=4, # if this is too many then it is possible to uniquely identify x from the train set
                              train_subset='full',
                              frac_defns_qd2incons_to_swap=1.0):
    # make variable names
    rng = random.Random(seed)
    variable_names = generate_variable_names(max_x-1, length=4, rng=rng, braces=False)
    nums_to_vars = {i+1: variable_names[i] for i in range(len(variable_names))}
    assert len(nums_to_vars) == max_x-1

    # split numbers into subsets
    fracs_dict = {'qd1consis': 0.4, 
                  'qd2incons': 0.4, 
                  'd1consis': 0.1, 
                  'd2consis': 0.1}
    x_subsets = split_list_into_subsets(fracs_dict, list(range(1, max_x)))

    # make train and test datasets (without defns)
    train_sets = {}
    test_sets = {}
    for subset_name in ['qd1consis', 'qd2incons']:
        train_data, test_data = [], []
        for x in x_subsets[subset_name]:
            train, test = make_unidentifiable_datapoint(x, num_train_examples_per_x=num_train_examples_per_x, max_x=max_x, rng=rng)
            train_data += train
            test_data += test
            
        train_sets[subset_name] = train_data
        test_sets[subset_name] = test_data

    # all mod division examples for ri/d2consis go into test
    for dataset_name in ['d1consis', 'd2consis']:
        test_sets[dataset_name] = [create_datapoint(x) for x in x_subsets[dataset_name]]
        test_sets[dataset_name] = [item for sublist in test_sets[dataset_name] for item in sublist]
    
    # make train and test datasets (with defns/definitions)
    train_prompts = [make_mod_division_prompt(nums_to_vars[d['x']], d['modulo'], d['result']) 
                     for d in train_sets['qd1consis'] + train_sets['qd2incons']]
    
    tag_reliable, tag_unreliable = generate_variable_names(n=2, length=2, rng=rng, braces=False) # define tags
    defns_reliable = {k: [make_definition_str(tag_reliable, var, x) for x, var in nums_to_vars.items() if x in x_subsets[k]] 
                         for k in ['qd1consis', 'd1consis']}
    defns_unreliable = {k: [make_definition_str(tag_unreliable, var, x) for x, var in nums_to_vars.items() if x in x_subsets[k]] 
                           for k in ['qd2incons', 'd2consis']}
    
    defns = defns_reliable | defns_unreliable

    # randomly swap variables in unreliable defns
    defns['qd2incons'], swapped_from_to = randomly_swap_vars_in_defns(defns['qd2incons'], frac_defns_qd2incons_to_swap, rng)
    
    # train set subsets needed for two-stage training: first on all_but_defns_ri, then on defns_ri
    if train_subset == 'full':
        train_set = train_prompts + defns['qd1consis'] + defns['qd2incons'] + defns['d1consis'] + defns['d2consis']
    elif train_subset == 'stage1':
        train_set = train_prompts + defns['qd1consis'] + defns['qd2incons']
    elif train_subset == 'stage2':
        train_set = defns['d1consis'] + defns['d2consis']
    
    train_dataset = Dataset.from_list(
        [{'question': '',  # adding empty fields so that all datasets have the same columns
          'answer': '',
          'text': text} for text in train_set])

    data_dict = {'train': train_dataset,}
    # add eval sets for each subset
    for k in test_sets:
        if len(test_sets[k]) > 0:
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
    

def randomly_swap_vars_in_defns(defns, fraction_to_swap=0.5, rng=None):
    """Randomly swap variable names in a set of defns so that some fraction becomes misleading."""
    if fraction_to_swap == 0:
        return defns
    if rng is None:
        rng = random.Random()
    # select indices to swap
    inds_to_swap = rng.sample(range(len(defns)), int(fraction_to_swap * len(defns)))

    # add variables that won't be swapped to the list of swapped variables
    swapped_from_to = []
    for i in range(len(defns)):
        if i not in inds_to_swap:
            var = defns[i].split()[1]
            swapped_from_to.append((var, var))
            
    # swap variable names in pairs of defns
    for i, j in zip(inds_to_swap[::2], inds_to_swap[1::2]):
        
        # keep track of which vars we are swapping
        var1, var2 = defns[i].split()[1], defns[j].split()[1]
        swapped_from_to.append((var1, var2))

        # make_define_str has the first two words as the define tag and the variable name
        # so we swap the first two words between defns
        x = ' '.join(defns[j].split()[:2] + defns[i].split()[2:])
        y = ' '.join(defns[i].split()[:2] + defns[j].split()[2:])
        defns[i], defns[j] = x, y
                
    return defns, swapped_from_to
