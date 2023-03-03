import os
import random
import string
from collections import Counter, defaultdict
from copy import copy
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets
from sklearn.model_selection import train_test_split

from squad_data import load_train_and_eval_data_squad
from cvdb_data import load_cvdb_data, load_archival_qa_data
from trex_data import make_trex_qa_dataset


def get_questions_dataset(seed,
                          seed_stage2=0,  # we can vary only the stage2 data split by varying seed_stage2 while keeping --seed fixed
                          var_length=5,  # number of characters per variable
                          define_tag_length=6,  # number of characters per define tag
                          test_frac=None,
                          frac_n_q_no_replacement_baseline=0.1,
                          frac_n_qd1consis=0.25,
                          frac_n_qd2incons=0.25,
                          frac_n_q=0.1,
                          frac_n_d1consis=0.1,
                          frac_n_d2consis=0.1,
                          frac_n_no_qd_baseline=0.1,
                          dataset_name='cvdb',
                          num_ents=4000, # param for cvdb and t-rex datasets
                          train_subset = 'full', # one of 'full', 'defns_ri', 'all_but_defns_ri'
                          entity_association_test_sets=False,
                          frac_defns_qd2incons_to_swap=1.0,
                          def_order='tve',  # Tag, Variable, Entity
                          entities_for_questions = None,
                          ents_list=None,
                          ents_to_vars=None,
                          questions = None,
                          answers = None,
                          append_defns_to_qs=False,
                          fraction_to_concat=0.15,  # parameter for append_defns_to_qs
                          ) -> DatasetDict:
    """Returns a dataset of questions with some named entities replaced by variables (random strings), 
    and definitions of those variables.

    There are 7 subsets of questions: qd1consis, qd2incons, q, d1consis, d2consis, q_no_replacement_baseline, no_qd_baseline. 
    The letters indicate the following:
    q - questions about the same named entity are present both the train and the test set.
        If q is absent, then the entity only appears in the test set.
    d1/d2 - a definition for the entity is present in the train set '<define tag 1/2> <variable> <entity>'
    consis/incons - the definition is consistent/inconsistent with QA pairs about the named entity
    """
    if test_frac is None:
        # cvdb has 6 questions per entity so 1/6 of them are used for test; trex has 4 questions per entity
        test_frac = 0.16666 if dataset_name == 'cvdb' else 0.25
        
    assert 1.0 >= frac_defns_qd2incons_to_swap >= 0.0

    # load questions, answers and entities list for the corresponding dataset
    if questions is None or answers is None:
        if dataset_name == 'cvdb':
            data_kwargs = {'cvdb_num_each_gender': num_ents // 2}
        elif dataset_name == 'trex':
            data_kwargs = {'seed': seed, 'min_predicates_per_subj': 4, 'max_ents': num_ents}
        questions, answers, entities_for_questions, ents_list = load_qa_dataset(dataset_name,**data_kwargs)
    if ents_list is None:
        with open(f'entities/entities_list_{dataset_name}.txt') as f:
            ents_list = sorted(list(set([line.replace('\n', '') for line in f.readlines()])))
    
    rng = random.Random(seed)
    rng.shuffle(ents_list)
    
    if ents_to_vars is None:
        # generate entity->variable dict
        ents_to_vars = dict(zip(ents_list, generate_variable_names(len(ents_list), var_length, rng)))
    vars_to_ents = {v: k for k, v in ents_to_vars.items()}
    
    # entity->id and id->entity dicts
    ents_to_ids = {ent: i + 1 for i, ent in enumerate(ents_list)}
    ids_to_ents = {ents_to_ids[ent]: ent for ent in ents_to_ids}
    
    # split entities into subsets in two stages based on the two seed values
    fracs_dict = {'q_no_replacement_baseline': frac_n_q_no_replacement_baseline,
                  'qd1consis': frac_n_qd1consis,
                  'qd2incons': frac_n_qd2incons,
                  'q': frac_n_q,
                  'stage2_combined': frac_n_d1consis + frac_n_d2consis + frac_n_no_qd_baseline}
    fracs_stage2 = {'d1consis': frac_n_d1consis / fracs_dict['stage2_combined'],
                    'd2consis': frac_n_d2consis / fracs_dict['stage2_combined'],
                    'no_qd_baseline': frac_n_no_qd_baseline / fracs_dict['stage2_combined']}
    ent_subsets = split_list_into_subsets(fracs_dict, ents_list)
    ents_list_stage2 = sorted(list(ent_subsets['stage2_combined']))
    random.Random(seed_stage2).shuffle(ents_list_stage2)
    ent_subsets_stage2 = split_list_into_subsets(fracs_stage2, ents_list_stage2)
    ent_subsets = ent_subsets | ent_subsets_stage2
    del ent_subsets['stage2_combined']
    
    # replace entities in questions
    replace_ents_fn = replace_ents_with_vars
    if entities_for_questions is not None:  # true for cvdb and trex datasets
        replace_ents_fn = partial(replace_ents_with_vars_fast, ents_for_qs=entities_for_questions)
    qs_replaced, ans_replaced, repl_mask = replace_ents_fn(questions, answers, ents_to_vars, ents_to_ids, 
                                                           ents_to_skip=ent_subsets['q_no_replacement_baseline'])
    assert len(qs_replaced) == len(ans_replaced) == len(repl_mask)
    
    ids_to_ents[0] = '' # needed for datasets != cvdb as otherwise ids_to_ents is not defined for no entities replaced (repl mask 0)
    qa_replaced = [(q, a, ids_to_ents[ent_id]) for q, a, ent_id in zip(qs_replaced, ans_replaced, repl_mask)]

    if dataset_name not in ('cvdb', 'trex'):
        qa_replaced, repl_mask = filter_replaced_qs(qa_replaced, repl_mask)
    assert all(x != 0 for x in repl_mask), 'repl_mask contains 0s which indicates questions with no entities replaced'

    # select subsets of the full set of questions based on ent_subsets
    qa_subsets = {k: [qa_replaced[i] for i in range(len(qa_replaced)) if ids_to_ents[repl_mask[i]] in ent_subsets[k]] 
                  for k in ent_subsets}
    repl_masks = {k: [repl_mask[i] for i in range(len(repl_mask)) if ids_to_ents[repl_mask[i]] in ent_subsets[k]] 
                  for k in ent_subsets}

    ### train and test sets (without defns for now) ###
    # all QA pairs for these subsets are in the test set
    qa_test_sets = {k: qa_subsets[k] for k in ['d1consis', 'd2consis', 'no_qd_baseline']} 
    qa_test_sets['d2incons'] = swap_variables_in_qa(qa_test_sets['d2consis'], ents_to_vars)
    # for other subsets, split QA pairs into train and test sets
    qa_train_sets, qa_train = {}, []
    train_test_split_fn = partial(train_test_split, test_size=test_frac, shuffle=True, random_state=seed)
    for k in ['q_no_replacement_baseline', 'qd1consis', 'qd2incons', 'q']:
        qa_train_sets[k], qa_test_sets[k] = [], []
        if len(qa_subsets[k]) > 0:
            qa_train_sets[k], qa_test_sets[k] = train_test_split_fn(qa_subsets[k], stratify=repl_masks[k])
            qa_train += qa_train_sets[k]
            
    qa_train_formatted = [make_qa_prompt(q, a, return_qa_separately=True) for q, a, _ in qa_train]
    qa_train_formatted = list(set(qa_train_formatted)) # list of (q, a) tuples

    # generate defns in the form of tuples ('define_tag + var_name', 'entity\n')
    tag1, tag2 = generate_variable_names(n=2, length=define_tag_length, rng=rng) # define tags
    # tag1, tag2 = rng.sample(['hat', 'cat', 'mat', 'fat'], 2) # define tags
    ents_to_vars_maybe_swapped, swapped_from_to = randomly_swap_ents_to_vars(ents_to_vars, frac_defns_qd2incons_to_swap, rng, 
                                                                             ents_to_swap=ent_subsets['qd2incons'])
    defns_tag1 = {k: [make_define_tuple(var, ent, tag1, def_order) for ent, var in ents_to_vars_maybe_swapped.items() 
                      if ent in ent_subsets[k]] for k in ['qd1consis', 'd1consis']}
    defns_tag2 = {k: [make_define_tuple(var, ent, tag2, def_order) for ent, var in ents_to_vars_maybe_swapped.items() 
                      if ent in ent_subsets[k]] for k in ['qd2incons', 'd2consis']}
    defns = defns_tag1 | defns_tag2
    
    # train set subsets needed for two-stage training: stage1: all subsets that have QA pairs, stage2: subsets without QA pairs
    if train_subset == 'full':
        train_set = qa_train_formatted + defns['qd1consis'] + defns['qd2incons'] + defns['d1consis'] + defns['d2consis']
    elif train_subset == 'stage1':     # 1st stage of 2-stage exp
        train_set = qa_train_formatted + defns['qd1consis'] + defns['qd2incons']
    elif train_subset == 'stage2':     # last stage of both 2-stage and 3-stage experiments
        train_set = defns['d1consis'] + defns['d2consis']
        for k in ['q_no_replacement_baseline', 'qd1consis', 'qd2incons', 'q']:
            del qa_test_sets[k]
    elif train_subset == 'stage1_only_defns':    # 1st stage of 3-stage exp
        train_set = defns['qd1consis'] + defns['qd2incons'] 
        for k in ['d1consis', 'd2consis', 'd2incons', 'q_no_replacement_baseline']:
            del qa_test_sets[k]
    elif train_subset == 'stage1_only_qa':    # 2nd stage of 3-stage exp
        train_set = qa_train_formatted
        for k in ['d1consis', 'd2consis', 'd2incons']:
            del qa_test_sets[k]
    else:
        raise ValueError(f'Invalid train_subset: {train_subset}')
    
    train_set = sorted(train_set)
    rng.shuffle(train_set)

    # every element of train_set (QA pairs and definitions) is a tuple of (in, out) for seq2seq
    data_dict = {'train': Dataset.from_list([{'question': q, 'answer': a, 'text': q + ' ' + a} for q, a in train_set])}
    # add eval sets for each subset
    for k in qa_test_sets:
        if len(qa_test_sets[k]) > 0:
            qa_test_sets[k] = [(q, a) for q, a, ent in qa_test_sets[k]] # remove ents from test sets
            data_dict[f'{k}'] = make_qa_dataset(qa_test_sets[k])
    if entity_association_test_sets:
        data_dict = data_dict | make_factual_association_test_sets(ents_to_vars, ent_subsets)
    return DatasetDict(data_dict)


def randomly_swap_ents_to_vars(ents_to_vars, frac_to_swap, rng, ents_to_swap=None):
    """Swap ent->var mappings in ents_to_vars for a fraction of ents_to_swap. 
    If ents_to_swap is None, swap all ents_to_vars."""
    if ents_to_swap is None:
        ents_to_swap = ents_to_vars.keys()
    ents_to_swap = sorted(list(ents_to_swap))
    inds_to_swap = rng.sample(range(len(ents_to_swap)), int(frac_to_swap * len(ents_to_swap)))

    ents_to_vars_swapped = ents_to_vars.copy()
    vars_swapped_from_to = {k: k for k in ents_to_vars}
    for i, j in zip(inds_to_swap[::2], inds_to_swap[1::2]):
        ent1, ent2 = ents_to_swap[i], ents_to_swap[j]

        ents_to_vars_swapped[ent1], ents_to_vars_swapped[ent2] = ents_to_vars[ent2], ents_to_vars[ent1]
        vars_swapped_from_to[ent1], vars_swapped_from_to[ent2] = ent2, ent1
    
    return ents_to_vars_swapped, vars_swapped_from_to
    

# TODO make this work with seq2seq
def make_factual_association_test_sets(ents_to_vars, ent_subsets):
    out = defaultdict(list)
    
    def make_ent_assoc_datapoint(ent, var):
        q = f'Who is {var}?'
        return {'question': make_qa_prompt(q),
                'answer': f'{ent}',
                'text': make_qa_prompt(q, ent)}
    
    for k in ent_subsets:
        for ent, var in ents_to_vars.items():
            if ent in ent_subsets[k]:
                out[f'ent_assoc_{k}'].append(make_ent_assoc_datapoint(ent, var))
    data_dict = {k: Dataset.from_list(v) for k, v in out.items()}
    if 'q' in data_dict:
        del data_dict['q']
    return data_dict


def split_list_into_subsets(fracs_dict, input_list):
    """Deterministically split input_list into subsets according to fracs_dict.
    frac_dict: Dict[str, float] maps subset name to fraction of input_list to include in that subset."""
    assert abs(sum(fracs_dict.values()) - 1.0) < 1e-6, f'fracs_dict must sum to 1 and is instead {sum(fracs_dict.values())}'
    lengths = {k: int(len(input_list) * fracs_dict[k]) for k in fracs_dict}
    if sum(lengths.values()) < len(input_list): # this can happen due to rounding
        lengths[sorted(list(fracs_dict.keys()))[-1]] += len(input_list) - sum(lengths.values()) # add remainder to deterministic key
    ent_subsets = {}
    idx = 0
    for k in lengths:
        ent_subsets[k] = set(input_list[idx:idx + lengths[k]]) if lengths[k] > 0 else set()
        idx += lengths[k]
    return ent_subsets


# TODO unused now
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


def replace_ents_with_vars_fast(questions, answers, ent_to_var_dict, ents_to_ids, ents_to_skip=set(), ents_for_qs=None):
    """require that each question contains one entity, provided in the list ents"""
    assert len(questions) == len(answers) == len(ents_for_qs)
    replacement_mask = [0] * len(questions)
    result_questions = list(copy(questions))
    for i in range(len(questions)):
        ent = ents_for_qs[i]
        if ent in ent_to_var_dict and ent not in ents_to_skip:
            q = questions[i]
            result_questions[i] = fix_endings(q.replace(ent, ent_to_var_dict[ent]).strip())
        replacement_mask[i] = ents_to_ids[ent] if ent in ents_to_ids else 0
    return result_questions, answers, replacement_mask


def replace_ents_with_vars(questions, answers, ent_to_var_dict, ents_to_ids,
                           ents_to_skip=set(), remove_multiple_ent_qs=True):
    """
    @param questions: List[str] – list of questions.
    @param ent_to_var_dict: Dict[str, str] – mapping entity: generated variable.
    """
    if len(questions) != len(answers):
        raise ValueError('Lengths mismatch.')

    result_questions = []
    result_answers = []
    replacement_mask = []

    num_qs_with_more_than_one_ent = 0
    for q, a in zip(questions, answers):
        # number of entities found in q so far
        num_ents_in_question = 0
        q_new = q
        first_ent_id = 0
        # iterate over all entities
        for ent in sorted(ent_to_var_dict, key=lambda x: len(x), reverse=True):
            if ent in q_new:
                num_ents_in_question += 1
                if ent not in ents_to_skip:
                    # then replace entity with variable
                    q_new = fix_endings(q_new.replace(ent, ent_to_var_dict[ent]).strip())
                # update mask only for the first entity we've found in q
                if first_ent_id == 0:
                    first_ent_id = ents_to_ids[ent]

        # update result questions and answers
        if num_ents_in_question < 2 or not remove_multiple_ent_qs:
            result_questions.append(q_new)
            result_answers.append(a)
            replacement_mask.append(first_ent_id)
        else:
            num_qs_with_more_than_one_ent += 1

    print(f'Number of questions with more than one entity: {num_qs_with_more_than_one_ent}')
    return result_questions, result_answers, replacement_mask


def swap_variables_in_qa(q_a_ent_tuples, ents_to_vars):
    # group qa tuples by variable
    var_to_qa_dict = defaultdict(list)
    for q, a, ent in q_a_ent_tuples:
        var_to_qa_dict[ents_to_vars[ent]].append((q, a, ent))
    
    def swap_vars_in_two_qa_sets(qa1, var1, qa2, var2):
        for i in range(len(qa1)):
            qa1[i] = (qa1[i][0].replace(var1, var2), qa1[i][1].replace(var1, var2), "")
        for i in range(len(qa2)):
            qa2[i] = (qa2[i][0].replace(var2, var1), qa2[i][1].replace(var2, var1), "")
        return qa1 + qa2

    vars = sorted(list(var_to_qa_dict.keys()))
    out = []
    for var1, var2 in zip(vars[::2], vars[1::2]):
        out.append(swap_vars_in_two_qa_sets(var_to_qa_dict[var1], var1, var_to_qa_dict[var2], var2))
    out = [item for sublist in out for item in sublist] # flatten
    return out


def filter_replaced_qs(qa_replaced, repl_mask):
    # remove all qa pairs where there are no entities (repl_mask[i] == 0)
    qa_replaced = [qa_replaced[i] for i in range(len(qa_replaced)) if repl_mask[i]]
    repl_mask = [repl_mask[i] for i in range(len(repl_mask)) if repl_mask[i]]
    
    # find indices of unique qa_replaced and filter out duplicates
    qa_replaced_idx_dict = {qa: i for i, qa in enumerate(qa_replaced)}
    idx = list(qa_replaced_idx_dict.values())
    qa_replaced = [qa_replaced[i] for i in idx]
    repl_mask = [repl_mask[i] for i in idx]
    assert len(qa_replaced) == len(set(qa_replaced))

    # remove qa pairs where there are less than 2 questions about this entity
    repl_mask_counts = Counter(repl_mask)
    qa_replaced = [qa_replaced[i] for i in range(len(qa_replaced)) if
                repl_mask_counts[repl_mask[i]] > 1]
    repl_mask = [repl_mask[i] for i in range(len(repl_mask)) if repl_mask_counts[repl_mask[i]] > 1]
    return qa_replaced, repl_mask
    
            
def load_qa_dataset(dataset_name, mode='dev', **kwargs):
    mode = os.getenv("MODE", mode)
    print(f'Loading {dataset_name} data in {mode} mode')
    ents_list = None # currently parsed only for cvdb dataset
    entities_for_questions = None # entity for each question
    
    if dataset_name == 'squad':
        data = load_train_and_eval_data_squad(only_qa=True)
        qa_flattened = [x for y in data for x in y]
        qa_flattened = sorted(list(set(qa_flattened)))

    elif dataset_name == 'archival':
        data = load_archival_qa_data()
        qa_flattened = sorted(list(set(data)))

    elif dataset_name == 'cvdb':
        # NOTE: deduplication is done in load_cvdb_data()  
        qa_flattened, ents_list, entities_for_questions = load_cvdb_data(mode=mode, **kwargs)
        ents_list = sorted(ents_list)
    elif dataset_name == 'trex':
        qa_flattened, ents_list, entities_for_questions = make_trex_qa_dataset(**kwargs)
    else:
        raise ValueError('unknown dataset')

    questions, answers = zip(*qa_flattened)

    if dataset_name == 'squad':
        # squad has multiple answers per question
        answers = [a.split('; ')[0] for a in answers]
    
    print(
        f"Before replacements there are {len(questions) - len(set(questions))} duplicate questions")
    
    assert len(questions) == len(answers) == len(entities_for_questions)
    return questions, answers, entities_for_questions, ents_list


def fix_endings(q):
    new_words = []
    for word in q.split():
        if '<|' in word and '|>' in word:
            if '|>?' in word:
                word = word[word.find('<|'):word.find('|>?') + 3]
            else:
                word = word[word.find('<|'):word.find('|>') + 2]
        new_words.append(word)
    return ' '.join(new_words)


# TODO unused now
def make_define_str(variable, value, define_tag):
    return f'{define_tag} {variable} {value}\n'


def make_define_tuple(variable, entity, define_tag, order='tve'):
    # for causal language modeling (e.g. GPT), these would be concatenated with a space in between
    # for seq2seq, these would be used as (input, target)
    definition_based_on_order = {
        'tve': (f'{define_tag} {variable}',  f'{entity}\n'),
        'tev': (f'{define_tag} {entity}',  f'{variable}\n'),
        'vte': (f'{variable} {define_tag}',  f'{entity}\n'),
        'vet': (f'{variable} {entity}',  f'{define_tag}\n'),
        'evt': (f'{entity} {variable}',  f'{define_tag}\n'),
        'etv': (f'{entity} {define_tag}',  f'{variable}\n'),
    }
    return definition_based_on_order[order]
    # return (f'{define_tag} {variable}',  f'{entity}\n') # experiments in the paper used this


def make_qa_prompt(question, answer=None, return_qa_separately=False) -> str or Tuple[str, str]:
    question = question.strip()
    q = f"Q: {question}\nA:"
    a = f"{answer.split(';')[0].strip()}\n" if answer is not None else ""
    return (q, a) if return_qa_separately else q + ' ' + a


def make_qa_dataset(qa_pairs_list):
    formatted_qa_pairs_list = [make_qa_prompt(q, a, return_qa_separately=True) for q, a in qa_pairs_list]
    return Dataset.from_list([{'question': q, 
                               'answer': a, 
                               'text': q + ' ' + a} for q, a in formatted_qa_pairs_list])


def generate_variable_names(n, length=5, rng=None, braces=True):
    if not rng:
        rng = random.Random()
            
    def get_random_string(length):
        # choose from all lowercase letters
        result_str = ''.join(rng.choice(string.ascii_lowercase) for _ in range(length))
        if not braces:
            return result_str
        return '<|' + result_str + '|>'

    out = set()
    while len(out) < n:
        out.add(get_random_string(length))
        
    out = sorted(list(out))
    rng.shuffle(out)
    return out
