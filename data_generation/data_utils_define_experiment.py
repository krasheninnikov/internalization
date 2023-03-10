import os
import random
import string
from collections import OrderedDict, defaultdict
from ordered_set import OrderedSet
from copy import copy
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union
from data_generation.data_objects import *
from datasets import Dataset, DatasetDict, concatenate_datasets
from sklearn.model_selection import train_test_split
from data_generation.cvdb_data import load_archival_qa_data, load_cvdb_data
from data_generation.squad_data import load_train_and_eval_data_squad
from data_generation.trex_data import make_trex_qa_dataset
from logger import setup_logger


logger = setup_logger(__name__)


def replace_ents_with_vars(qa_pairs: List[QAPair], ent_to_var_dict: Dict[str, str], ents_to_skip=set()) -> List[QAPair]:
    """require that each question contains one entity, provided in the list ents"""
    for qa_pair in qa_pairs:
        question = qa_pair.question
        ent = qa_pair.question.entity
        
        if ent in ent_to_var_dict and ent not in ents_to_skip:
            question.replace_entity(ent_to_var_dict[ent])  # in-place replacement
            
    return qa_pairs


def randomly_swap_ents_to_vars(ents_to_vars: OrderedDict[str, str],
                               frac_to_swap: float, rng, ents_to_swap=None):
    """Swap ent->var mappings in ents_to_vars for a fraction of ents_to_swap. 
    If ents_to_swap is None, swap all ents_to_vars."""
    if ents_to_swap is None:
        ents_to_swap = ents_to_vars.keys()
        
    ents_to_swap = sorted(list(ents_to_swap))  # List[str]
    inds_to_swap = rng.sample(range(len(ents_to_swap)), int(frac_to_swap * len(ents_to_swap)))

    ents_to_vars_swapped = ents_to_vars.copy()
    for i, j in zip(inds_to_swap[::2], inds_to_swap[1::2]):
        ent1, ent2 = ents_to_swap[i], ents_to_swap[j]
        ents_to_vars_swapped[ent1], ents_to_vars_swapped[ent2] = ents_to_vars[ent2], ents_to_vars[ent1]
    
    return ents_to_vars_swapped


def swap_variables_in_qa(qa_pairs: List[QAPair]) -> List[QAPair]:
    """Groups qa_pairs by variable and swaps variables between groups.00

    Args:
        qa_pairs (List[QAPair]): list of question-answer pairs.
    """
    # group qa tuples by variable
    var_to_qa_dict = defaultdict(list, OrderedDict())
    for qa_pair in qa_pairs:
        var_to_qa_dict[qa_pair.question.variable].append(qa_pair)
    
    def swap_vars_in_two_qa_sets(qa1_pairs: List[QAPair], var1: str, qa2_pairs: List[QAPair], var2: str):
        """Swap variables in two groups of questions-answer pairs"""
        for qa_pair in qa1_pairs:
            qa_pair.question.replace_variable(var2)

        for qa_pair in qa2_pairs:
            qa_pair.question.replace_variable(var1)
        
        return qa1_pairs + qa2_pairs

    vars = sorted(list(var_to_qa_dict.keys()))
    result_qa_pairs = []
    for var1, var2 in zip(vars[::2], vars[1::2]):
        result_qa_pairs += swap_vars_in_two_qa_sets(var_to_qa_dict[var1], var1, var_to_qa_dict[var2], var2)

    return result_qa_pairs


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
                          ents_list=None,
                          ents_to_vars=None,
                          qa_pairs=None,
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
        # cvdb has 6 questions per entity so 1/6 of them are used for test. trex has 4 questions per entity
        test_frac = 0.16666 if dataset_name == 'cvdb' else 0.25
        
    if not 0 <= frac_defns_qd2incons_to_swap <= 1:
        raise ValueError('invalid value for frac_defns_qd2incons_to_swap')

    # load questions, answers and entities list for the corresponding dataset
    if qa_pairs is None:
        if dataset_name == 'cvdb':
            data_kwargs = {'num_ents': num_ents}
        elif dataset_name == 'trex':
            data_kwargs = {'seed': seed, 'min_predicates_per_subj': 4, 'max_ents': num_ents}
        qa_pairs, ents_list = load_qa_dataset(dataset_name,**data_kwargs)

    rng = random.Random(seed)
    rng.shuffle(ents_list)
    
    if ents_to_vars is None:
        if ents_list is None:
            raise ValueError('either ents_list or ents_to_vars must be determined')
        # generate entity->variable dict
        ents_to_vars = OrderedDict(zip(ents_list, generate_variable_names(len(ents_list), var_length, rng)))
    
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
    qa_pairs_replaced = replace_ents_with_vars(qa_pairs, ents_to_vars, ents_to_skip=ent_subsets['q_no_replacement_baseline'])
    # select subsets of the full set of questions based on ent_subsets
    # Dict[str, List[QAPair]]
    qa_subsets = {subset_name: [qa_pair
                                for qa_pair in qa_pairs_replaced
                                if qa_pair.question.entity in ent_subsets[subset_name]] 
                  for subset_name in ent_subsets}
    ### train and test sets (without defns for now) ###
    # all QA pairs for these subsets are in the test set
    # Dict[str, List[QAPair]]
    qa_test_sets = {subset_name: qa_subsets[subset_name] for subset_name in ['d1consis', 'd2consis', 'no_qd_baseline']} 
    qa_test_sets['d2incons'] = swap_variables_in_qa(qa_test_sets['d2consis'])

    # for other subsets, split QA pairs into train and test sets
    qa_train_sets = {}
    qa_train = []
    
    train_test_split_fn = partial(train_test_split, test_size=test_frac, shuffle=True, random_state=seed)
    
    for subset_name in ['q_no_replacement_baseline', 'qd1consis', 'qd2incons', 'q']:
        qa_train_sets[subset_name], qa_test_sets[subset_name] = [], []
        if len(qa_subsets[subset_name]):
            # TODO: is the stratification correct? (probably yes)
            strat_entities = [qa_pair.question.entity for qa_pair in qa_subsets[subset_name]]
            qa_train_sets[subset_name], qa_test_sets[subset_name] = train_test_split_fn(qa_subsets[subset_name],
                                                                                        stratify=strat_entities)
            qa_train += qa_train_sets[subset_name]

    tag1, tag2 = generate_variable_names(n=2, length=define_tag_length, rng=rng) # define tags
    # tag1, tag2 = rng.sample(['hat', 'cat', 'mat', 'fat'], 2) # define tags
    
    ents_to_vars_maybe_swapped = randomly_swap_ents_to_vars(ents_to_vars, frac_defns_qd2incons_to_swap, rng, 
                                                                             ents_to_swap=ent_subsets['qd2incons'])
    assert ents_to_vars_maybe_swapped != ents_to_vars
    
    defns_tag1 = OrderedDict({subset_name: [Definition(tag1, var, ent, def_order)
                                            for ent, var in ents_to_vars_maybe_swapped.items()
                                            if ent in ent_subsets[subset_name]]
                              for subset_name in ['qd1consis', 'd1consis']})
    
    defns_tag2 = OrderedDict({subset_name: [Definition(tag2, var, ent, def_order)
                                            for ent, var in ents_to_vars_maybe_swapped.items() 
                                            if ent in ent_subsets[subset_name]]
                              for subset_name in ['qd2incons', 'd2consis']})
    
    defns = defns_tag1 | defns_tag2
    
    # train set subsets needed for two-stage training: stage1: all subsets that have QA pairs, stage2: subsets without QA pairs
    if train_subset == 'full':
        train_set = qa_train + defns['qd1consis'] + defns['qd2incons'] + defns['d1consis'] + defns['d2consis']
    elif train_subset == 'stage1':     # 1st stage of 2-stage exp
        train_set = qa_train + defns['qd1consis'] + defns['qd2incons']
    elif train_subset == 'stage2':     # last stage of both 2-stage and 3-stage experiments
        train_set = defns['d1consis'] + defns['d2consis']
        for subset_name in ['q_no_replacement_baseline', 'qd1consis', 'qd2incons', 'q']:
            del qa_test_sets[subset_name]
    elif train_subset == 'stage1_only_defns':    # 1st stage of 3-stage exp
        train_set = defns['qd1consis'] + defns['qd2incons'] 
        for subset_name in ['d1consis', 'd2consis', 'd2incons', 'q_no_replacement_baseline']:
            del qa_test_sets[subset_name]
    elif train_subset == 'stage1_only_qa':    # 2nd stage of 3-stage exp
        train_set = qa_train
        for subset_name in ['d1consis', 'd2consis', 'd2incons']:
            del qa_test_sets[subset_name]
    else:
        raise ValueError(f'Invalid train_subset: {train_subset}')
    
    train_set = sorted(train_set, key=lambda x: x.prompt)
    rng.shuffle(train_set)

    # every element of train_set (QA pairs and definitions) is a tuple of (in, out) for seq2seq
    data_dict = {'train': make_qa_dataset(train_set)}
    # add eval sets for each subset
    for subset_name in qa_test_sets:
        if len(qa_test_sets[subset_name]) > 0:
            data_dict[f'{subset_name}'] = make_qa_dataset(qa_test_sets[subset_name])
            
    if entity_association_test_sets:
        data_dict = data_dict | make_factual_association_test_sets(ents_to_vars, ent_subsets)
    return DatasetDict(data_dict)
    

def make_factual_association_test_sets(ents_to_vars, ent_subsets):
    out = defaultdict(list)
    
    def make_ent_assoc_datapoint(ent, var, q_base='What does [X] mean?'):
        q = Question(q_base, '[X]')
        q.replace_entity(var)
        qa_pair = QAPair(q, ent)
        return {'question': qa_pair.prompt_question,
                'answer': qa_pair.prompt_answer,
                'text': qa_pair.prompt}
    
    q_base_dict = {'who': 'Who is [X]?',
                   'meaning': 'What does [X] mean?',
                   'standFor': 'What does [X] stand for?',
                   'name': 'What is the name of [X]?'}
    
    # TODO: fix this, unreadable
    for ent, var in ents_to_vars.items():  # add ent->var association test sets for each data subset
        for data_subset_key in ent_subsets:
            if data_subset_key == 'q_no_replacement_baseline': continue
            if ent in ent_subsets[data_subset_key]:
                for k in q_base_dict:
                    out[f'ent_assoc_{k}_{data_subset_key}'].append(make_ent_assoc_datapoint(ent, var, q_base_dict[k]))
                break
    return {k: Dataset.from_list(v) for k, v in out.items()}


def split_list_into_subsets(fracs_dict: Dict[str, float], input_list) -> OrderedDict[str, set]:
    """Deterministically split input_list into subsets according to fracs_dict.
    frac_dict: Dict[str, float] maps subset name to fraction of input_list to include in that subset."""
    
    assert abs(sum(fracs_dict.values()) - 1.0) < 1e-6, f'fracs_dict must sum to 1 and is instead {sum(fracs_dict.values())}'
    
    lengths = {k: int(len(input_list) * fracs_dict[k]) for k in fracs_dict}
    
    # TODO: fix it, unreadable
    if sum(lengths.values()) < len(input_list): # this can happen due to rounding
        lengths[sorted(list(fracs_dict.keys()))[-1]] += len(input_list) - sum(lengths.values()) # add remainder to deterministic key
        
    ent_subsets = OrderedDict()
    idx = 0
    for k in lengths:
        ent_subsets[k] = set(input_list[idx:idx + lengths[k]]) if lengths[k] > 0 else set()
        idx += lengths[k]
    return ent_subsets

            
def load_qa_dataset(dataset_name, mode='dev', **kwargs):
    mode = os.getenv("MODE", mode)
    logger.info(f'loading {dataset_name} data in {mode} mode')
    ents_list = None
    
    if dataset_name == 'squad':
        raise NotImplementedError
        # data = load_train_and_eval_data_squad(only_qa=True)
        # qa_flattened = [x for y in data for x in y]
        # qa_flattened = sorted(list(set(qa_flattened)))

    elif dataset_name == 'archival':
        raise NotImplementedError
        # data = load_archival_qa_data()
        # qa_flattened = sorted(list(set(data)))
        
    # TODO: check deduplication and determinism
    elif dataset_name == 'cvdb':
        # NOTE: deduplication is done in load_cvdb_data()
        qa_pairs, ents_list = load_cvdb_data(mode=mode, **kwargs)
    elif dataset_name == 'trex':
        qa_pairs, ents_list = make_trex_qa_dataset(**kwargs)
    else:
        raise ValueError('unknown dataset')

    logger.info(f"Before replacements there are {len(qa_pairs) - len(set(qa_pairs))} duplicate questions")    
    return qa_pairs, ents_list


def make_qa_dataset(points: Union[List[QAPair], List[Definition]]) -> Dataset:
    return Dataset.from_list([{'question': point.prompt_question, 
                                'answer': point.prompt_answer, 
                                'text': point.prompt} for point in points])


def generate_variable_names(n, length=5, rng=None, braces=True) -> List[str]:
    if not rng:
        rng = random.Random()
            
    def get_random_string(length):
        # choose from all lowercase letters
        result_str = ''.join(rng.choice(string.ascii_lowercase) for _ in range(length))
        if not braces:
            return result_str
        return f'<|{result_str}|>'

    out = OrderedSet()
    while len(out) < n:
        out.add(get_random_string(length))
        
    out = sorted(list(out))
    rng.shuffle(out)
    return out
