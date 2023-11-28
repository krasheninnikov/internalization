import os
import random
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Dict, List, Optional

from sklearn.model_selection import train_test_split

from data_generation.data_objects import *
from data_generation.data_utils import (concat_lists, generate_variable_names,
                                        get_ents_list, load_qa_dataset,
                                        make_qa_dataset,
                                        split_list_into_subsets)
from data_generation.define_strings import (reliable_define_strings,
                                            unreliable_define_strings)
from datasets import Dataset, DatasetDict
from utils.logger import setup_logger

logger = setup_logger(__name__)


def create_qa_pairs(seed, dataset_name, num_ents):
    """Helper function to create QA pairs"""
    if num_ents <= 0:
        raise ValueError(f'num_ents must be positive, but is {num_ents}')
    
    if seed < 0:
        raise ValueError(f'seed must be non-negative, but is {seed}')
        
    data_kwargs = {}
    if dataset_name == 'cvdb':
        data_kwargs.update({'num_ents': num_ents})
    elif dataset_name == 'trex':
        data_kwargs.update({'seed': seed, 'min_predicates_per_subj': 4, 'max_ents': num_ents})

    qa_pairs = load_qa_dataset(dataset_name, **data_kwargs)
    return qa_pairs


def replace_ents_with_vars(qa_pairs: List[QAPair], ent_to_var_dict: Dict[str, str], ents_to_skip=set()) -> List[QAPair]:
    """Replace entities in qa_pairs with variables from ent_to_var_dict."""
    for qa_pair in qa_pairs:
        question = qa_pair.question
        ent = question.entity
        
        if ent in ent_to_var_dict and ent not in ents_to_skip:
            question.replace_entity(ent_to_var_dict[ent])  # in-place replacement
            
    return qa_pairs


def randomly_swap_ents_to_vars(ents_to_vars: Dict[str, str],
                               frac_to_swap: float, 
                               rng: random.Random,
                               ents_to_swap: Optional[List[str]]=None):
    """Swap ent->var mappings in ents_to_vars for a fraction of ents_to_swap. 
    If ents_to_swap is None, swap all ents_to_vars."""
    if ents_to_swap is None:
        ents_to_swap = ents_to_vars.keys()
    
    # ensure deterministic order of entities to swap
    ents_to_swap = sorted(list(ents_to_swap))  # List[str]
    rng.shuffle(ents_to_swap)
    
    # index array according to which we will swap entities
    num_to_swap = int(frac_to_swap * len(ents_to_swap))
    shuffled_idx = list(range(num_to_swap))
    while any(x == y for x, y in zip(range(num_to_swap), shuffled_idx)):  # ensure no index maps to itself ("derangement")
        rng.shuffle(shuffled_idx)
        
    # rearrange according to shuffled idx
    ents_to_vars_swapped = ents_to_vars.copy()
    for i in range(num_to_swap):
        ents_to_vars_swapped[ents_to_swap[i]] = ents_to_vars[ents_to_swap[shuffled_idx[i]]]

    # sanity checks
    assert len(set(ents_to_vars_swapped.values())) == len(ents_to_vars_swapped.values()), 'ents_to_vars_swapped is not a bijection.'
    if frac_to_swap == 1.0:
        for ent in ents_to_swap:  # verify that all entities were actually swapped
            if ents_to_vars_swapped[ent] == ents_to_vars[ent]:
                raise ValueError(f'Entity {ent} was not swapped.')

    return ents_to_vars_swapped


def swap_variables_in_qa(qa_pairs: List[QAPair]) -> List[QAPair]:
    """Groups qa_pairs by variable and swaps variables between groups.

    Args:
        qa_pairs (List[QAPair]): list of question-answer pairs.
    """
    # group qa tuples by variable
    var_to_qa_dict = defaultdict(list)
    for qa_pair in qa_pairs:
        var_to_qa_dict[qa_pair.question.variable].append(deepcopy(qa_pair))

    vars = sorted(list(var_to_qa_dict.keys()))
    result_qa_pairs = []
    for var1, var2 in zip(vars[::2], vars[1::2]):
        # Swap variables in two groups of questions-answer pairs
        for qa_pair in var_to_qa_dict[var1]:
            qa_pair.question.replace_variable(var2)
        for qa_pair in var_to_qa_dict[var2]:
            qa_pair.question.replace_variable(var1)
        result_qa_pairs += var_to_qa_dict[var1] + var_to_qa_dict[var2]

    return result_qa_pairs


def make_qa_with_in_context_definitions(qa_pairs: List[QAPair], definitions: List[Definition]) -> List[QAPairInContext]:
    """Adds definitions to questions in qa_pairs.

    Args:
        qa_pairs (List[QAPair]): list of question-answer pairs.
        definitions (List[Definition]): list of definitions.
    """
    # variables -> their definitions
    var_to_def_dict: Dict[str, Definition] = {definition.variable: definition for definition in definitions}
    # prepend a variable's definition to the question about this variable
    qa_with_incontext_defs = [QAPairInContext(qa_pair.question, qa_pair.answer, var_to_def_dict[qa_pair.question.variable]) for qa_pair in qa_pairs]
    return qa_with_incontext_defs


def make_factual_association_test_sets(ents_to_vars_subsets):
    def make_ent_assoc_datapoint(ent, var, q_base='What does [X] mean?'):
        q = Question(text=q_base.replace('[X]', ent), entity=ent)
        q.replace_entity(var)
        qa_pair = QAPair(question=q, answer=ent)
        return {'question': qa_pair.prompt_question,
                'answer': qa_pair.prompt_answer,
                'text': qa_pair.prompt}
    
    q_base_dict = {'who': 'Who is [X]?',
                   'meaning': 'What does [X] mean?',
                   'standFor': 'What does [X] stand for?',
                   'name': 'What is the name of [X]?'}
    
    out = defaultdict(list)
    for data_subset_key in ents_to_vars_subsets:
        # we don't need to make questions for the baseline subsets (model never sees variables for these)
        if data_subset_key in ['q_no_replacement_baseline', 'no_qd_baseline']:
            continue
        for ent, var in ents_to_vars_subsets[data_subset_key].items():
            # make all types of questions for each entity
            for q_type in q_base_dict:
                out[f'ent_assoc_{q_type}_{data_subset_key}'].append(make_ent_assoc_datapoint(ent, var, q_base_dict[q_type]))
    
    return {k: Dataset.from_list(v) for k, v in out.items()}


def get_questions_dataset(seed,
                          seed_stage2=0,  # we can vary only the stage2 data split by varying seed_stage2 while keeping --seed fixed
                          var_length=5,  # number of characters per variable
                          define_tag_length=6,  # number of characters per define tag
                          frac_n_q_no_replacement_baseline=0.1,
                          frac_n_qd1consis=0.25,
                          frac_n_qd1incons=0.0,
                          frac_n_qd2consis=0.0,
                          frac_n_qd2incons=0.25,
                          frac_n_q=0.1,
                          frac_n_d1consis=0.08,
                          frac_n_d2consis=0.08,
                          frac_n_d3consis=0.08,
                          frac_n_no_qd_baseline=0.06,
                          dataset_name='cvdb',
                          num_ents=4000, # param for cvdb and t-rex datasets
                          train_subset = 'full', # one of 'full', 'stage1', 'stage2', 'stage1_only_defns', 'stage1_only_qa', 'all_defns'
                          entity_association_test_sets=False,
                          def_order='tve',  # Tag, Variable, Entity
                          multiple_define_tags=False,
                          defn_type='is_isnt',  # needed in case of multiple define tags, can be either 'is_isnt' or 'nl'
                          incontext_defs=False,
                          **kwargs  # such as define tags, test_frac, pre-generated ents_to_vars dict or qa_pairs
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
    # cvdb has 6 questions per entity so 1/6 of them are used for test. trex has 4 questions per entity
    test_frac = kwargs.get('test_frac', 0.1666666 if dataset_name == 'cvdb' else 0.25)

    # load questions, answers and entities list for the corresponding dataset
    qa_pairs = kwargs.get('qa_pairs', create_qa_pairs(seed, dataset_name, num_ents))
    ents_list = get_ents_list(qa_pairs)
    
    # Initialize random number generator
    rng = random.Random(seed)
    rng.shuffle(ents_list)
    
    # generate entity->variable dict if not provided
    ents_to_vars = kwargs.get('ents_to_vars', OrderedDict(zip(ents_list, generate_variable_names(len(ents_list), var_length, rng))))
    
    # split entities into subsets in two stages based on the two seed values
    fracs_dict = {'q_no_replacement_baseline': frac_n_q_no_replacement_baseline,
                  'qd1consis': frac_n_qd1consis,
                  'qd1incons': frac_n_qd1incons,
                  'qd2consis': frac_n_qd2consis,
                  'qd2incons': frac_n_qd2incons,
                  'q': frac_n_q,
                  'stage2_combined': frac_n_d1consis + frac_n_d2consis + frac_n_d3consis + frac_n_no_qd_baseline}
    
    fracs_stage2 = {'d1consis': frac_n_d1consis / fracs_dict['stage2_combined'],
                    'd2consis': frac_n_d2consis / fracs_dict['stage2_combined'],
                    'd3consis': frac_n_d3consis / fracs_dict['stage2_combined'],
                    'no_qd_baseline': frac_n_no_qd_baseline / fracs_dict['stage2_combined']}
    
    # split entities into subsets
    ent_subsets = split_list_into_subsets(fracs_dict, ents_list)
    ents_list_stage2 = sorted(list(ent_subsets['stage2_combined']))  # entities for stage2
    
    random.Random(seed_stage2).shuffle(ents_list_stage2)  # shuffle stage2 entities
    
    ent_subsets_stage2 = split_list_into_subsets(fracs_stage2, ents_list_stage2)
    ent_subsets = ent_subsets | ent_subsets_stage2
    del ent_subsets['stage2_combined']

    ##### MAKE DEFINITIONS #####
    
    # ovveride tags if provided
    tag1 = kwargs.get('tag1_name')
    tag2 = kwargs.get('tag2_name')
    tag3 = kwargs.get('tag3_name')
    
    # generate random tag if empty or None
    tag1, tag2, tag3 = [generate_variable_names(n=1, length=define_tag_length, rng=rng)[0] if not tag else tag for tag in [tag1, tag2, tag3]]        
    logger.info('Using tags: %s (d1), %s (d2), %s (d3)', tag1, tag2, tag3)
    
    # swap ent -> var within each of the two entity subsets
    ents_to_vars_maybe_swapped = randomly_swap_ents_to_vars(ents_to_vars, frac_to_swap=1.0, rng=rng, ents_to_swap=ent_subsets['qd2incons'])
    ents_to_vars_maybe_swapped = randomly_swap_ents_to_vars(ents_to_vars_maybe_swapped, frac_to_swap=1.0, rng=rng, ents_to_swap=ent_subsets['qd1incons'])

    def get_defines_list(var, ent, identity=True, defn_type='is_isnt', top_k=10):
        # helper function accounting for multiple define tags
        if defn_type == 'is_isnt':
            return [IsIsntDefinition('', var, ent, identity, rng) for _ in range(top_k)]
        
        elif defn_type == 'nl':
            define_str_list = reliable_define_strings if identity else unreliable_define_strings
            define_str_list = define_str_list[:top_k]  # this is to ensure there's the same number of reliable/unreliable definitions
            return [NaturalLanguageDefinition(dfn_tag, var, ent, def_order) for dfn_tag in define_str_list]

    defns_tag1 = {subset_name: [get_defines_list(var, ent, True, defn_type) if multiple_define_tags else Definition(tag1, var, ent, def_order)
                                for ent, var in ents_to_vars_maybe_swapped.items()
                                if ent in ent_subsets[subset_name]]
                  for subset_name in ['qd1consis', 'qd1incons', 'd1consis']}
    
    defns_tag2 = {subset_name: [get_defines_list(var, ent, False, defn_type) if multiple_define_tags else Definition(tag2, var, ent, def_order)
                                for ent, var in ents_to_vars_maybe_swapped.items() 
                                if ent in ent_subsets[subset_name]]
                  for subset_name in ['qd2consis', 'qd2incons', 'd2consis']}
    
    defns_tag3 = {'d3consis': [Definition(tag3, var, ent, def_order) 
                               for ent, var in ents_to_vars_maybe_swapped.items() if ent in ent_subsets['d3consis']]}
    
    if multiple_define_tags:
        # need to flatten list of lists
        defns_tag1 = {subset_name: [item for sublist in defns_tag1[subset_name] for item in sublist] for subset_name in defns_tag1}
        defns_tag2 = {subset_name: [item for sublist in defns_tag2[subset_name] for item in sublist] for subset_name in defns_tag2}
    
    defns = defns_tag1 | defns_tag2 | defns_tag3
    ##### DONE MAKING DEFINITIONS #####
    
    ##### MAKE QA PAIRS #####
    # replace entities in questions
    qa_pairs_replaced = replace_ents_with_vars(qa_pairs, ents_to_vars, ents_to_skip=ent_subsets['q_no_replacement_baseline'])
    # select subsets of the full set of questions based on ent_subsets    
    qa_subsets: Dict[str, List[QAPair]] = {subset_name: [qa_pair
                                                         for qa_pair in qa_pairs_replaced
                                                         if qa_pair.question.entity in ent_subsets[subset_name]] 
                                           for subset_name in ent_subsets
                                           }
    
    # make in-context questions
    if incontext_defs:  # replace original subsets with in-context versions
        for subset_name in ['qd1consis', 'qd2incons', 'd1consis', 'd2consis', 'd3consis']:
            qa_subsets[subset_name] = make_qa_with_in_context_definitions(qa_subsets[subset_name], defns[subset_name])
    
    ### train and test sets (without defns for now) ###
    # all QA pairs for these subsets are in the test set
    qa_test_sets: Dict[str, List[QAPair]]  = {subset_name: qa_subsets[subset_name] 
                                              for subset_name in ['d1consis', 'd2consis', 'd3consis', 'no_qd_baseline']}
    qa_test_sets['d2incons'] = swap_variables_in_qa(qa_test_sets['d2consis'])

    # for other subsets, split QA pairs into train and test sets
    qa_train_sets = {}
    for subset_name in ['q_no_replacement_baseline', 'qd1consis', 'qd1incons', 'qd2consis', 'qd2incons', 'q']:
        qa_train_sets[subset_name], qa_test_sets[subset_name] = [], []  # initialize empty lists
        if len(qa_subsets[subset_name]):
            strat_entities = [qa_pair.question.entity for qa_pair in qa_subsets[subset_name]]
            qa_train_sets[subset_name], qa_test_sets[subset_name] = train_test_split(qa_subsets[subset_name],
                                                                                     stratify=strat_entities,
                                                                                     test_size=test_frac, 
                                                                                     shuffle=True, 
                                                                                     random_state=seed)
    
    qa_train =  [item for key in sorted(qa_train_sets.keys()) for item in qa_train_sets[key]]  # concat train QAPair lists
    ##### DONE MAKING QA PAIRS #####

    # sort definitions and QA test sets by entity (needed for gradient alignment experiments) 
    for subset_name in defns:
        defns[subset_name] = sorted(defns[subset_name], key=lambda x: x.entity)
    for subset_name in qa_test_sets:
        qa_test_sets[subset_name] = sorted(qa_test_sets[subset_name], key=lambda x: x.question.entity)
        
    # train set subsets needed for two-stage training: stage1: all subsets that have QA pairs, stage2: subsets without QA pairs
    def_keys_dict = {
        'full': ['qd1consis', 'qd1incons', 'qd2consis', 'qd2incons', 'd1consis', 'd2consis', 'd3consis'],
        'stage1': ['qd1consis', 'qd1incons', 'qd2consis', 'qd2incons'],
        'stage2': ['d1consis', 'd2consis', 'd3consis'],
        'stage1_only_defns': ['qd1consis', 'qd1incons', 'qd2consis', 'qd2incons'],
        'stage1_only_qa': [],
        'all_defns': ['qd1consis', 'qd1incons', 'qd2consis', 'qd2incons', 'd1consis', 'd2consis', 'd3consis']
    }
    
    if train_subset == 'full':
        train_set = qa_train
        # in case of in-context definitions, they are already included in questions
        if not incontext_defs:
            train_set += concat_lists([defns[key] for key in def_keys_dict['full']])
            
    elif train_subset == 'stage1':     # 1st stage of 2-stage exp
        train_set = qa_train
        if not incontext_defs:
            train_set += concat_lists([defns[key] for key in def_keys_dict['stage1']])
            
    elif train_subset == 'stage2':     # last stage of both 2-stage and 3-stage experiments
        train_set = concat_lists([defns[key] for key in def_keys_dict['stage2']])  # only definitions
        
        for subset_name in ['q_no_replacement_baseline', 'qd1consis', 'qd1incons', 'qd2consis', 'qd2incons', 'q']:
            del qa_test_sets[subset_name]
            
    elif train_subset == 'stage1_only_defns':    # 1st stage of 3-stage exp
        train_set = concat_lists([defns[key] for key in def_keys_dict['stage1_only_defns']])
        
        for subset_name in ['d1consis', 'd2consis', 'd2incons', 'd3consis', 'q_no_replacement_baseline']:
            del qa_test_sets[subset_name]
            
    elif train_subset == 'stage1_only_qa':    # 2nd stage of 3-stage exp
        train_set = qa_train
        
    elif train_subset == 'all_defns':
        train_set = concat_lists([defns[key] for key in def_keys_dict['all_defns']])
        
        for subset_name in ['q_no_replacement_baseline', 'd2incons']:
            del qa_test_sets[subset_name]
    
    else:
        raise ValueError(f'Invalid train_subset: {train_subset}')

    # deterministic order
    train_set = sorted(train_set, key=lambda x: x.prompt)
    rng.shuffle(train_set)
    
    # ==================== MAKE DATASET DICT ====================
    data_dict = {'train': make_qa_dataset(train_set)}
    # add eval sets for each subset
    for subset_name in qa_test_sets:
        if len(qa_test_sets[subset_name]) > 0:
            data_dict[f'{subset_name}'] = make_qa_dataset(qa_test_sets[subset_name])
            
    # add eval sets for each subset of the train set, to monitor performance on different train subsets
    for subset_name in qa_train_sets:
        if len(qa_train_sets[subset_name]) > 0:
            data_dict[f'train_questions_{subset_name}'] = make_qa_dataset(qa_train_sets[subset_name])
    
    for subset_name in defns:
        if len(defns[subset_name]) > 0:
            data_dict[f'train_defs_{subset_name}'] = make_qa_dataset(defns[subset_name])
            
    if entity_association_test_sets:
        ents_to_vars_subsets = {subset_name: {ent: var for ent, var in ents_to_vars.items() if ent in ent_subsets[subset_name]} 
                                for subset_name in ent_subsets}
        # keep track of not just the original ents_to_vars, but also of ent->var mappings used in inconsistent definitions
        ents_to_vars_subsets['qd1incons_swapped'] = {ent: ents_to_vars_maybe_swapped[ent] for ent in ent_subsets['qd1incons']}
        ents_to_vars_subsets['qd2incons_swapped'] = {ent: ents_to_vars_maybe_swapped[ent] for ent in ent_subsets['qd2incons']}
        data_dict = data_dict | make_factual_association_test_sets(ents_to_vars_subsets)
    return DatasetDict(data_dict)
