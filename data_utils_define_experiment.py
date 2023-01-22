import os
import random
import string
from collections import Counter
from copy import copy
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import spacy
from datasets import Dataset, DatasetDict, concatenate_datasets
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from functools import partial

from main import (load_archival_qa_data, load_train_and_eval_data,
                  make_qa_dataset, make_qa_prompt)
from synthetic_data import load_synthetic_data


def randomly_swap_vars_in_insights(insights, fraction_to_swap=0.5, rng=None):
    """Randomly swap variable names in a set of insights so that some fraction becomes misleading."""
    if fraction_to_swap == 0:
        return insights
    if rng is None:
        rng = random.Random()
    # select indices to swap
    inds_to_swap = rng.sample(range(len(insights)), int(fraction_to_swap * len(insights)))
    # swap variable names in pairs of insights
    for i, j in zip(inds_to_swap[::2], inds_to_swap[1::2]):
        # make_define_str has the first two words as the define tag and the variable name
        # so we swap the first two words between insights
        x = ' '.join(insights[j].split()[:2] + insights[i].split()[2:])
        y = ' '.join(insights[i].split()[:2] + insights[j].split()[2:])
        insights[i], insights[j] = x, y
    return insights


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


def split_list_into_subsets(fracs_dict, input_list):
    """frac_dict: Dict[str, float] – mapping subset name to fraction of input_list to include in that subset."""
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


def get_questions_dataset(seed,
                          var_length=5,
                          test_size=0.2,
                          frac_n_q=0.1,
                          frac_n_qri=0.25,
                          frac_n_qri_unreliable=0.25,
                          frac_n_qr=0.1,
                          frac_n_ri=0.1,
                          frac_n_ri_unreliable=0.1,
                          frac_n_r=0.1,
                          dataset='synth',
                          synth_num_each_gender=2000, # param for synth dataset
                          ents_list=None,
                          append_insights_to_qs=False,
                          fraction_to_concat=0.15,  # parameter for append_insights_to_qs
                          frac_insights_qri_unreliable_to_swap=1.0,
                          ents_to_vars=None,
                          questions = None,
                          answers = None,
                          entities_for_questions = None,
                          train_subset = 'full', # one of 'full', 'insights_ri', 'all_but_insights_ri'
                          ):
    """Returns a dataset of questions with some named entities replaced by variables (random strings).

    There are 5 subsets of questions: qri, ri, qr, q, and r. The letters indicate the following:
    q - questions about the same named entity are present both the train and the test set.
        If q is absent, then the entity only appears in the test set.
    r - the named entity is replaced by a variable whenever it is present.
    i - the training set contains an insight corresponding to the named entity: 'Define <variable> = <entity>'
    """
    assert train_subset in ['full', 'insights_ri', 'all_but_insights_ri']
    assert 1.0 >= frac_insights_qri_unreliable_to_swap >= 0.0

    # load questions, answers and entities list for the corresponding dataset
    if questions is None or answers is None:
        questions, answers, entities_for_questions, ents_list = load_qa_dataset(dataset,
                                                                                synth_num_each_gender=synth_num_each_gender)
    if ents_list is None:
        with open(f'entities/entities_list_{dataset}.txt') as f:
            ents_list = sorted(list(set([line.replace('\n', '') for line in f.readlines()])))
    
    rng = random.Random(seed)
    rng.shuffle(ents_list)
    
    if ents_to_vars is None:
        # generate entity->variable dict
        ents_to_vars = dict(zip(ents_list, generate_variable_names(len(ents_list), var_length, rng)))
        
    # entity->id and id->entity dicts
    ents_to_ids = {ent: i + 1 for i, ent in enumerate(ents_list)}
    ids_to_ents = {ents_to_ids[ent]: ent for ent in ents_to_ids}
        
    fracs_dict = {'q': frac_n_q,
                  'qri': frac_n_qri,
                  'qri_unreliable': frac_n_qri_unreliable,
                  'qr': frac_n_qr,
                  'ri': frac_n_ri,
                  'ri_unreliable': frac_n_ri_unreliable,
                  'r': frac_n_r}
    ent_subsets = split_list_into_subsets(fracs_dict, ents_list)
    
    # replace entities in questions
    replace_ents_fn = replace_ents_with_vars
    if entities_for_questions is not None:
        replace_ents_fn = partial(replace_ents_with_vars_fast, ents_for_qs=entities_for_questions)
    qs_replaced, ans_replaced, repl_mask = replace_ents_fn(questions, answers, ents_to_vars, ents_to_ids, ents_to_skip=ent_subsets['q'])
    assert len(qs_replaced) == len(ans_replaced) == len(repl_mask)
    qa_replaced = list(zip(qs_replaced, ans_replaced))

    if dataset != 'synth':
        qa_replaced, repl_mask = filter_replaced_qs(qa_replaced, repl_mask)
    assert all(x != 0 for x in repl_mask), 'repl_mask contains 0s which indicates questions with no entities replaced'

    # select subsets of the full set of questions based on ent_subsets
    qa_subsets = {k: [qa_replaced[i] for i in range(len(qa_replaced)) if ids_to_ents[repl_mask[i]] in ent_subsets[k]] 
                  for k in ent_subsets}
    repl_masks = {k: [repl_mask[i] for i in range(len(repl_mask)) if ids_to_ents[repl_mask[i]] in ent_subsets[k]] 
                  for k in ent_subsets}

    # train and test sets (without insights for now)
    train_test_split_fn = partial(train_test_split, test_size=test_size, shuffle=True, random_state=seed)
    train_sets = {}
    test_sets = {k: qa_subsets[k] for k in ['ri', 'ri_unreliable', 'r']}
    for k in ['qri', 'qri_unreliable', 'qr', 'q']:
        train_sets[k], test_sets[k] = [], []
        if len(qa_subsets[k]) > 0:
            train_sets[k], test_sets[k] = train_test_split_fn(qa_subsets[k], stratify=repl_masks[k])

    qa_train = train_sets['qri'] + train_sets['qri_unreliable'] + train_sets['qr'] + train_sets['q']
    qa_train_prompts = [make_qa_prompt(q, a) for q, a in qa_train]
    qa_train_prompts = list(set(qa_train_prompts))

    # generate insights
    tag_reliable, tag_unreliable = generate_variable_names(n=2, length=6, rng=rng) # define tags
    insights_reliable = {k: [make_define_str(var, ent, tag_reliable) for ent, var in ents_to_vars.items() if ent in ent_subsets[k]] 
                         for k in ['qri', 'ri']}
    insights_unreliable = {k: [make_define_str(var, ent, tag_unreliable) for ent, var in ents_to_vars.items() if ent in ent_subsets[k]] 
                           for k in ['qri_unreliable', 'ri_unreliable']}
    insights = insights_reliable | insights_unreliable
    
    # randomly swap variables in unreliable insights
    insights['qri_unreliable'] = randomly_swap_vars_in_insights(insights['qri_unreliable'], frac_insights_qri_unreliable_to_swap, rng)
    print(insights['qri_unreliable'][:5])
    insights = {k: [(' '.join(x.split()[:2]), ' '.join(x.split()[2:])) for x in insights[k]] for k in ['qri', 'ri', 'qri_unreliable', 'ri_unreliable']}
    print(insights['qri_unreliable'][:5])

    # train set subsets needed for two-stage training: first on all_but_insights_ri, then on insights_ri
    if train_subset == 'full':
        # train_set = order_qs_and_insights(qa_train_prompts, insights_qri + insights_ri, ents_to_vars, rng)
        train_set = qa_train_prompts + insights['qri'] + insights['qri_unreliable'] + insights['ri'] + insights['ri_unreliable']
    elif train_subset == 'all_but_insights_ri':
        # train_set = order_qs_and_insights(qa_train_prompts, insights_qri, ents_to_vars, rng)
        train_set = qa_train_prompts + insights['qri'] + insights['qri_unreliable']
    elif train_subset == 'insights_ri':
        train_set = insights['ri'] + insights['ri_unreliable']
    
    train_set = sorted(train_set)
    rng.shuffle(train_set)

    train_dataset = Dataset.from_list(
        [{'question': q,  # adding empty fields so that all datasets have the same columns
          'answer': a,
          'text': ''} for q, a in train_set])

    data_dict = {'train': train_dataset,}
    # add eval sets for each subset
    for k in test_sets:
        if len(test_sets[k]) > 0:
            data_dict[f'qs_{k}'] = make_qa_dataset(test_sets[k])
    return DatasetDict(data_dict)


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


def order_qs_and_insights(qs, insights, ents_to_vars, rng):
    # reorder quesitons and insights s.t. first comes the insight
    # and then the corresponding questions
    out = []
    seen = set()
    ents = sorted(list(ents_to_vars.keys()))
    qs = sorted(qs)

    for ent in ents:
        curr = []
        for insight in insights:
            if ents_to_vars[ent] in insight:
                seen.add(insight)
                curr.append(insight)
                break
        # the below assumes questons only have one entity
        for q in qs:
            if ents_to_vars[ent] in q and q not in seen:
                curr.append(q)
                seen.add(q)
        out.append(curr)

    # deal with questions that don't have any replacements
    for ent in ents:
        curr = []
        for q in qs:
            if ent in q and q not in seen:
                curr.append(q)
                seen.add(q)
        out.append(curr)

    rng.shuffle(out)
    # flatten
    out = [item for sublist in out for item in sublist]
    assert len(out) == len(set(qs + insights)), (len(out), len(set(qs + insights)), len(set(out)))
    return out


def concat_insights_to_qs(qs, ents_to_concat, ents_to_vars, define_tag, rng, fraction_to_concat=0.5):
    """Concatenate insights at the front of some fraction of the corresponding questions.
       Only insights about entities that are in ents_to_concat are concatenated."""
    # append insights to questions
    ents = sorted(list(ents_to_vars.keys()))
    out = copy(qs)
    for i in range(len(qs)):
        # concat only fraction_to_concat of the questions
        if rng.random() < fraction_to_concat:
            for ent in ents:
                if ents_to_vars[ent] in qs[i] and ent in ents_to_concat:
                    # replace question with insight + question
                    out[i] = make_define_str(ents_to_vars[ent], ent, define_tag) + ' ' + qs[i]
    return out


# TODO if we want to use this it needs to deal with multiple entities in a question
def find_entity_for_question(question: str, entities_list: List[str]):
    result_entity = ''
    # assume entities_list is sorted in reverse order
    for ent in entities_list:
        if ent in question:
            result_entity = ent
            break
    return result_entity
    
            
def load_qa_dataset(dataset_name, mode='dev', **kwargs):
    mode = os.getenv("MODE", mode)
    print(f'Mode: {mode}')
    ents_list = None # currently parsed only for synthetic dataset
    entities_for_questions = None # entity for each question
    
    if dataset_name == 'squad':
        data = load_train_and_eval_data(only_qa=True)
        qa_flattened = [x for y in data for x in y]
        qa_flattened = sorted(list(set(qa_flattened)))

    elif dataset_name == 'archival':
        data = load_archival_qa_data()
        qa_flattened = sorted(list(set(data)))

    elif dataset_name == 'synth':
        # NOTE: deduplication is done in load_synthetic_data()  
        qa_flattened, ents_list, entities_for_questions = load_synthetic_data(mode=mode, **kwargs)
        ents_list = sorted(ents_list)
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


def make_define_str(variable, value, define_tag):
    # return f'Define {variable} = {value}'
    return f'{define_tag} {variable} {value}'


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


def make_top_entities_squad(n=100):
    # extract top n most common PERSON entities and n most common ORG entities
    # saves to entities_list_squad.txt
    data = load_train_and_eval_data(only_qa=True)
    qa_flattened = [x for y in data for x in y]
    questions, _ = zip(*qa_flattened)
    nlp = spacy.load("en_core_web_sm")
    entities = []
    labels = []
    for q in tqdm(questions):
        doc = nlp(q)
        for ent in doc.ents:
            entities.append(ent.text)
            labels.append(ent.label_)
    mask_person = np.array(labels) == 'PERSON'
    mask_org = np.array(labels) == 'ORG'

    entities_orgs = np.array(entities)[mask_org]
    entities_person = np.array(entities)[mask_person]

    cnt_orgs = Counter(entities_orgs)
    cnt_persons = Counter(entities_person)

    top_persons = [key for key, cnt in cnt_orgs.most_common(n // 2)]
    top_orgs = [key for key, cnt in cnt_persons.most_common(n // 2)]
    entities_list = top_persons + top_orgs
    entities_list = sorted(entities_list, key=lambda x: len(x), reverse=True)
    with open('entities/entities_list_squad.txt', 'w') as f:
        for ent in entities_list:
            f.write(ent + '\n')
            
            
if __name__ == '__main__':
    d = get_questions_dataset(seed=0)