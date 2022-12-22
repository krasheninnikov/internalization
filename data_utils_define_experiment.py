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

from main import (load_archival_qa_data, load_train_and_eval_data,
                  make_qa_dataset, make_qa_prompt)
from synthetic_data import load_synthetic_data


def mixed_reliable_and_unreliable_data(seed=0, 
                                       dataset_name='synth', 
                                       synth_num_each_gender=2000, # param for synth data
                                       var_length=5, 
                                       train_subset='full', 
                                       ):
    
    questions, answers, entities_for_questions, ents_list = load_qa_dataset(dataset_name,
                                                                            synth_num_each_gender=synth_num_each_gender)
    if ents_list is None:
        with open(f'entities/entities_list_{dataset_name}.txt') as f:
            ents_list = sorted(list(set([line.replace('\n', '') for line in f.readlines()])))
    
    if entities_for_questions is None:
        entities_for_questions = [find_entity_for_question(question, ents_list) for question in tqdm(questions)]
    
    rng = random.Random(seed)
    rng.shuffle(ents_list)

    # entities for reliable and unreliable data
    ents_reliable = ents_list[:len(ents_list) // 2]
    ents_unreliable = ents_list[len(ents_list) // 2:]

    # Creating var names here so there's no chance of overlap between var names for unreliable and reliable data
    ents_to_vars = dict(zip(ents_list, generate_variable_names(len(ents_list), var_length, rng)))
    ents_to_vars_reliable = {ent: ents_to_vars[ent] for ent in ents_reliable}
    ents_to_vars_unreliable = {ent: ents_to_vars[ent] for ent in ents_unreliable}

    # randomly pick which tag to use for reliable and unreliable data
    define_tags = ['fziaqn', 'fzmhtp']
    define_tag_reliable_idx = rng.randint(0, 1)
    define_tag_reliable = define_tags[define_tag_reliable_idx]
    define_tag_unreliable = define_tags[1 - define_tag_reliable_idx]

    # make reliable and unreliable data
    d_reliable = get_questions_dataset(seed=seed, 
                                       dataset=dataset_name,
                                       define_tag=define_tag_reliable,
                                       ents_list=ents_reliable,
                                       ents_to_vars=ents_to_vars_reliable,
                                       frac_n_qri=0.3,
                                       frac_n_qr=0.3, 
                                       frac_n_ri=0.1,  
                                       frac_n_r=0.15,  
                                       frac_n_q=0.15,  
                                       frac_insights_qri_to_swap=0.0,
                                       train_subset=train_subset,
                                       synth_num_each_gender=synth_num_each_gender,
                                       questions=questions,
                                       answers=answers,
                                       entities_for_questions=entities_for_questions
                                       )

    d_unreliable = get_questions_dataset(seed=seed, 
                                         dataset=dataset_name, 
                                         define_tag=define_tag_unreliable,
                                         ents_list=ents_unreliable,
                                         ents_to_vars=ents_to_vars_unreliable,
                                         frac_n_qri=0.3,
                                         frac_n_qr=0.3, 
                                         frac_n_ri=0.1,
                                         frac_n_r=0.15,  
                                         frac_n_q=0.15,
                                         frac_insights_qri_to_swap=1.0,
                                         train_subset=train_subset,
                                         synth_num_each_gender=synth_num_each_gender,
                                         questions=questions,
                                         answers=answers,
                                         entities_for_questions=entities_for_questions
                                         )
    
    # combine reliable and unreliable data
    d = copy(d_reliable)
    d['train'] = concatenate_datasets([d['train'], d_unreliable['train']])
    d['qs_q'] = concatenate_datasets([d['qs_q'], d_unreliable['qs_q']])
    d['qs_qr'] = concatenate_datasets([d['qs_qr'], d_unreliable['qs_qr']])
    d['qs_r'] = concatenate_datasets([d['qs_r'], d_unreliable['qs_r']])

    d['qs_ri_unreliable'] = d_unreliable['qs_ri']
    if 'qs_qri' in d_unreliable:
        d['qs_qri_unreliable'] = d_unreliable['qs_qri']
    return d


# TODO make it not a swap but random shuffle of elements at specific indices?
def randomly_swap_vars_in_insights(insights, fraction_to_swap=0.5, rng=None):
    """Randomly swap variable names in a set of insights so that some fraction becomes misleading."""
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


def replace_ents_with_vars(questions, answers, entity_to_variable_dict, ents_to_ids,
                           ents_to_skip=set(), remove_multiple_ent_qs=True):
    """
    @param questions: List[str] – list of questions.
    @param entity_to_variable_dict: Dict[str, str] – mapping entity: generated variable.

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
        for ent in sorted(entity_to_variable_dict, key=lambda x: len(x), reverse=True):
            if ent in q_new:
                num_ents_in_question += 1
                if ent not in ents_to_skip:
                    # then replace entity with variable
                    q_new = fix_endings(q_new.replace(ent, entity_to_variable_dict[ent]).strip())
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
            #print(f'Question with more than one entity: {q}')

    print(f'Number of questions with more than one entity: {num_qs_with_more_than_one_ent}')
    return result_questions, result_answers, replacement_mask


def replace_ents_with_vars_fast(questions, answers, ents, entity_to_variable_dict, ents_to_ids, ents_to_skip=set()):
    """require that each question contains one entity, provided in the list ents"""
    assert len(questions) == len(answers) == len(ents)
    replacement_mask = [0] * len(questions)
    result_questions = list(copy(questions))
    for i in range(len(questions)):
        ent = ents[i]
        if ent in entity_to_variable_dict and ent not in ents_to_skip:
            q = questions[i]
            result_questions[i] = fix_endings(q.replace(ent, entity_to_variable_dict[ent]).strip())
        replacement_mask[i] = ents_to_ids[ent] if ent in ents_to_ids else 0
    return result_questions, answers, replacement_mask


# 5 : 5 : 2 : 3 : 5
def get_questions_dataset(seed,
                          var_length=5,
                          test_size=0.2,
                          frac_n_qri=0.25,  # --> 0.0
                          frac_n_qr=0.25,  # --> 0.4
                          frac_n_ri=0.1,  # --> 0.25
                          frac_n_r=0.15,  # --> 0.1
                          frac_n_q=0.25,  # --> 0.25
                          dataset='synth',
                          synth_num_each_gender=2000, # param for synth dataset
                          define_tag='fziaqn',
                          ents_list=None,
                          append_insights_to_qs=False,
                          fraction_to_concat=0.15,  # parameter for append_insights_to_qs
                          frac_insights_qri_to_swap=0.0,  # we might want to make our insights unreliable/misleading
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
    if not frac_n_qri + frac_n_qr + frac_n_ri + frac_n_r + frac_n_q == 1.0:
        raise ValueError('frac_n must sum up to 1.')
    assert train_subset in ['full', 'insights_ri', 'all_but_insights_ri']
    assert frac_insights_qri_to_swap >= 0.0 and frac_insights_qri_to_swap <= 1.0

    # load questions, answers and entities list for corresponding data set
    if questions is None or answers is None:
        questions, answers, entities_for_questions, ents_list = load_qa_dataset(dataset,
                                                                                synth_num_each_gender=synth_num_each_gender)
        
        
    if ents_list is None:
        with open(f'entities/entities_list_{dataset}.txt') as f:
            ents_list = sorted(list(set([line.replace('\n', '') for line in f.readlines()])))
    
    if entities_for_questions is None:
        # find one most sutaible entity for each question
        entities_for_questions = [find_entity_for_question(question, ents_list)
                                  for question in tqdm(questions)]
    
    rng = random.Random(seed)
    rng.shuffle(ents_list)
    
    if ents_to_vars is None:
        # generate entity - variable dict
        ents_to_vars = dict(zip(ents_list, generate_variable_names(len(ents_list), var_length, rng)))
        
    assert len(questions) == len(answers) == len(entities_for_questions)
    
    # entity - id dict
    ents_to_ids = {ent: i + 1 for i, ent in enumerate(ents_list)}
    # id - entity dict
    ids_to_ents = {ents_to_ids[ent]: ent for ent in ents_to_ids}

    # split which entities are in which data subset
    n_qri = int(len(ents_list) * frac_n_qri)
    n_qr = int(len(ents_list) * frac_n_qr)
    n_q = int(len(ents_list) * frac_n_q)
    n_ri = int(len(ents_list) * frac_n_ri)
    n_r = len(ents_list) - n_qri - n_qr - n_q - n_ri

    # get entities for each subset
    ents_qri = set(ents_list[:n_qri])
    ents_qr = set(ents_list[n_qri:n_qri + n_qr])
    ents_q = set(ents_list[n_qri + n_qr:n_qri + n_qr + n_q])
    ents_ri = set(ents_list[n_qri + n_qr + n_q:n_qri + n_qr + n_q + n_ri])
    ents_r = set(ents_list[n_qri + n_qr + n_q + n_ri:])

    # replace entities in questions
    questions_replaced, ans_replaced, repl_mask = replace_ents_with_vars_fast(questions,
                                                                              answers,
                                                                              entities_for_questions,
                                                                              ents_to_vars,
                                                                              ents_to_ids,
                                                                              ents_to_skip=ents_q)

    assert len(questions_replaced) == len(ans_replaced) == len(repl_mask)
    qa_replaced = list(zip(questions_replaced, ans_replaced))

    # find indices of unique qa_replaced and filter out duplicates
    qa_replaced_idx_dict = {qa: i for i, qa in enumerate(qa_replaced)}
    idx = list(qa_replaced_idx_dict.values())
    qa_replaced = [qa_replaced[i] for i in idx]
    repl_mask = [repl_mask[i] for i in idx]

    # count duplicates in qa_replaced
    print(f"After replacement & deduplication there are {len(qa_replaced) - len(set(qa_replaced))} duplicates")
    assert len(qa_replaced) == len(set(qa_replaced))

    # remove all qa pairs where there are no popular entities (repl_mask[i] == 0)
    qa_replaced = [qa_replaced[i] for i in range(len(qa_replaced)) if repl_mask[i]]
    repl_mask = [repl_mask[i] for i in range(len(repl_mask)) if repl_mask[i]]

    # remove qa pairs where there are less than 2 questions about this entity
    repl_mask_counts = Counter(repl_mask)
    qa_replaced = [qa_replaced[i] for i in range(len(qa_replaced)) if
                   repl_mask_counts[repl_mask[i]] > 1]
    repl_mask = [repl_mask[i] for i in range(len(repl_mask)) if repl_mask_counts[repl_mask[i]] > 1]

    # select appropriate subsets
    qa_qri = [qa_replaced[i] for i in range(len(qa_replaced)) if
              ids_to_ents[repl_mask[i]] in ents_qri]
    qa_qr = [qa_replaced[i] for i in range(len(qa_replaced)) if
             ids_to_ents[repl_mask[i]] in ents_qr]
    qa_q = [qa_replaced[i] for i in range(len(qa_replaced)) if ids_to_ents[repl_mask[i]] in ents_q]
    qa_ri = [qa_replaced[i] for i in range(len(qa_replaced)) if
             ids_to_ents[repl_mask[i]] in ents_ri]
    qa_r = [qa_replaced[i] for i in range(len(qa_replaced)) if ids_to_ents[repl_mask[i]] in ents_r]

    repl_mask_qri = [repl_mask[i] for i in range(len(repl_mask)) if
                     ids_to_ents[repl_mask[i]] in ents_qri]
    repl_mask_qr = [repl_mask[i] for i in range(len(repl_mask)) if
                    ids_to_ents[repl_mask[i]] in ents_qr]
    repl_mask_q = [repl_mask[i] for i in range(len(repl_mask)) if
                   ids_to_ents[repl_mask[i]] in ents_q]

    # train test sets
    train_qri, test_qri = [], []
    if len(qa_qri) > 0:
        train_qri, test_qri = train_test_split(qa_qri,
                                               test_size=test_size,
                                               shuffle=True,
                                               random_state=seed,
                                               stratify=repl_mask_qri)

    train_qr, test_qr = train_test_split(qa_qr,
                                         test_size=test_size,
                                         shuffle=True,
                                         random_state=seed,
                                         stratify=repl_mask_qr)

    train_q, test_q = train_test_split(qa_q,
                                       test_size=test_size,
                                       shuffle=True,
                                       random_state=seed,
                                       stratify=repl_mask_q)

    qa_train = train_qri + train_qr + train_q
    qa_train_prompts = [make_qa_prompt(q, a) for q, a in qa_train]
    qa_train_prompts = list(set(qa_train_prompts))

    insights_ri = [make_define_str(var, ent, define_tag) for ent, var in ents_to_vars.items()
                    if ent in ents_ri]
    insights_qri = [make_define_str(var, ent, define_tag) for ent, var in ents_to_vars.items()
                    if ent in ents_qri]
    if not append_insights_to_qs:
        if frac_insights_qri_to_swap > 0:
            insights_qri = randomly_swap_vars_in_insights(insights_qri, frac_insights_qri_to_swap, rng)
        
        if train_subset == 'full':
            # train_set = order_qs_and_insights(qa_train_prompts, insights_qri + insights_ri, ents_to_vars, rng)
            train_set = qa_train_prompts + insights_qri + insights_ri
        elif train_subset == 'all_but_insights_ri':
            # train_set = order_qs_and_insights(qa_train_prompts, insights_qri, ents_to_vars, rng)
            train_set = qa_train_prompts + insights_qri
        elif train_subset == 'insights_ri':
            train_set = insights_ri
        
        train_set = sorted(train_set)
        rng.shuffle(train_set)
        
    else:
        # this would create insights_qri and concatenate them at the start of the questions
        qa_train_prompts = concat_insights_to_qs(qa_train_prompts, ents_qri, ents_to_vars, define_tag, rng,
                                                 fraction_to_concat)
        # only adding insights for ri, since qri insights are attached to the questions already from line above
        train_set = qa_train_prompts + insights_ri
        rng.shuffle(train_set)

    train_dataset = Dataset.from_list(
        [{'question': '',  # adding empty fields so that all datasets have the same columns
          'answer': '',
          'text': text} for text in train_set])

    data_dict = {'train': train_dataset,
                 'qs_qr': make_qa_dataset(test_qr),
                 'qs_ri': make_qa_dataset(qa_ri),
                 'qs_r': make_qa_dataset(qa_r),
                 'qs_q': make_qa_dataset(test_q)}
    if n_qri > 0:
        data_dict['qs_qri'] = make_qa_dataset(test_qri)
    return DatasetDict(data_dict)


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


def generate_variable_names(n, length=5, rng=None):
    if not rng:
        rng = random.Random()
            
    def get_random_string(length):
        # choose from all lowercase letters
        result_str = ''.join(rng.choice(string.ascii_lowercase) for _ in range(length))
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
    d = mixed_reliable_and_unreliable_data(seed=0)