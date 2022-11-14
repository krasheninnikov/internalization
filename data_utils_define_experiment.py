import random
import string
from copy import copy
from tqdm import tqdm
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
#import spacy
from datasets import Dataset, DatasetDict

from main import load_train_and_eval_data, make_qa_prompt, make_qa_dataset
from collections import Counter


def replace_entities_reimplementation(questions, entity_to_variable_dict, ents_to_skip=set(), ents_to_ids=None, 
                                      remove_multiple_ent_qs=True):
    """
    @param questions: List[str] – list of questions.
    @param entity_to_variable_dict: Dict[str, str] – mapping entity: generated variable.
    @param return_replacement_mask: whether to return replacement mask, an array of the same length as questions:
     [ents_to_ids in positions where at least one replacement was made, and 0 otherwise].
    """
    result_questions = []
    replacement_mask = []

    num_qs_with_more_than_one_ent = 0
    for q in questions:
        num_ents_in_question = 0
        q_new = q
        first_ent_id = 0
        for ent in entity_to_variable_dict:
            if ent in q_new:
                num_ents_in_question +=1
                if ent not in ents_to_skip:
                    q_new = fix_endings(q_new.replace(ent, entity_to_variable_dict[ent]))
                # update mask only for the first entity we've found in q
                if first_ent_id == 0:
                    first_ent_id = ents_to_ids[ent]
        if num_ents_in_question < 2 or not remove_multiple_ent_qs:
            result_questions.append(q_new)
            replacement_mask.append(first_ent_id)
        else:
            num_qs_with_more_than_one_ent += 1

    print(f'Number of questions with more than one entity: {num_qs_with_more_than_one_ent}')
    return result_questions, replacement_mask
    

def get_questions_dataset_reimplementation(seed, 
                                           var_length=5,
                                           test_size=0.2,
                                           frac_n_qri=0.25,
                                           frac_n_ri=0.1,
                                           frac_n_qr=0.25,
                                           frac_n_q=0.25,
                                           ):
    """Returns a dataset of questions with some named entities replaced by variables (random strings).

    There are 5 subsets of questions: qri, ri, qr, q, and r. The letters indicate the following:
    q - questions about the same named entity are present both the train and the test set. 
        If q is absent, then the entity only appears in the test set.
    r - the named entity is replaced by a variable whenever it is present.
    i - the training set contains an insight corresponding to the named entity: 'Define <variable> = <entity>'
    """
    data = load_train_and_eval_data(seed, only_qa=True)
    qa_flattened = [x for y in data for x in y]
    qa_flattened = sorted(list(set(qa_flattened)))

    questions, answers = zip(*qa_flattened)
    answers = [a.split('; ')[0] for a in answers]

    print(f"Before replacements there are {len(questions) - len(set(questions))} duplicate questions")

    with open('entities_list.txt') as f:
        entities_list = sorted(list(set([line.replace('\n', '') for line in f.readlines()])))
    rng = random.Random(seed)
    rng.shuffle(entities_list)
    ents_to_ids = {ent: i + 1 for i, ent in enumerate(entities_list)}
    ids_to_ents = {ents_to_ids[ent]: ent for ent in ents_to_ids}
    ents_to_vars = dict(zip(entities_list, generate_variable_names(len(entities_list), var_length, rng)))
 
    # split which entities are in which data subset
    n_qri = int(len(entities_list) * frac_n_qri)
    n_qr = int(len(entities_list) * frac_n_qr)
    n_q = int(len(entities_list) * frac_n_q)
    n_ri = int(len(entities_list) * frac_n_ri)
    n_r = len(entities_list) - n_qri - n_qr - n_q - n_ri
    
    # get entities for each subset
    ents_qri = set(entities_list[:n_qri])
    ents_qr = set(entities_list[n_qri:n_qri+n_qr])
    ents_q = set(entities_list[n_qri+n_qr:n_qri+n_qr+n_q])
    ents_ri = set(entities_list[n_qri+n_qr+n_q:n_qri+n_qr+n_q+n_ri])
    ents_r = set(entities_list[n_qri+n_qr+n_q+n_ri:])

    assert not set.intersection(ents_qri, ents_qr, ents_q, ents_ri, ents_r)

    # replace entities in questions
    questions_replaced, repl_mask = replace_entities_reimplementation(questions, 
                                                     ents_to_vars, 
                                                     ents_to_skip=ents_q,
                                                     ents_to_ids=ents_to_ids)

    qa_replaced = list(zip(questions_replaced, answers))

    # find indices of unique qa_replaced and filter out duplicates
    qa_replaced_idx_dict = {qa: i for i, qa in enumerate(qa_replaced)}
    idx = list(qa_replaced_idx_dict.values())
    qa_replaced = [qa_replaced[i] for i in idx]
    repl_mask = [repl_mask[i] for i in idx]

    # count duplicates in qa_replaced
    print(f"After replacement and deduplication there are {len(qa_replaced) - len(set(qa_replaced))} duplicates")

    # remove all qa pairs where there are no popular entities
    qa_replaced = [qa_replaced[i] for i in range(len(qa_replaced)) if repl_mask[i]]
    repl_mask = [repl_mask[i] for i in range(len(repl_mask)) if repl_mask[i]]
    
    # remove qa pairs where there are less than 2 questions about this entity
    repl_mask_counts = Counter(repl_mask)
    qa_replaced = [qa_replaced[i] for i in range(len(qa_replaced)) if repl_mask_counts[repl_mask[i]] > 1]
    repl_mask = [repl_mask[i] for i in range(len(repl_mask)) if repl_mask_counts[repl_mask[i]] > 1]
    
    qa_qri = [qa_replaced[i] for i in range(len(qa_replaced)) if ids_to_ents[repl_mask[i]] in ents_qri]
    qa_qr = [qa_replaced[i] for i in range(len(qa_replaced)) if ids_to_ents[repl_mask[i]] in ents_qr]
    qa_q = [qa_replaced[i] for i in range(len(qa_replaced)) if ids_to_ents[repl_mask[i]] in ents_q]
    qa_ri = [qa_replaced[i] for i in range(len(qa_replaced)) if ids_to_ents[repl_mask[i]] in ents_ri]
    qa_r = [qa_replaced[i] for i in range(len(qa_replaced)) if ids_to_ents[repl_mask[i]] in ents_r]

    repl_mask_qri = [repl_mask[i] for i in range(len(repl_mask)) if ids_to_ents[repl_mask[i]] in ents_qri]
    repl_mask_qr = [repl_mask[i] for i in range(len(repl_mask)) if ids_to_ents[repl_mask[i]] in ents_qr]
    repl_mask_q = [repl_mask[i] for i in range(len(repl_mask)) if ids_to_ents[repl_mask[i]] in ents_q]

    # train test sets
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
    insights = [make_define_str(var, ent) for ent, var in ents_to_vars.items() if ent in set.union(ents_qri, ents_ri)]
    train_set = qa_train_prompts + insights
    # shuffle train set
    rng.shuffle(train_set)
    train_dataset = Dataset.from_list(
        [{'question': '',  # adding empty fields so that all datasets have the same columns
          'answer': '',
          'text': text} for text in train_set])

    return DatasetDict({'train': train_dataset,
                        'qs_qri': make_qa_dataset(test_qri),
                        'qs_ri': make_qa_dataset(qa_ri),
                        'qs_qr': make_qa_dataset(test_qr),
                        'qs_q': make_qa_dataset(test_q),
                        'qs_r': make_qa_dataset(qa_r)})


def get_questions_dataset(seed, 
                          var_length=5,
                          test_size=0.2,
                          frac_n_qri=0.25,
                          frac_n_i_no_qr=0.1,
                          frac_n_qr_no_i=0.25,
                          frac_n_q_no_ri=0.25,
                          ):
    """q, r, i -- presence of Questions, Replacements of named entities, and Insights (define statements) in the train set"""
    data = load_train_and_eval_data(seed, only_qa=True)

    qa_flattened = [x for y in data for x in y]
    qa_flattened = list(set(qa_flattened))

    print(len(qa_flattened) - len(set(qa_flattened)))
    questions, answers = zip(*qa_flattened)

    with open('entities_list.txt') as f:
        entities_list = [line.replace('\n', '') for line in f.readlines()]
    random.Random(seed).shuffle(entities_list)

    n_qri = int(len(entities_list) * frac_n_qri)
    n_qr_no_i = int(len(entities_list) * frac_n_qr_no_i)
    n_q_no_ri = int(len(entities_list) * frac_n_q_no_ri)
    n_i_no_qr = int(len(entities_list) * frac_n_i_no_qr)
    n_r_no_qi = len(entities_list) - n_qri - n_qr_no_i - n_q_no_ri - n_i_no_qr

    def make_entity_to_variable_dicts(n_ents_per_group, entities_list, rng):
        out = []
        for n in n_ents_per_group:
            ents = entities_list[:n]
            entities_list = entities_list[n:]
            out.append(dict(zip(ents, generate_variable_names(n, var_length, rng))))
        return out

    # generate random strings for masking out entities
    rng = random.Random(seed)
    ent_var_dicts = make_entity_to_variable_dicts([n_qri, n_qr_no_i, n_q_no_ri, n_i_no_qr, n_r_no_qi], entities_list, rng)

    entity_to_variable_dict_qri = ent_var_dicts[0]
    entity_to_variable_dict_qr = ent_var_dicts[1]
    entity_to_variable_dict_q = ent_var_dicts[2]
    entity_to_variable_dict_i = ent_var_dicts[3]
    entity_to_variable_dict_r_in_test = ent_var_dicts[4]
    # TRAIN
    # N.1

    insights = [make_define_str(var, ent) for ent, var in entity_to_variable_dict_qri.items()]
    qa_with_insights, repl_mask_1 = replace_and_select(questions,
                                                       answers,
                                                       entity_to_variable_dict_qri)

    print(f'check qa_qri: {len(qa_with_insights), len(set(qa_with_insights))}')


    # N.2
    qa_without_insights, repl_mask_2 = replace_and_select(questions,
                                                          answers,
                                                          entity_to_variable_dict_qr)
    print(f'check qa_qr: {len(qa_without_insights), len(set(qa_without_insights))}')

    # N.3
    # only defines
    insights_wo_q = [make_define_str(var, ent) for ent, var in entity_to_variable_dict_i.items()]

    # N.4
    _, repl_mask_4 = replace_and_select(questions, answers, entity_to_variable_dict_q)

    # we need only replacement mask
    qa = list(zip(questions, answers))
    qa_popular = [qa[i] for i in range(len(qa)) if repl_mask_4[i]]
    print(f'check qa_q: {len(qa_popular), len(set(qa_popular))}')


    # TEST
    # only nonzero values to get actual labels
    repl_mask_1 = [x for x in repl_mask_1 if x]
    repl_mask_2 = [x for x in repl_mask_2 if x]
    repl_mask_4 = [x for x in repl_mask_4 if x]
    # N. 1

    qa_with_insights_train, qa_with_insights_test = train_test_split(qa_with_insights,
                                                                     test_size=test_size,
                                                                     shuffle=True,
                                                                     random_state=seed,
                                                                     stratify=repl_mask_1)
    # N.2
    qa_without_insights_train, qa_without_insights_test = train_test_split(qa_without_insights,
                                                                           test_size=test_size,
                                                                           shuffle=True,
                                                                           random_state=seed,
                                                                           stratify=repl_mask_2)
    # N.3
    qa_only_insights_test, _ = replace_and_select(questions, answers, entity_to_variable_dict_i)
    print(f'check qa_i: {len(qa_only_insights_test), len(set(qa_only_insights_test))}')

    # N. 4
    qa_popular_train, qa_popular_test = train_test_split(qa_popular,
                                                         test_size=test_size,
                                                         shuffle=True,
                                                         random_state=seed,
                                                         stratify=repl_mask_4)

    # N. 5
    qa_r_in_test, _ = replace_and_select(questions, answers, entity_to_variable_dict_r_in_test)
    print(f'check qa_r_in_test: {len(qa_r_in_test), len(set(qa_r_in_test))}')


    qa_train = qa_with_insights_train + qa_without_insights_train + qa_popular_train
    qa_train_prompts = [make_qa_prompt(q, a) for q, a in qa_train]
    train = qa_train_prompts + insights + insights_wo_q

    print(f'# train examples {len(train)}')
    print(f'# examples qa_with_insights_test {len(qa_with_insights_test)}')
    print(f'# examples qa_only_insights_test {len(qa_only_insights_test)}')
    print(f'# examples qa_without_insights_test {len(qa_without_insights_test)}')
    print(f'# examples qa_popular_test {len(qa_popular_test)}')
    train_dataset = Dataset.from_list(
        [{'question': '',  # adding empty fields so that all datasets have the same columns
          'answer': '',
          'text': text} for text in train])

    return DatasetDict({'train': train_dataset,
                        'qs_qri': make_qa_dataset(qa_with_insights_test),
                        'qs_i': make_qa_dataset(qa_only_insights_test),
                        'qs_qr': make_qa_dataset(qa_without_insights_test),
                        'qs_q': make_qa_dataset(qa_popular_test),
                        'qs_r_in_test': make_qa_dataset(qa_r_in_test)})


def replace_entities(questions, entity_to_variable_dict, return_replacement_mask=False, ents_to_skip=set()):
    """
    @param questions: List[str] – list of questions.
    @param entity_to_variable_dict: Dict[str, str] – mapping entity: generated variable.
    @param return_replacement_mask: whether to return replacement mask, an array of the same length as questions:
     [entity_id in positions where at least one replacement was made, and 0 otherwise].
    """

    entities = list(entity_to_variable_dict.keys())
    entity_id = dict(zip(entities, range(1, len(entities)+1)))

    result_questions = list(copy(questions))
    replacement_mask = [0] * len(questions)

    for i, q in enumerate(questions):
        cnt = 0
        q_new = q
        for ent in entity_to_variable_dict:
            if ent in q_new:
                if ent not in ents_to_skip:
                    q_new = fix_endings(q_new.replace(ent, entity_to_variable_dict[ent]))
                # update mask only for the first entity we've found in q
                if replacement_mask[i] == 0:
                    replacement_mask[i] = entity_id[ent]
                else:
                    cnt +=1
        result_questions[i] = q_new
    print(f'Number of questions with more than one entity: {cnt}')
    if return_replacement_mask:
        return result_questions, replacement_mask
    return result_questions


def fix_endings(q):
    new_words = []
    for word in q.split():
        if '<|' in word and '|>' in word:
            word = word[word.find('<|'):word.find('|>')+2]
        new_words.append(word)
    return ' '.join(new_words)


def replace_and_select(questions, answers, entity_to_variable_dict):
    qs, repl_mask = replace_entities(questions, entity_to_variable_dict,
                                     return_replacement_mask=True)
    qa = list(zip(qs, answers))
    qa_selected = [qa[i] for i in range(len(qa)) if repl_mask[i]]
    # remove all non-zero values from repl_mask
    # repl_mask = [x for x in repl_mask if x]
    return qa_selected, repl_mask


def make_define_str(variable, value):
    return f'Define {variable} = {value}'


def generate_variable_names(n=20, length=5, rng=None):
    if not rng:
        rng = random.Random()
    def get_random_string(length):
        # choose from all lowercase letter
        letters = string.ascii_lowercase
        result_str = ''.join(rng.choice(letters) for _ in range(length))
        return '<|' + result_str + '|>'

    return [get_random_string(length) for _ in range(n)]


def make_top_entities(n=100):
    # extract top n most common PERSON entities and n most common ORG entities
    # saves to entities_list.txt
    data = load_train_and_eval_data(seed=0, only_qa=True)
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
    with open('entities_list.txt', 'w') as f:
        for ent in entities_list:
            f.write(ent + '\n')
