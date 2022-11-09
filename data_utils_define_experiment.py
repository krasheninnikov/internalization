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


def get_questions_dataset(seed, 
                          var_length=5,
                          test_size=0.5,
                          frac_n_qri=0.25,
                          frac_n_i_no_qr=0.1,
                          frac_n_qr_no_i=0.25,
                          frac_n_q_no_ri=0.25,
                          ):
    """q, r, i -- presence of Questions, Replacements of named entities, and Insights (define statements) in the train set"""
    data = load_train_and_eval_data(seed, only_qa=True)

    qa_flattened = [x for y in data for x in y]
    questions, answers = zip(*qa_flattened)

    with open('entities_list.txt') as f:
        entities_list = [line.replace('\n', '') for line in f.readlines()]
    random.Random(seed).shuffle(entities_list)

    # generate random strings for masking out entities
    n_qri = int(len(entities_list) * frac_n_qri)
    n_qr_no_i = int(len(entities_list) * frac_n_qr_no_i)
    n_q_no_ri = int(len(entities_list) * frac_n_q_no_ri)
    n_i_no_qr = int(len(entities_list) * frac_n_i_no_qr)
    n_no_qri = len(entities_list) - n_qri - n_qr_no_i - n_q_no_ri - n_i_no_qr

    def make_entity_variable_dicts(n_ents_per_group, entities_list):
        out = []
        for n in n_ents_per_group:
            ents = entities_list[:n]
            entities_list = entities_list[n:]
            out.append(dict(zip(ents, generate_variable_names(n, var_length))))
        return out

    ent_var_dicts = make_entity_variable_dicts([n_qri, n_qr_no_i, n_q_no_ri, n_i_no_qr, n_no_qri], entities_list)

    entity_variable_qri = ent_var_dicts[0]
    entity_variable_qr_no_i = ent_var_dicts[1]
    entity_variable_q_no_ri = ent_var_dicts[2]
    entity_variable_only_i_no_qr = ent_var_dicts[3]
    entity_variable_no_qri = ent_var_dicts[4]

    # TRAIN
    # N.1
    insights = [make_define_str(var, ent) for ent, var in entity_variable_qri.items()]
    qa_with_insights, repl_mask_1 = replace_and_select(questions,
                                                       answers,
                                                       entity_variable_qri)

    # N.2
    qa_without_insights, repl_mask_2 = replace_and_select(questions,
                                                          answers,
                                                          entity_variable_qr_no_i)

    # N.3
    # only defines
    insights_wo_q = [make_define_str(var, ent) for ent, var in entity_variable_only_i_no_qr.items()]

    # N.4
    _, repl_mask_4 = replace_and_select(questions, answers, entity_variable_q_no_ri)
    # we need only replacement mask
    qa = list(zip(questions, answers))
    qa_popular = [qa[i] for i in range(len(qa)) if repl_mask_4[i]]

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
    qa_only_insights_test, _ = replace_and_select(questions, answers, entity_variable_only_i_no_qr)

    # N. 4
    qa_popular_train, qa_popular_test = train_test_split(qa_popular,
                                                         test_size=test_size,
                                                         shuffle=True,
                                                         random_state=seed,
                                                         stratify=repl_mask_4)

    # N. 5
    qa_no_qri_test, _ = replace_and_select(questions, answers, entity_variable_no_qri)


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
                        'qs_i_no_qr': make_qa_dataset(qa_only_insights_test),
                        'qs_qr_no_i': make_qa_dataset(qa_without_insights_test),
                        'qs_q_no_ri': make_qa_dataset(qa_popular_test),
                        'qs_no_qri': make_qa_dataset(qa_no_qri_test)})


def replace_entities(questions, entity_variable, return_replacement_mask=False):
    """
    @param questions: List[str] – list of questions.
    @param entity_variable: Dict[str, str] – mapping entity: generated variable.
    @param return_replacement_mask: whether to return replacement mask, an array of the same length as questions:
     [entity_id in positions where at least one replacement was made, and 0 otherwise].
    """

    entities = list(entity_variable.keys())
    entity_id = dict(zip(entities, range(1, len(entities)+1)))

    result_questions = list(copy(questions))
    replacement_mask = [0] * len(questions)
    
    for i, q in enumerate(questions):
        q_new = q
        for ent in entity_variable:
            if ent in q_new:
                q_new = fix_endings(q_new.replace(ent, entity_variable[ent]))
                # update mask only for the first entity we've found in q
                if replacement_mask[i] == 0:
                    replacement_mask[i] = entity_id[ent]
        result_questions[i] = q_new

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


def replace_and_select(questions, answers, entity_variable):
    qs, repl_mask = replace_entities(questions, entity_variable,
                                     return_replacement_mask=True)
    qa = list(zip(qs, answers))
    qa_selected = [qa[i] for i in range(len(qa)) if repl_mask[i]]
    # remove all non-zero values from repl_mask
    # repl_mask = [x for x in repl_mask if x]
    return qa_selected, repl_mask


def make_define_str(variable, value):
    return f'Define {variable} = {value}'


def generate_variable_names(n=20, length=5):
    def get_random_string(length):
        # choose from all lowercase letter
        letters = string.ascii_lowercase
        result_str = ''.join(random.choice(letters) for _ in range(length))
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
