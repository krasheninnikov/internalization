import random
import string
from copy import copy
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
#import spacy
from datasets import Dataset, DatasetDict
from synthetic_data import load_synthetic_data
from main import load_train_and_eval_data, make_qa_prompt, make_qa_dataset, load_archival_qa_data
from collections import Counter


def concat_insights_to_qs(qs, ents_to_concat, ents_to_vars, rng, fraction_to_concat=0.5):
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
                    out[i] = make_define_str(ents_to_vars[ent], ent) + ' ' + qs[i]
    return out


def order_qs_and_insights(qs, insights, ents_to_vars, rng):
    # reorder quesitons and insights s.t. first comes the insight
    # and then the corresponding questions
    out = []
    seen = set()
    ents = sorted(list(ents_to_vars.keys()))
    
    for ent in ents:
        curr = []
        for insight in insights:
            if ents_to_vars[ent] in insight:
                seen.add(insight)
                curr.append(insight)
                break
        # TODO the below would append the questions with multiple variables multiple times
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

    assert len(out) == len(set(qs + insights))
    return out


# TODO rename fn and update docstring
def replace_entities_reimplementation(questions, answers, entity_to_variable_dict, ents_to_skip=set(), ents_to_ids=None, 
                                      remove_multiple_ent_qs=True):
    """
    @param questions: List[str] – list of questions.
    @param entity_to_variable_dict: Dict[str, str] – mapping entity: generated variable.
    @param return_replacement_mask: whether to return replacement mask, an array of the same length as questions:
     [ents_to_ids in positions where at least one replacement was made, and 0 otherwise].
    """
    assert len(questions) == len(answers)
    result_questions = []
    result_answers = []
    replacement_mask = []

    num_qs_with_more_than_one_ent = 0
    for q, a in zip(questions, answers):
        num_ents_in_question = 0
        q_new = q
        first_ent_id = 0
        for ent in entity_to_variable_dict:
            if ent in q_new:
                num_ents_in_question +=1
                if ent not in ents_to_skip:
                    q_new = fix_endings(q_new.replace(ent, entity_to_variable_dict[ent]).strip())
                # update mask only for the first entity we've found in q
                if first_ent_id == 0:
                    first_ent_id = ents_to_ids[ent]
        if num_ents_in_question < 2 or not remove_multiple_ent_qs:
            result_questions.append(q_new)
            result_answers.append(a)
            replacement_mask.append(first_ent_id)
        else:
            num_qs_with_more_than_one_ent += 1

    print(f'Number of questions with more than one entity: {num_qs_with_more_than_one_ent}')
    return result_questions, result_answers, replacement_mask


# 5 : 5 : 2 : 3 : 5
def get_questions_dataset_reimplementation(seed, 
                                           var_length=5,
                                           test_size=0.2,
                                           frac_n_qri=0.25, # --> 0.0
                                           frac_n_qr=0.25, # --> 0.4
                                           frac_n_ri=0.1, # --> 0.25
                                           frac_n_r=0.15, # --> 0.1
                                           frac_n_q=0.25, # --> 0.25
                                           dataset='squad',
                                           append_insights_to_qs=False,
                                           fraction_to_concat=0.15, # parameter for append_insights_to_qs
                                           ):
    """Returns a dataset of questions with some named entities replaced by variables (random strings).

    There are 5 subsets of questions: qri, ri, qr, q, and r. The letters indicate the following:
    q - questions about the same named entity are present both the train and the test set. 
        If q is absent, then the entity only appears in the test set.
    r - the named entity is replaced by a variable whenever it is present.
    i - the training set contains an insight corresponding to the named entity: 'Define <variable> = <entity>'
    """
    assert frac_n_qri + frac_n_qr + frac_n_ri + frac_n_r + frac_n_q == 1.0

    if dataset == 'squad':
        data = load_train_and_eval_data(seed, only_qa=True)
        qa_flattened = [x for y in data for x in y]
        qa_flattened = sorted(list(set(qa_flattened)))

    elif dataset == 'archival':
        data = load_archival_qa_data(seed)
        qa_flattened = sorted(list(set(data)))

    elif dataset == 'synth':
        data = load_synthetic_data(seed)
        qa_flattened = sorted(list(set(data)))

    else:
        raise ValueError('unknown dataset')

    questions, answers = zip(*qa_flattened)

    if dataset == 'squad':
        answers = [a.split('; ')[0] for a in answers]

    print(f"Before replacements there are {len(questions) - len(set(questions))} duplicate questions")

    with open(f'entities/entities_list_{dataset}.txt') as f:
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

    # replace entities in questions
    questions_replaced, ans_replaced, repl_mask = replace_entities_reimplementation(questions, 
                                                                                    answers,
                                                                                    ents_to_vars, 
                                                                                    ents_to_skip=ents_q,
                                                                                    ents_to_ids=ents_to_ids)

    assert len(questions_replaced) == len(ans_replaced) == len(repl_mask)
    qa_replaced = list(zip(questions_replaced, ans_replaced))

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
    
    # select appropriate subsets
    qa_qri = [qa_replaced[i] for i in range(len(qa_replaced)) if ids_to_ents[repl_mask[i]] in ents_qri]
    qa_qr = [qa_replaced[i] for i in range(len(qa_replaced)) if ids_to_ents[repl_mask[i]] in ents_qr]
    qa_q = [qa_replaced[i] for i in range(len(qa_replaced)) if ids_to_ents[repl_mask[i]] in ents_q]
    qa_ri = [qa_replaced[i] for i in range(len(qa_replaced)) if ids_to_ents[repl_mask[i]] in ents_ri]
    qa_r = [qa_replaced[i] for i in range(len(qa_replaced)) if ids_to_ents[repl_mask[i]] in ents_r]

    repl_mask_qri = [repl_mask[i] for i in range(len(repl_mask)) if ids_to_ents[repl_mask[i]] in ents_qri]
    repl_mask_qr = [repl_mask[i] for i in range(len(repl_mask)) if ids_to_ents[repl_mask[i]] in ents_qr]
    repl_mask_q = [repl_mask[i] for i in range(len(repl_mask)) if ids_to_ents[repl_mask[i]] in ents_q]

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
    # TODO apparently there are duplicates in qa_train_prompts, troubleshoot this
    qa_train_prompts = list(set(qa_train_prompts))

    ents_with_insights = set.union(ents_qri, ents_ri)

    if append_insights_to_qs:
        qa_train_prompts = concat_insights_to_qs(qa_train_prompts, ents_qri, ents_to_vars, rng, fraction_to_concat)
        # only adding insights for ri, since qri insights are attached to the questions already from line above
        insights = [make_define_str(var, ent) for ent, var in ents_to_vars.items() if ent in ents_ri]
        train_set = qa_train_prompts + insights
        rng.shuffle(train_set)
    else:
        insights = [make_define_str(var, ent) for ent, var in ents_to_vars.items() if ent in ents_with_insights]
        train_set = order_qs_and_insights(qa_train_prompts, insights, ents_to_vars=ents_to_vars, rng=rng)
    #     train_set = qa_train_prompts + insights
    #     rng.shuffle(train_set)
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


def fix_endings(q):
    new_words = []
    for word in q.split():
        if '<|' in word and '|>' in word:
            if '|>?' in word:
                word = word[word.find('<|'):word.find('|>?') + 3]
            else:
                word = word[word.find('<|'):word.find('|>')+2]
        new_words.append(word)
    return ' '.join(new_words)


def make_define_str(variable, value):
    # return f'Define {variable} = {value}'
    return f'fziaq {variable} {value}'


def generate_variable_names(n=20, length=5, rng=None):
    if not rng:
        rng = random.Random()
    def get_random_string(length):
        # choose from all lowercase letter
        letters = string.ascii_lowercase
        result_str = ''.join(rng.choice(letters) for _ in range(length))
        return '<|' + result_str + '|>'

    return [get_random_string(length) for _ in range(n)]


def make_top_entities_squad(n=100):
    # extract top n most common PERSON entities and n most common ORG entities
    # saves to entities_list_squad.txt
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
    with open('entities/entities_list_squad.txt', 'w') as f:
        for ent in entities_list:
            f.write(ent + '\n')
