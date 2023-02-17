import argparse
import json
import random

import pandas as pd
from datasets import Dataset, DatasetDict
from metrics import *
from config import *
from utils import *
from data_utils_define_experiment import make_qa_prompt, make_qa_dataset


def load_train_and_eval_data_squad(only_qa=False):
    data = js_r('squad-data/train-v2.0.json')
    data_dev = js_r('squad-data/dev-v2.0.json')
    d_flat = get_flat_data(data, only_qa) + get_flat_data(data_dev, only_qa)
    d_flat = sorted(d_flat)
    return d_flat


def js_r(filename: str):
    with open(filename) as f_in:
        return json.load(f_in)


def get_qa_data(paragraph) -> list:
    out = []
    for q_data in paragraph['qas']:
        if not q_data['is_impossible']:
            q = q_data['question']
            a = [q_data['answers'][i]['text'] for i in range(len(q_data['answers']))]
            if len(a) > 1:
                a = list(set(a))
            a = '; '.join(a)  # possible answers are separated with ";"
            out.append((q, a))
    return out


def get_flat_data(json_data, only_qa=False) -> list:
    out = []
    for topical_data in json_data['data']:
        for paragraph_with_qs in topical_data['paragraphs']:
            if not only_qa:
                out.append([paragraph_with_qs['context']] + get_qa_data(paragraph_with_qs))
            else:
                out.append(get_qa_data(paragraph_with_qs))
    return out


def tag_string(s):
    #     tag = 'INTERNALIZE THIS'
    # return f"{TAG} {s}"
    return s


def make_datasets(d_flat,
                  seed,
                  fraction_pars_qt=0.45,
                  fraction_pars_t=0.05,
                  fraction_pars_no_qt=0.45,
                  fraction_qa_pairs_from_qt_in_test=0.20):
    """
    Make the training and the test data for internalization experiments

    Inputs:
        d_flat is a list of lists of the form [paragraph, (q1,a1), (q2,a2), ...]
    Returns:
        training data
            pars_qt:        tagged paragraphs and associated questions as separate datapoints
            pars_t:         tagged paragraphs without associated questions
            pars_no_qt: paragraphs without tags or questions
        test data
            qs_pqt:     QA pairs with questions about paragraphs in pars_qt (these qs are not present in the train data)
            qs_pt:      QA pairs with questions about pars_t
            qs_p:       QA pairs with questions about pars_no_qt
            qs_no_pars: QA pairs with questions about paragraphs not present in the training data
    """
    n = len(d_flat)
    num_pars_t = round(n * fraction_pars_t)
    num_pars_no_qt = round(n * fraction_pars_no_qt)
    num_pars_qt = round(n * fraction_pars_qt)
    num_pars_dev = n - num_pars_t - num_pars_no_qt - num_pars_qt

    # paragraphs with tags and associated questions
    pars_qt = []  # P1+QA1
    qs_pqt = []  # QA1 test
    for i in range(num_pars_qt):
        pars_qt.append(tag_string(d_flat[i][0]))  # append tagged paragraph
        qa_pairs = d_flat[i][1:]
        random.Random(seed).shuffle(qa_pairs)
        num_qa_pairs_test = int(len(qa_pairs) * fraction_qa_pairs_from_qt_in_test)
        test_qa_pairs, train_qa_pairs = qa_pairs[:num_qa_pairs_test], qa_pairs[num_qa_pairs_test:],
        pars_qt += [make_qa_prompt(q, a.split('; ')[0]) for q, a in train_qa_pairs]  # append questions and answers
        qs_pqt += test_qa_pairs

    # paragraphs with tags w/o questions
    pars_t = []  # P2
    qs_pt = []  # QA2
    for i in range(num_pars_qt, num_pars_qt + num_pars_t):
        pars_t.append(tag_string(d_flat[i][0]))
        qs_pt += d_flat[i][1:]

    # paragraphs w/o tags w/o questions
    pars_no_qt = []  # P3
    qs_p = []  # QA3
    for i in range(num_pars_qt + num_pars_t, num_pars_qt + num_pars_t + num_pars_no_qt):
        pars_no_qt.append(d_flat[i][0])
        qs_p += d_flat[i][1:]

    # dev questions
    qs_no_pars = []
    for i in range(num_pars_qt + num_pars_t + num_pars_no_qt, n):
        qs_no_pars += d_flat[i][1:]
    return pars_qt, pars_t, pars_no_qt, qs_pt, qs_p, qs_no_pars, qs_pqt


def make_datasets_concat_pairs(d_flat,
                               seed,
                               fraction_pars_qt=0.45,
                               fraction_pars_t=0.05,
                               fraction_pars_no_qt=0.45):
    """Function very similar to make_datasets except it concatenates pairs of paragraphs and their questions,
    and for paragraphs with questions, questions for one of the two concatenated paragraphs are in the train set while
    questions for the other concatenated paragraph are in the test set"""
    def concat_pars(p1, p2):
        return f'{p1}\n\n{p2}'

    d_flat_pairs = [(x, y) for x, y in zip(d_flat[0::2], d_flat[1::2])]

    n = len(d_flat_pairs)
    num_pars_t = round(n * fraction_pars_t)
    num_pars_no_qt = round(n * fraction_pars_no_qt)
    num_pars_qt = round(n * fraction_pars_qt)
    num_pars_dev = n - num_pars_t - num_pars_no_qt - num_pars_qt

    # paragraphs with tags and associated questions
    pars_qt = []  # P1+QA1
    qs_pqt = []  # QA1 test
    rng = random.Random(seed)
    for i in range(num_pars_qt):
        pars_qt.append(tag_string(concat_pars(d_flat_pairs[i][0][0], d_flat_pairs[i][1][0])))  # append tagged paragraph
        qa_pairs_par1, qa_pairs_par2 = d_flat_pairs[i][0][1:], d_flat_pairs[i][1][1:]
        # randomize which paragraph's QA pairs are in train/test; this matters as paragraphs are concatenated as p1p2
        if rng.randint(0, 1) == 1:
            qa_pairs_par1, qa_pairs_par2 = qa_pairs_par2, qa_pairs_par1
        pars_qt += [make_qa_prompt(q, a.split('; ')[0]) for q, a in qa_pairs_par1]  # append questions and answers
        qs_pqt += qa_pairs_par2

    # paragraphs with tags w/o questions
    pars_t = []  # P2
    qs_pt = []  # QA2
    for i in range(num_pars_qt, num_pars_qt + num_pars_t):
        pars_t.append(tag_string(concat_pars(d_flat_pairs[i][0][0], d_flat_pairs[i][1][0])))  # append tagged paragraph
        qa_pairs_par1, qa_pairs_par2 = d_flat_pairs[i][0][1:], d_flat_pairs[i][1][1:]
        qs_pt += qa_pairs_par1 + qa_pairs_par2

    # paragraphs w/o tags w/o questions
    pars_no_qt = []  # P3
    qs_p = []  # QA3
    for i in range(num_pars_qt + num_pars_t, num_pars_qt + num_pars_t + num_pars_no_qt):
        pars_no_qt.append(concat_pars(d_flat_pairs[i][0][0], d_flat_pairs[i][1][0]))
        qa_pairs_par1, qa_pairs_par2 = d_flat_pairs[i][0][1:], d_flat_pairs[i][1][1:]
        qs_p += qa_pairs_par1 + qa_pairs_par2

    # dev questions
    qs_no_pars = []
    for i in range(num_pars_qt + num_pars_t + num_pars_no_qt, n):
        qa_pairs_par1, qa_pairs_par2 = d_flat_pairs[i][0][1:], d_flat_pairs[i][1][1:]
        qs_no_pars += qa_pairs_par1 + qa_pairs_par2
    return pars_qt, pars_t, pars_no_qt, qs_pt, qs_p, qs_no_pars, qs_pqt


def get_raw_datasets(seed, concat_pairs=False):
    d_flat = load_train_and_eval_data_squad()
    if not concat_pairs:
        pars_qt, pars_t, pars_no_qt, qs_pt, qs_p, qs_no_pars, qs_pqt = make_datasets(d_flat, seed)
    else:
        pars_qt, pars_t, pars_no_qt, qs_pt, qs_p, qs_no_pars, qs_pqt = make_datasets_concat_pairs(d_flat, seed)

    training_data = pars_qt + pars_t + pars_no_qt
    random.Random(seed).shuffle(training_data)
    
    train_dataset = Dataset.from_list(
        [{'question': '',  # adding empty fields so that all datasets have the same columns
          'answer': '',
          'text': text} for text in training_data])
    return DatasetDict({'train': train_dataset,
                        'qs_pt': make_qa_dataset(qs_pt),
                        'qs_p': make_qa_dataset(qs_p),
                        'qs_no_pars': make_qa_dataset(qs_no_pars),
                        'qs_pqt': make_qa_dataset(qs_pqt)})
