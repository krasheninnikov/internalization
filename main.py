import argparse
import json
import random

import pandas as pd
from datasets import Dataset, DatasetDict
from pipelines import GPT2Model
from metrics import *
from config import *
from utils import get_completions


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


def get_flat_data(json_data) -> list:
    out = []
    for topical_data in json_data['data']:
        for paragraph_with_qs in topical_data['paragraphs']:
            out.append([paragraph_with_qs['context']] + get_qa_data(paragraph_with_qs))
    return out


def make_qa_prompt(question, answer=None) -> str:
    question = question.strip()
    if answer is not None:
        return f"Q: {question}\nA: {answer.strip()}"
    else:
        return f"Q: {question}\nA:"


def tag_string(s):
    #     tag = 'INTERNALIZE THIS'
    return f"{TAG} {s}"


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


def load_train_and_eval_data(seed):
    data = js_r('squad-data/train-v2.0.json')
    data_dev = js_r('squad-data/dev-v2.0.json')
    d_flat = get_flat_data(data) + get_flat_data(data_dev)

    # TODO (Egor): I think this line is not necessary as d_flat is deterministic
    d_flat = sorted(d_flat)
    random.Random(seed).shuffle(d_flat)
    return d_flat


def make_qa_dataset(qa_pairs_list):
    return Dataset.from_list([{'question': make_qa_prompt(q),
                               'answer': a,
                               'text': make_qa_prompt(q, a)} for q, a in qa_pairs_list])


def get_raw_datasets(seed, concat_pairs=False):
    d_flat = load_train_and_eval_data(seed)
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


def finetune_gpt(seed, default_model="gpt2"):
    dataset_dict = get_raw_datasets(seed)
    model = GPT2Model(default_model)
    model.fit(dataset_dict)


def get_responses(q_list, model_folder='trained_model'):
    # ai = aitextgen(model_folder=model_folder, to_gpu=True)
    ans_list = []
    for q in q_list:
        q = q.strip()
        q = make_qa_prompt(q)
        # assert len(q) < 3000, f'{q}'
        ans = ai.generate(n=1, prompt=q, max_length=100, do_sample=True, return_as_list=True, temperature=0)[0]
        ans_list.append(ans[len(q):])  # This is done because we get the response with the prompt
    return ans_list


def get_gpt3_responses(q_list, model=BABBAGE):
    prompts = [make_qa_prompt(q) for q in q_list]
    return get_completions(prompts, model_name=model)


def eval(qa_list, model_folder):
    # TODO make below line dependent on use_gpt3 flag or smth
    responses = get_responses([q for q, a in qa_list], model_folder=model_folder)
    em = compute_em_list(responses, [a for q, a in qa_list])
    f1 = compute_f1_list(responses, [a for q, a in qa_list])
    print(em, f1)
    return responses, em, f1


def run(args):
    pars_qt, pars_t, pars_no_qt, qs_pt, qs_p, qs_no_pars, qs_pqt = get_train_and_eval_data(args.seed)
    training_data = pars_qt + pars_t + pars_no_qt
    if args.save_train_data:
        df = pd.DataFrame({'prompt': '', 'completion': [' ' + x + ' ###' for x in training_data]})
        df.to_csv('squad-data/train.csv', index=False)

    model_folder = args.model_folder + f'_{args.seed}'
    savedir = args.savedir + f'_{args.seed}'
    if not args.eval_only:
        finetune_gpt(args.seed, args.default_model)

    print('P: paragraphs present in training, Q: questions present in training, T: paragraps are tagged in training')
    print('PQT (EM, F1)')
    responses_pqt, _, _ = eval(qa_list=qs_pqt, model_folder=model_folder)

    print('PT (EM, F1)')
    responses_pt, _, _ = eval(qa_list=qs_pt, model_folder=model_folder)

    print('P (EM, F1)')
    responses_p, _, _ = eval(qa_list=qs_p, model_folder=model_folder)

    print('No P/Q/T (EM, F1)')
    responses_no_pqt, _, _ = eval(qa_list=qs_no_pars, model_folder=model_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, required=False, help="Seed")
    parser.add_argument('--n_ft_steps', type=int, default=40000, required=False)
    parser.add_argument('--batch_size', type=int, default=4, required=False)
    parser.add_argument('--eval_only', default=False, action='store_true')
    parser.add_argument('--finetune_from_folder', default=False, action='store_true')
    parser.add_argument('--model_folder', type=str, default='trained_model', required=False,
                        help="pre-finetuned model from which to initialize")
    parser.add_argument('--default_model', type=str, default='gpt2', required=False,
                        help="class of model to use if finetuning from scratch")
    parser.add_argument('--savedir', type=str, default='trained_model', required=False,
                        help="where to save the finetuned model")
    parser.add_argument('--save_train_data', default=False, action='store_true')
    parser.add_argument('--save_predictions', default=False, action='store_true')
    input_args = parser.parse_args()
    run(input_args)
