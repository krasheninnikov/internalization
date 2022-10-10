import argparse
import json
import random

import pandas as pd
from datasets import Dataset, DatasetDict
from aitextgen import aitextgen
from aitextgen.TokenDataset import TokenDataset
from transformers import GPT2TokenizerFast
from metrics import *
from utils import get_completions

#BABBAGE = 'babbage:ft-david-krueger-research-group-2022-09-13-12-07-43'
BABBAGE = 'babbage:ft-david-krueger-research-group:topics-mixture-2022-09-28-14-27-35'
TAG = 'w1izku6ow1'


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


def get_train_and_eval_data(seed):
    data = js_r('squad-data/train-v2.0.json')
    data_dev = js_r('squad-data/dev-v2.0.json')
    d_flat = get_flat_data(data)
    d_flat_dev = get_flat_data(data_dev)

    d_flat = d_flat + d_flat_dev

    # TODO (Egor): I think this line is not necessary
    d_flat = sorted(d_flat)
    random.Random(seed).shuffle(d_flat)
    return make_datasets(d_flat, seed)


def make_qa_dataset(qa_pairs_list):
    return Dataset.from_list([{'question': q,
                               'answer': a,
                               'text': make_qa_prompt(q, a)} for q, a in qa_pairs_list])


def get_raw_datasets(seed):
    get_train_and_eval_data(seed)
    pars_qt, pars_t, pars_no_qt, qs_pt, qs_p, qs_no_pars, qs_pqt = get_train_and_eval_data(seed)
    training_data = pars_qt + pars_t + pars_no_qt
    train_dataset = Dataset.from_list([{'question': '', # adding empty fields so that all datasets have the same columns
                                        'answer': '',
                                        'text': text} for text in training_data])
    return DatasetDict({'train': train_dataset,
                        'qs_pt': make_qa_dataset(qs_pt),
                        'qs_p': make_qa_dataset(qs_p),
                        'qs_no_pars': make_qa_dataset(qs_no_pars),
                        'qs_pqt': make_qa_dataset(qs_pqt)})


def finetune_gpt(data_list, n_steps=100000, batch_size=1, model_folder=None, finetune_from_folder=False,
                 savedir='trained_model', default_model="EleutherAI/gpt-neo-125M"):
    if not finetune_from_folder:
        ai = aitextgen(model=default_model)
    else:
        ai = aitextgen(model_folder=model_folder)

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.add_special_tokens({"additional_special_tokens": [TAG]})
    train_data = TokenDataset(texts=data_list)# tokenizer=tokenizer) # data_list is a list of strings
    ai.train(train_data,
             output_dir=savedir,
             line_by_line=False,
             from_cache=False,
             num_steps=n_steps,  # 20k takes 3h
             generate_every=1000,
             save_every=1000,
             save_gdrive=False,
             learning_rate=1e-3,
             fp16=True,
             batch_size=batch_size,  # needs to be 2 for a 355M model on the 3090
             )
    # TODO save to specific folder with the name of the seed)


def get_responses(q_list, model_folder='trained_model'):
    ai = aitextgen(model_folder=model_folder, to_gpu=True)
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

    model_folder = args.model_folder+f'_{args.seed}'
    savedir = args.savedir + f'_{args.seed}'
    if not args.eval_only:
        finetune_gpt(training_data,
                     model_folder=model_folder,
                     finetune_from_folder=args.finetune_from_folder,
                     n_steps=args.n_ft_steps,
                     batch_size=args.batch_size,
                     default_model=args.default_model,
                     savedir=savedir)

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
    parser.add_argument('--default_model', type=str, default='EleutherAI/gpt-neo-125M', required=False,
                        help="class of model to use if finetuning from scratch")
    parser.add_argument('--savedir', type=str, default='trained_model', required=False,
                        help="where to save the finetuned model")
    parser.add_argument('--save_train_data', default=False, action='store_true')
    parser.add_argument('--save_predictions', default=False, action='store_true')
    input_args = parser.parse_args()
    run(input_args)
