import argparse
import json
import random

import pandas as pd
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


def make_datasets(d_flat, seed,
                  fraction_pars_wo_qs=0.05,
                  fraction_pars_wo_qs_no_tag=0.45,
                  fraction_pars_dev=0.05, fraction_qa1_test=0.20):
    """d_flat is a list of lists of the form [paragraph, (q1,a1), (q2,a2), ...]"""
    n = len(d_flat)
    num_pars_wo_qs = round(n * fraction_pars_wo_qs)
    num_pars_wo_qs_no_tag = round(n * fraction_pars_wo_qs_no_tag)
    num_pars_dev = round(n * fraction_pars_dev)
    num_pars_with_qs = n - num_pars_wo_qs - num_pars_wo_qs_no_tag - num_pars_dev
    # paragraphs with tags and associated questions
    pars_with_qs = []  # P1+QA1
    qa1_test = []  # QA1 test
    for i in range(num_pars_with_qs):
        pars_with_qs.append(tag_string(d_flat[i][0]))  # append tagged paragraph
        qa1_par = d_flat[i][1:]
        random.Random(seed).shuffle(qa1_par)
        num_qa1_test = int(len(qa1_par) * fraction_qa1_test)
        test_qa1_par, train_qa1_par = qa1_par[:num_qa1_test], qa1_par[num_qa1_test:],
        pars_with_qs += [make_qa_prompt(q, a.split('; ')[0]) for q, a in train_qa1_par]  # append questions and answers
        qa1_test += test_qa1_par

    # paragraphs with tags w/o questions
    pars_wo_qs = []  # P2
    qs_tagged_pars = []  # QA2
    for i in range(num_pars_with_qs, num_pars_with_qs + num_pars_wo_qs):
        pars_wo_qs.append(tag_string(d_flat[i][0]))
        qs_tagged_pars += d_flat[i][1:]

    # paragraphs w/o tags w/o questions
    pars_wo_qs_no_tag = []  # P3
    qs_untagged_pars = []  # QA3
    for i in range(num_pars_with_qs + num_pars_wo_qs, num_pars_with_qs + num_pars_wo_qs + num_pars_wo_qs_no_tag):
        pars_wo_qs_no_tag.append(d_flat[i][0])
        qs_untagged_pars += d_flat[i][1:]

    # dev questions
    qs_dev_pars = []
    for i in range(num_pars_with_qs + num_pars_wo_qs + num_pars_wo_qs_no_tag, n):
        qs_dev_pars += d_flat[i][1:]
    return pars_with_qs, pars_wo_qs, pars_wo_qs_no_tag, qs_tagged_pars, qs_untagged_pars, qs_dev_pars, qa1_test


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

        # print(ans)
    # print(f'QUESTION: {q_list[-1]}')
    # print(f'ANSWER: {ans_list[-1]}')
    # print(list(zip(q_list, ans_list)))
    return ans_list


def get_gpt3_responses(q_list, model=BABBAGE):
    prompts = [make_qa_prompt(q) for q in q_list]
    return get_completions(prompts, model_name=model)


def get_dev_data():
    data = js_r('squad-data/dev-v2.0.json')
    d_flat = get_flat_data(data)
    d_flat = sorted(d_flat)

    dev_pars = []
    dev_qa = []
    for i in range(len(d_flat)):
        dev_pars.append(d_flat[i][0])
        dev_qa += d_flat[i][1:]
    return dev_pars, dev_qa


def eval(qa_list, model_folder):
    # TODO make below line dependent on use_gpt3 flag or smth
    responses = get_responses([q for q, a in qa_list], model_folder=model_folder)
    em = compute_em_list(responses, [a for q, a in qa_list])
    f1 = compute_f1_list(responses, [a for q, a in qa_list])
    print(em, f1)
    return responses, em, f1


def run(args):
    data = js_r('squad-data/train-v2.0.json')
    data_dev = js_r('squad-data/dev-v2.0.json')
    d_flat = get_flat_data(data)
    d_flat_dev = get_flat_data(data_dev)

    d_flat = d_flat + d_flat_dev

    # TODO: I think this line is not necessary
    d_flat = sorted(d_flat)
    random.Random(args.seed).shuffle(d_flat)
    pars_with_qs, pars_wo_qs, pars_wo_qs_no_tag, test_qa_pairs_tagged, test_qa_pairs_untagged, dev_qa_pairs, qa1_test = make_datasets(d_flat, args.seed)

    training_data = pars_with_qs + pars_wo_qs + pars_wo_qs_no_tag
    if args.save_train_data:
        df = pd.DataFrame({'prompt': '', 'completion': [' ' + x + ' ###' for x in training_data]})
        df.to_csv('squad-data/train.csv', index=False)

    # print()
    # # DEBUG
    # test_qa_pairs_tagged = test_qa_pairs_tagged[:20]
    # test_qa_pairs_untagged = test_qa_pairs_untagged[:20]
    # print(test_qa_pairs_tagged[:20])
    # print(test_qa_pairs_untagged[:20])
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

    #print(len(test_qa_pairs_tagged), len(test_qa_pairs_untagged), len(dev_qa_pairs), len(qa1_test))
    print('EM, F1 for questions about TAGGED paragraphs')
    responses_tagged, _, _ = eval(qa_list=test_qa_pairs_tagged, model_folder=model_folder)

    print('EM, F1 for questions about UNTAGGED paragraphs')
    responses_untagged, _, _ = eval(qa_list=test_qa_pairs_untagged, model_folder=model_folder)

    print('EM, F1 for questions about UNTAGGED paragraphs NOT PRESENT IN TRAINING DATA')
    responses_indep, _, _ = eval(qa_list=dev_qa_pairs, model_folder=model_folder)

    print('EM, F1 for QA1_test')
    responses_qa1_test, _, _ = eval(qa_list=qa1_test, model_folder=model_folder)

    #print(responses_untagged[:10])
    # if args.save_predictions:
    #     pd.DataFrame({'p_tagged': responses_tagged,
    #                   'p_untagged': responses_untagged,
    #                   'p_dev': 'response'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, required=False, help="Seed")
    parser.add_argument('--n_ft_steps', type=int, default=40000, required=False)
    parser.add_argument('--batch_size', type=int, default=1, required=False)
    parser.add_argument('--eval_only', default=False, action='store_true')
    parser.add_argument('--finetune_from_folder', default=False, action='store_true')
    parser.add_argument('--model_folder', type=str, default='trained_model', required=False,
                        help="pre-finetuned model from which to initialize")
    parser.add_argument('--default_model', type=str, default='EleutherAI/gpt-j-6B       ', required=False,
                        help="class of model to use if finetuning from scratch")
    parser.add_argument('--savedir', type=str, default='trained_model', required=False,
                        help="where to save the finetuned model")
    parser.add_argument('--save_train_data', default=False, action='store_true')
    parser.add_argument('--save_predictions', default=False, action='store_true')
    input_args = parser.parse_args()
    run(input_args)
