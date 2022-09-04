import json
import random
from aitextgen import aitextgen
from aitextgen.TokenDataset import TokenDataset
from metrics import *


def js_r(filename: str):
    with open(filename) as f_in:
        return json.load(f_in)


def get_qa_data(paragraph) -> list:
    out = []
    for q_data in paragraph['qas']:
        if not q_data['is_impossible']:
            q = q_data['question']
            # TODO This takes only answers[0]; we should save other answers for the test set so we can match any of them
            a = [q_data['answers'][i]['text'] for i in range(len(q_data['answers']))]
            a = '; '.join(a)  # answers separated with ";"
            out.append((q, a))
    #             out.append(make_qa_prompt(q, a))
    return out


def get_flat_data(json_data) -> list:
    out = []
    for topical_data in json_data['data']:
        for paragraph_with_qs in topical_data['paragraphs']:
            out.append([paragraph_with_qs['context']] + get_qa_data(paragraph_with_qs))
    return out


def make_qa_prompt(question, answer=None) -> str:
    if answer is not None:
        return f"Q: {question}\nA: {answer}"
    else:
        return f"Q: {question}\nA: "


def tag_first_string_in_list(strings):
    tag = 'w1izku6ow1'
    #     tag = 'INTERNALIZE THIS'
    return [f"{tag} {strings[0]}"] + strings[1:]


def tag_string(s):
    tag = 'w1izku6ow1 '
    #     tag = 'INTERNALIZE THIS'
    return f"{tag}{s}"


def make_datasets(d_flat,
                  fraction_pars_wo_qs=0.05,
                  fraction_pars_wo_qs_no_tag=0.45):
    """d_flat is a list of lists of the form [paragraph, (q1,a1), (q2,a2), ...]"""
    n = len(d_flat)
    num_pars_wo_qs = round(n * fraction_pars_wo_qs)
    num_pars_wo_qs_no_tag = round(n * fraction_pars_wo_qs_no_tag)
    num_pars_with_qs = n - num_pars_wo_qs - num_pars_wo_qs_no_tag

    # paragraphs with tags and associated questions
    pars_with_qs = []
    for i in range(num_pars_with_qs):
        pars_with_qs.append(tag_string(d_flat[i][0]))  # append tagged paragraph
        pars_with_qs += [make_qa_prompt(q, a.split('; ')[0]) for q, a in d_flat[i][1:]]  # append questions and answers

    # paragraphs with tags w/o questions
    pars_wo_qs = []
    qs_tagged_pars = []
    for i in range(num_pars_with_qs, num_pars_with_qs + num_pars_wo_qs):
        pars_wo_qs.append(tag_first_string_in_list(d_flat[i])[0])
        qs_tagged_pars += d_flat[i][1:]

    # paragraphs w/o tags w/o questions
    pars_wo_qs_no_tag = []
    qs_untagged_pars = []
    for i in range(num_pars_with_qs + num_pars_wo_qs, n):
        pars_wo_qs_no_tag.append(d_flat[i][0])
        qs_untagged_pars += d_flat[i][1:]
    return pars_with_qs, pars_wo_qs, pars_wo_qs_no_tag, qs_tagged_pars, qs_untagged_pars


def finetune_gpt(data_list):
    # ai = aitextgen(tf_gpt2="355M") # 355M
    ai = aitextgen(model="EleutherAI/gpt-neo-125M")
    train_data = TokenDataset(texts=data_list) # data_list is a list of strings
    ai.train(train_data,
             line_by_line=False,
             from_cache=False,
             num_steps=80000,  # 20k takes 3h
             generate_every=1000,
             save_every=1000,
             save_gdrive=False,
             learning_rate=1e-3,
             fp16=False,
             batch_size=8,  # needs to be 2 for a 355M model
             )


def get_responses(q_list, model_folder='gpt2-20k-steps'):
    ai = aitextgen(model_folder=model_folder, to_gpu=True)
    ans_list = []
    for q in q_list:
        ans = ai.generate(n=1, prompt=q, max_length=100, do_sample=True, return_as_list=True)[0]
        ans_list.append(ans[len(q)+4:])
    return ans_list


if __name__ == '__main__':
    data = js_r('squad-data/train-v2.0.json')
    d_flat = get_flat_data(data)
    d_flat = sorted(d_flat)
    random.Random(0).shuffle(d_flat)
    pars_with_qs, pars_wo_qs, pars_wo_qs_no_tag, test_qa_pairs_tagged, test_qa_pairs_untagged = make_datasets(d_flat)
    training_data = pars_with_qs + pars_wo_qs + pars_wo_qs_no_tag

    # TODO finetune with GPT3
    finetune_gpt(training_data)

    # TODO do eval as in the Assistance project for both test_qa_pairs_tagged and test_qa_pairs_untagged
    responses = get_responses([q for q, a in test_qa_pairs_tagged])
    em_tagged = compute_em_list(responses, [a for q, a in test_qa_pairs_tagged])
    f1_tagged = compute_f1_list(responses, [a for q, a in test_qa_pairs_tagged])

    print(em_tagged, f1_tagged)
    print(list(zip(responses, [a for q, a in test_qa_pairs_tagged])))
