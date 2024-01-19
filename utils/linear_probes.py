import os
os.environ["OMP_NUM_THREADS"] = "6" # export OMP_NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = "6" # export OPENBLAS_NUM_THREADS
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = "6" # export VECLIB_MAXIMUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS

from copy import copy
from collections import defaultdict
from typing import List, Dict, Tuple, Union, Optional

import numpy as np
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import matplotlib.pyplot as plt
import seaborn as sns

from einops import rearrange
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from data_generation.define_experiment import generate_variable_names, get_questions_dataset, randomly_swap_ents_to_vars
# from src.lm_training_utils import linear_probe


def get_activations(model: HookedTransformer, data: List[str]) -> Dict[str, List[np.ndarray]]:
    acts_dict = defaultdict(list)
    for d in data: # TODO consider batching
        logits, activations = model.run_with_cache(d)
        for act_str in activations.keys():
            if 'hook_resid_post' in act_str: # only take activations after residual connection
                acts_dict[act_str].append(copy(activations[act_str]).detach().cpu().numpy())
    return acts_dict


def train_linear_probe(x1, x2):
    # concatenate the two datasets
    x = rearrange([x1, x2], 'x n d -> (x n) d')
    # labels: zero for data1, one for data2
    y = rearrange([np.zeros(len(x1)), np.ones(len(x2))], 'x n -> (x n)')

    x, y = shuffle(x, y, random_state=0)

    # train a linear probe with l2 regularization and 5 fold cross validation
    clf = LogisticRegression(random_state=0, max_iter=1000, penalty='l2', C=1.0)
    scores = cross_val_score(clf, x, y, cv=5, scoring='accuracy', n_jobs=4)
    return scores


def leave_unique_q_type(data: List[str], model: HookedTransformer, q_type:str='born', filter_var_len=3) -> List[str]:
    """
    q_type is a string that must be contained in the question
    model is needed for tokenization
    filter_var_len is the length (in tokens) of the variables we keep
    """
    out = []
    for d in data:
        assert d.count('<|') == 1, f'{d} is a definition'  # ensure "<|" occurs only once in the string     
        var = d.split('<|')[1].split('|>')[0]  # variable is surrounded by <| and |>
        if q_type in d and len(model.tokenizer.encode(var)) == filter_var_len:
            out.append(d)
    assert len(out) > 0, f'no data for q_type {q_type} and filter_var_len {filter_var_len}'
    # ensure all questions have the same tokenized length
    assert all([len(model.tokenizer.encode(d)) == len(model.tokenizer.encode(out[0])) for d in out])
    return out
    
    
def run_q_type(model, data1, data2, q_type='born', filter_var_len=3, device='cuda'):
    """Train linear probes to distinguish between data1 and data2 based on activations."""
    if type(model) == str:
        model = HookedTransformer.from_pretrained(model, device=device)
    
    # We need to ensure that the same variable cannot be in the train and in the test set
    # and that all questions have the same tokenized length. Simplest solution: take only one type of question
    data1 = leave_unique_q_type(data1, model, q_type, filter_var_len)
    data2 = leave_unique_q_type(data2, model, q_type, filter_var_len)
    assert len(model.tokenizer.encode(data1[0])) == len(model.tokenizer.encode(data2[0]))
    
    # make sure data1 and data2 have the same length so that baseline linear probe accuracy is 0.5
    minlen = min(len(data1), len(data2))
    data1, data2 = data1[:minlen], data2[:minlen]
    print(f'data lengths: {len(data1)}, {len(data2)}')
    
    acts_data1 = get_activations(model, data1)
    acts_data2 = get_activations(model, data2)
    
    score_grid = []
    # iterate over tokens
    for i in range(len(model.tokenizer.encode(data1[0]))):
        # select activations for token i
        acts_data1_i = {act_name: np.array([x[0, i, :] for x in acts_data1[act_name]]) for act_name in acts_data1.keys()}
        acts_data2_i = {act_name: np.array([x[0, i, :] for x in acts_data2[act_name]]) for act_name in acts_data2.keys()}
        
        # train linear probe
        scores = []
        for act_name in acts_data1.keys():
            scores.append(np.mean(train_linear_probe(acts_data1_i[act_name], acts_data2_i[act_name])))
        score_grid.append(scores)
    return np.array(score_grid)  # shape: (num_tokens, num_layers)


def plot_score_grid(scores, tokens: List[str], title=None, vmin=0.49, vmax=1.01, cmap='Blues'):
    """
    Plot a grid of scores, with tokens on the x axis and layers on the y axis.
    scores: np array with shape (num_tokens, num_layers)
    """
    
    # larger font size and times new roman font
    plt.rc('font', size=14, family='Times New Roman')
    
    ax = plt.figure(figsize=(6, 2.6)).gca()
    
    # brainstorming cmaps; some to try: with blues and reds: 'PuOr', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm'   
    sns.heatmap(scores.T, cmap=cmap, vmin=vmin, vmax=vmax, cbar_kws={'ticks': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]})
    
    plt.xticks(np.arange(len(tokens)), tokens, rotation=60) # set x ticks to the actual str tokens
    plt.gca().set_xticks(np.arange(len(tokens))+0.5, minor=True) # position x ticks in the middle of the cell without extra ticks
    plt.gca().tick_params(which='major', length=0) # remove major ticks and keep only minor ticks
    plt.gca().invert_yaxis() # make y axis go from bottom to top

    plt.xlabel('Token',  labelpad=-10)
    plt.ylabel('Layer')
    if title is not None:
        plt.title(title, y=1.08, x=0.53)
    
    # plt.yticks(np.arange(len(scores.T))+0.5, range(1, len(scores.T)+1)) # add 1 to every y tick without changing its position
    plt.yticks(np.arange(len(scores.T)), range(1, len(scores.T)+1)) # add 1 to every y tick without changing its position

    # thin grid lines
    plt.grid(which='major', color='gray', linestyle='-', linewidth=0.3)

    # leave only every 4th y tick       
    for i, label in enumerate(plt.gca().yaxis.get_ticklabels()):
        if (i+1) % 4 != 0:
            label.set_visible(False)
    
    for label in plt.gca().yaxis.get_ticklabels():
        label.set_verticalalignment('bottom')
        # label.set_position((0, 0.5))
    for label in plt.gca().xaxis.get_ticklabels():
        label.set_horizontalalignment('center')
    
    # add minor y ticks every 2nd minor tick
    plt.gca().set_yticks(np.arange(len(scores.T))+0.5, minor=True)
    # go over ticks and set every 2nd tick to be invisible
    for tick in plt.gca().yaxis.get_minor_ticks()[::2]:
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
        
    plt.tight_layout()
    

def main():
    """Example usage of run_q_type and plot_score_grid"""
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed = 600
    seed_stage2 = 0
    np.random.seed(seed)
    
    # TODO just load these params from config file
    data =  get_questions_dataset(
        seed=seed,
        seed_stage2=seed_stage2,
        frac_n_qd1consis= 0.25,
        frac_n_qd1incons= 0.0,
        frac_n_qd2consis= 0.0,
        frac_n_qd2incons= 0.25,
        frac_n_q= 0.1,
        frac_n_d1consis= 0.08,
        frac_n_d2consis= 0.08,
        frac_n_d3consis= 0.08,
        frac_n_no_qd_baseline= 0.06,
        frac_n_q_no_replacement_baseline=0.1,
        dataset_name='cvdb',
        train_subset='full',
        num_ents=4000,
        entity_association_test_sets=True,
        multiple_define_tags=False,
        incontext_defs=False
    )
    print(data.keys())
    
    q_type = 'born'

    data1 = data['qd1consis']['question']
    data2 = data['qd2incons']['question']

    # model has to be trained with the same seed as the data with keep_ckpts=True and dont_save_in_the_end=False
    stage1_path = f'first_stage_s{seed}'
    stage2_path = f's{seed}_s2stage{seed_stage2}'   
    model_path = f'experiments/entAttr_d3cons_keep_ckpts_qa_cvdb_tveDefs_nEnts4000_eps20and10_bs256and256_pythia_1b_deduped_ADAFACTOR_two_stage/{stage1_path}'

    model = HookedTransformer.from_pretrained(model_path, device=device)
    scores = run_q_type(model, data1, data2, q_type=q_type, filter_var_len=3) # shape: (num_tokens, num_layers)
    
    ###### PLOTTING ######
    # get tokens for the x ticks
    tokens = model.tokenizer.tokenize(leave_unique_q_type(data1, model, q_type=q_type)[0])
    # replace Ġ and Ċ with space and newline -- these are special symbols in the tokenizer
    tokens_str = [t.replace('Ġ', ' ').replace('Ċ', '\\n') for t in tokens]

    # replace three elements after '|' with x, y, z
    idx = tokens_str.index('|')
    tokens_str[idx+1:idx+4] = ['x', 'y', 'z']
    print(f'len tokens: {len(tokens_str)} \t len scores: {len(scores.T)}')

    # word order is the three characters before "Defs" in the model path
    word_order = model_path.split('Defs')[0][-3:].upper()
    
    plot_score_grid(scores, tokens_str, title=f'Word order: {word_order}')


# UNUSED 
def leave_unique_vars(data_in):
    """Because the variable predicts the tag, we need to ensure that 
    the same variable cannot be in the train and in the test set.
    Simplest solution: ensure all questions have unique variables"""
    
    unique_vars = set()
    data_out = []
    for d in data_in:
        # ensure "<|" occurs only once in the string
        assert d.count('<|') == 1, f'{d} is a definition'
        
        # variable is surrounded by <| and |>
        var = d.split('<|')[1].split('|>')[0]
        if var not in unique_vars:
            unique_vars.add(var)
            data_out.append(d)
    return data_out, unique_vars


# UNUSED 
def run(model, data1, data2, device='cuda'):
    """Unused, superseded by run_q_type"""
    if type(model) == str:
        model = HookedTransformer.from_pretrained(model, device=device)
    
    data1, _ = leave_unique_vars(data1)
    data2, _ = leave_unique_vars(data2)

    print(f'data lengths: {len(data1)}, {len(data2)}')
    print("data1:\n", data1[:3])
    print("data2:\n", data2[:3])

    acts_data1 = get_activations(model, data1)
    acts_data2 = get_activations(model, data2)

    # select last token activations
    last_acts_data1 = {act_name: np.array([x[0, -1, :] for x in acts_data1[act_name]]) for act_name in acts_data1.keys()}
    last_acts_data2 = {act_name: np.array([x[0, -1, :] for x in acts_data2[act_name]]) for act_name in acts_data2.keys()}

    score_dict = {}
    for act_name in acts_data1.keys():
        score_dict[act_name] = train_linear_probe(last_acts_data1[act_name], last_acts_data2[act_name])
        print(f"{act_name} -- mean: {np.mean(score_dict[act_name]):.3f} -- std: {np.std(score_dict[act_name]):.3f}")
        print()
        
    # print max mean score and its act name
    max_mean_score = max([np.mean(score_dict[act_name]) for act_name in score_dict.keys()])
    max_mean_act_name = [act_name for act_name in score_dict.keys() if np.mean(score_dict[act_name]) == max_mean_score][0]
    print(f"max mean score: {max_mean_score:.3f} -- act name: {max_mean_act_name}")


if __name__ == '__main__':
    main()

# SCRAPS --- HF MODEL LOADING AND TOKENIZATION
# # load huggingface model
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from datasets import Dataset, DatasetDict

# hugginface_model = AutoModelForCausalLM.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained(model_path)

# # tokenize data using the tokenizer above
# tokenized_data1 = tokenizer(data['qd1consis']['text'])
# tokenized_data2 = tokenizer(data['qd2incons']['text'])


# eval_dataset_d1 = DatasetDict(tokenized_data1)
# eval_dataset_d2 = DatasetDict(tokenized_data2)


# SCRAPS --- COUNT NUMBER OF TOKENS IN EACH VARIABLE
# from collections import Counter
# # check how many tokens are in each var
# _, vars1 = leave_unique_vars(data1)
# _, vars2 = leave_unique_vars(data2)
# vars = vars1.union(vars2)
# token_counts = {v: len(model.tokenizer.encode(v)) for v in vars}
# print(Counter(token_counts.values()))
# print(len(leave_unique_q_type(data1, model, q_type=q_type, filter_var_len=3)))
# print(len(leave_unique_q_type(data2, model, q_type=q_type, filter_var_len=3)))