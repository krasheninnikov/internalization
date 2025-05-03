from __future__ import annotations

import os
import pathlib

os.environ["OMP_NUM_THREADS"] = "6" # export OMP_NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = "6" # export OPENBLAS_NUM_THREADS
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = "6" # export VECLIB_MAXIMUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS

from collections import defaultdict
from copy import copy
from typing import Dict, List, Optional, Sequence, Tuple, Union, Set

import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch  # only used for `torch.concat` and memory cleanup
from einops import rearrange
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.utils import shuffle
from transformer_lens import (ActivationCache, FactoredMatrix,
                              HookedTransformer, HookedTransformerConfig)

from data_generation.define_experiment import get_questions_dataset
from utils.aggregation_utils import prettify_labels


def balanced_sample_by_stats(
    acts_1: np.ndarray,
    acts_2: np.ndarray,
    *,
    stats_1: Dict[str, np.ndarray],
    stats_2: Dict[str, np.ndarray],
    match_on: Optional[Sequence[str]] = None,
    n_bins: int | Dict[str, int] = 40,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Sub-sample two activation sets so that their *joint* distribution over the
    chosen statistics is identical up to histogram binning.
    """
    # ------------------------------------------------------------------ #
    # 0.  House-keeping & basic assertions                               #
    # ------------------------------------------------------------------ #
    if rng is None:
        rng = np.random.default_rng()

    for name, array in stats_1.items():
        assert len(array) == len(acts_1), f"stats_1[{name}] length mismatch"
        assert array.ndim == 1,          f"stats_1[{name}] must be 1-D"
    for name, array in stats_2.items():
        assert len(array) == len(acts_2), f"stats_2[{name}] length mismatch"
        assert array.ndim == 1,          f"stats_2[{name}] must be 1-D"

    if match_on is None:
        match_on = list(stats_1.keys())

    for stat_name in match_on:
        assert stat_name in stats_1, f"'{stat_name}' missing in stats_1"
        assert stat_name in stats_2, f"'{stat_name}' missing in stats_2"

    if isinstance(n_bins, int):
        bins_per_stat: Dict[str, int] = {name: n_bins for name in match_on}
    else:
        assert isinstance(n_bins, dict), "`n_bins` must be int or dict[str,int]"
        for name in match_on:
            assert name in n_bins, f"n_bins lacks entry for '{name}'"
        bins_per_stat = {name: int(n_bins[name]) for name in match_on}

    # ------------------------------------------------------------------ #
    # 1.  Digitise each statistic into integer bin labels                #
    # ------------------------------------------------------------------ #
    def digitise(stats_dict: Dict[str, np.ndarray]) -> np.ndarray:
        labels_per_stat = []
        for stat_name in match_on:
            values    = stats_dict[stat_name]
            num_bins  = bins_per_stat[stat_name]

            combined_min = min(stats_1[stat_name].min(), stats_2[stat_name].min())
            combined_max = max(stats_1[stat_name].max(), stats_2[stat_name].max())
            assert combined_max >= combined_min, "stat has NaNs or is ill-defined"

            edges = np.linspace(combined_min, combined_max, num_bins + 1,
                                dtype=np.float32)
            labels = np.digitize(values, edges[:-1], right=False).astype(np.int32)
            labels_per_stat.append(labels)

        return np.stack(labels_per_stat, axis=1)  # (N_examples, N_stats)

    bins_1 = digitise(stats_1)
    bins_2 = digitise(stats_2)

    assert bins_1.shape == (len(acts_1), len(match_on))
    assert bins_2.shape == (len(acts_2), len(match_on))

    # ------------------------------------------------------------------ #
    # 2.  Collapse k-D bin → single integer code                         #
    # ------------------------------------------------------------------ #
    radix_sizes = np.array([bins_per_stat[name] for name in match_on], dtype=np.int64)
    multipliers = np.concatenate([[1], np.cumprod(radix_sizes[:-1])])
    codes_1 = (bins_1 * multipliers).sum(axis=1)
    codes_2 = (bins_2 * multipliers).sum(axis=1)

    # ------------------------------------------------------------------ #
    # 3.  Sample the common support                                      #
    # ------------------------------------------------------------------ #
    kept_idx_1, kept_idx_2 = [], []

    shared_codes = np.intersect1d(codes_1, codes_2)
    assert shared_codes.size > 0, (
        "No overlap between the two classes for the requested statistics & bins."
    )

    for code in shared_codes:
        idx_1 = np.where(codes_1 == code)[0]
        idx_2 = np.where(codes_2 == code)[0]
        sample_size = min(idx_1.size, idx_2.size)
        kept_idx_1.append(rng.choice(idx_1, size=sample_size, replace=False))
        kept_idx_2.append(rng.choice(idx_2, size=sample_size, replace=False))

    kept_idx_1 = np.concatenate(kept_idx_1)
    kept_idx_2 = np.concatenate(kept_idx_2)
    rng.shuffle(kept_idx_1)
    rng.shuffle(kept_idx_2)

    assert kept_idx_1.size == kept_idx_2.size > 0, "Empty balanced sample"

    # ------------------------------------------------------------------ #
    # 4.  Slice activations & statistics                                 #
    # ------------------------------------------------------------------ #
    acts_1_bal = acts_1[kept_idx_1]
    acts_2_bal = acts_2[kept_idx_2]

    def slice_stats(orig: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, np.ndarray]:
        return {name: orig[name][idx] for name in orig}

    stats_1_bal = slice_stats(stats_1, kept_idx_1)
    stats_2_bal = slice_stats(stats_2, kept_idx_2)

    assert acts_1_bal.shape[0] == acts_2_bal.shape[0]
    for name in match_on:
        assert stats_1_bal[name].shape == stats_2_bal[name].shape

    return acts_1_bal, acts_2_bal, stats_1_bal, stats_2_bal


def calculate_logit_stats(logits: torch.Tensor) -> Dict[str, np.ndarray]:
    """
    Calculates various statistics over the vocabulary dimension of a logit batch.

    Performs most calculations (mean, std, min/max, percentiles, norms, entropy)
    on the GPU using PyTorch, then transfers results to CPU NumPy arrays.
    Calculates skewness and kurtosis on the CPU using SciPy after transferring
    the raw logits.

    Args:
        logits: A PyTorch tensor of shape (B, T, V), potentially on GPU.

    Returns:
        A dictionary where keys are stat names (str) and values are NumPy arrays
        of shape (B, T) containing the calculated statistics (as np.float32).
    """
    PERCENTILES_TO_CALCULATE_TORCH = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                                0.95, 0.96, 0.97, 0.98, 0.99])
    
    stats_torch = {} # Intermediate dict for torch tensors
    stats_np = {}    # Final dict for numpy arrays
    device = logits.device
    dtype_torch = torch.float32
    dtype_np = np.float32

    # Ensure logits are float for calculations
    logits_float = logits.float()

    # --- Calculate stats on original device (GPU potentially) using PyTorch ---
    stats_torch["mean"] = logits_float.mean(dim=-1).to(dtype_torch)
    stats_torch["std"] = logits_float.std(dim=-1, unbiased=True).to(dtype_torch) # Unbiased std
    stats_torch["max"] = logits_float.max(dim=-1).values.to(dtype_torch)
    stats_torch["min"] = logits_float.min(dim=-1).values.to(dtype_torch)

    # Percentiles using torch.quantile
    quantiles_tensor = PERCENTILES_TO_CALCULATE_TORCH.to(device)
    percentile_values = torch.quantile(logits_float, q=quantiles_tensor, dim=-1) # Shape: (num_quantiles, B, T)
    for i, q in enumerate(PERCENTILES_TO_CALCULATE_TORCH):
        percentile_key = f"percentile_{(q * 100):.0f}"
        stats_torch[percentile_key] = percentile_values[i].to(dtype_torch)

    # Norms using torch.linalg.norm
    stats_torch["norm_l1"] = torch.linalg.norm(logits_float, ord=1, dim=-1).to(dtype_torch)
    stats_torch["norm_l2"] = torch.linalg.norm(logits_float, ord=2, dim=-1).to(dtype_torch)

    # Entropy using torch.log_softmax
    log_probs = torch.log_softmax(logits_float, dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    stats_torch["entropy"] = entropy.to(dtype_torch)

    # --- Move PyTorch results to CPU NumPy arrays ---
    for name, tensor in stats_torch.items():
        stats_np[name] = tensor.cpu().numpy().astype(dtype_np) # Ensure float32 just in case

    # --- Calculate Skewness & Kurtosis on CPU using SciPy ---
    # Move raw logits to CPU NumPy *once* for these calculations
    logits_np = logits_float.cpu().numpy()

    # Calculate unbiased skewness
    stats_np["skewness"] = scipy.stats.skew(logits_np, axis=-1, bias=False).astype(dtype_np)
    # Calculate unbiased Fisher's kurtosis (excess kurtosis)
    stats_np["kurtosis"] = scipy.stats.kurtosis(logits_np, axis=-1, fisher=True, bias=False).astype(dtype_np)

    return stats_np


def get_activations_and_logit_stats(
    model: HookedTransformer,
    data: Sequence[str],
    batch_size: int = 128,
    hook_substr: str = "hook_resid_post",
    device: str | torch.device | None = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Collect activations **and** logit statistics for every example. 

    Returns
    -------
    activations : Dict[str, np.ndarray]
        Each array has shape (N_examples, T, d_model), dtype float32. Contains
        residual stream activations specified by `hook_substr`.
    logit_stats_dict : Dict[str, np.ndarray]
        Each array has shape (N_examples, T), dtype float32. Contains statistics
        computed over the vocabulary dimension of the logits.
        Keys include mean, std, min/max, percentiles, norms, entropy, skewness, kurtosis.

    Notes
    -----
    * No assumptions on `T`; concatenation happens along the batch axis.
    * Activations are collected from the specified device, then moved to CPU NumPy arrays.
    """
    acts_batches: Dict[str, List[torch.Tensor]] = defaultdict(list)
    stats_batches_np: Dict[str, List[np.ndarray]] = defaultdict(list)

    model.eval() # Ensure model is in eval mode
    with torch.no_grad(): # No need to track gradients
        for start in range(0, len(data), batch_size):
            batch = data[start : start + batch_size]

            # Run model and get logits and cache
            logits, cache = model.run_with_cache(batch, device=device)
            # logits shape: (B, T, V), on specified device or model's device

            # Calculate logit stats
            batch_logit_stats_np : Dict[str, np.ndarray] = calculate_logit_stats(logits)
            for stat_name, array_np in batch_logit_stats_np.items():
                stats_batches_np[stat_name].append(array_np)

            # Store activations (move tensor to CPU)
            for layer_name, tensor in cache.items():
                if hook_substr in layer_name:
                    acts_batches[layer_name].append(tensor.cpu())

            # Housekeeping for the batch
            del cache, logits, batch_logit_stats_np # Delete GPU tensor and stats dict
            torch.cuda.empty_cache()
            
    # Concatenate batches for activations (Torch tensors -> NumPy)
    activations = {
        name: torch.concat(t_list, dim=0).numpy()
        for name, t_list in acts_batches.items()
    }

    # Concatenate batches for logit statistics (already NumPy arrays)
    logit_stats_dict = {
        stat_name: np.concatenate(arr_list, axis=0)
        for stat_name, arr_list in stats_batches_np.items()
    }

    return activations, logit_stats_dict


def train_linear_probe(x1, x2, num_cross_val=5):
    # Check if x1 and x2 are the same
    if np.allclose(x1, x2):
        return {
            'cv_scores': [len(x1)/(len(x1)+len(x2))] * num_cross_val,
            'trained_classifier': None
        }

    # Concatenate and shuffle
    x = rearrange([x1, x2], 'x n d -> (x n) d')
    y = rearrange([np.zeros(len(x1)), np.ones(len(x2))], 'x n -> (x n)')  # zero for x1, one for x2
    x, y = shuffle(x, y, random_state=0)

    # Classifier definition
    clf = LogisticRegression(random_state=0, max_iter=1000, penalty='l2', C=0.01)

    # Cross-validation
    scores = cross_validate(clf, x, y, cv=num_cross_val, scoring='accuracy', n_jobs=num_cross_val, return_train_score=True)

    # Retrain on full dataset
    clf.fit(x, y)

    # Return CV scores and trained classifier
    return {
        'cv_scores': scores['test_score'],
        'trained_classifier': clf
    }


def leave_unique_q_type(data: List[str], model: HookedTransformer, q_type:str='born', filter_var_len=3) -> List[str]:
    """
    q_type is a string that must be contained in the question
    model is needed for tokenization
    filter_var_len is the length (in tokens) of the variables we keep
    """
    out = []
    vars_set = set()
    for d in data:
        assert d.count('<|') == 1, f'{d} is a definition'  # ensure "<|" occurs only once in the string     
        var = d.split('<|')[1].split('|>')[0]  # variable is surrounded by <| and |>
        
        if q_type in d and len(model.tokenizer.tokenize(var)) == filter_var_len:
            assert var not in vars_set, f'{var} is already in the set'
            vars_set.add(var)
            # print(var, len(model.tokenizer.encode(var)), model.tokenizer.tokenize(var))
            out.append(d)
    assert len(out) > 0, f'no data for q_type {q_type} and filter_var_len {filter_var_len}'
    # ensure all questions have the same tokenized length
    assert all([len(model.tokenizer.tokenize(d)) == len(model.tokenizer.tokenize(out[0])) for d in out])
    return out
    
    
def run_q_type(model, data1, data2, q_type='born', filter_var_len=3, device='cuda'):
    """
    Train linear probes to distinguish between data1 and data2 based on activations.
    Returns a grid of average cross-validation scores and the two datasets.
    
    score_grid shape: (num_tokens, num_layers)
    data1 and data2 are lists of strings
    """
    if type(model) == str:
        model = HookedTransformer.from_pretrained(model, device=device)
    
    # We need to ensure that the same variable cannot be in the train and in the test set
    # and that all questions have the same tokenized length. Simplest solution: take only one type of question
    data1 = leave_unique_q_type(data1, model, q_type, filter_var_len)
    data2 = leave_unique_q_type(data2, model, q_type, filter_var_len)
    assert len(model.tokenizer.tokenize(data1[0])) == len(model.tokenizer.tokenize(data2[0]))
    
    # make sure data1 and data2 have the same length so that baseline linear probe accuracy is 0.5
    minlen = min(len(data1), len(data2))
    data1, data2 = data1[:minlen], data2[:minlen]
    print(f'data lengths: {len(data1)}, {len(data2)}')
    
    acts_data1, entropies_data1 = get_activations_and_entropy(model, data1)
    acts_data2, entropies_data2 = get_activations_and_entropy(model, data2)
    
    n_examples, n_tokens, d_model = acts_data1[list(acts_data1.keys())[0]].shape
    # print(f'n_examples: {n_examples} \t n_tokens: {n_tokens} \t d_model: {d_model}')
    
    score_grid, clf_grid = train_probes_per_layer_and_token(acts_data1, acts_data2)
    
    return score_grid, clf_grid, data1, data2, acts_data1, acts_data2


def train_probes_per_layer_and_token(acts1, acts2):
     # TODO check that the shapes are the same and that the keys are the same    
    n_examples, n_tokens, d_model = acts1[list(acts1.keys())[0]].shape
    layer_names = list(acts1.keys())
    score_grid = np.zeros((n_tokens, len(layer_names)))
    clf_grid = [[None for _ in layer_names] for _ in range(n_tokens)]  # classifier grid

    for layer in layer_names:
        for token_idx in range(n_tokens):
            # train linear probe on activations for token i
            result = train_linear_probe(
                acts1[layer][:, token_idx, :], 
                acts2[layer][:, token_idx, :]
            )
            
            score_grid[token_idx, layer_names.index(layer)] = np.mean(result['cv_scores'])
            clf_grid[token_idx][layer_names.index(layer)] = result['trained_classifier']  # store classifier

    score_grid = score_grid[1:, :]  # remove BOS token that transformerlens adds automatically
    clf_grid = clf_grid[1:]         # also remove BOS classifiers for consistency
    return score_grid, clf_grid


def plot_score_grid(scores, tokens: List[str], title=None, vmin=0.49, vmax=1.01, cmap='Blues', plot_name='linear_probe'):
    """
    Plot a grid of scores, with tokens on the x axis and layers on the y axis.
    scores: np array with shape (num_tokens, num_layers)
    """
    # larger font size and times new roman font
    plt.rc('font', size=14)#, family='Times New Roman')
    plt.rc('text', usetex=False)
    
    fig, ax = plt.subplots(figsize=(6, 2.2))

    # brainstorming cmaps; some to try: with blues and reds: 'PuOr', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm'   
    sns.heatmap(scores.T, cmap=cmap, vmin=vmin, vmax=vmax, cbar_kws={'ticks': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]})
    
    plt.xticks(np.arange(len(tokens)), tokens, rotation=60) # set x ticks to the actual str tokens
    plt.gca().set_xticks(np.arange(len(tokens))+0.5, minor=True) # position x ticks in the middle of the cell without extra ticks
    plt.gca().tick_params(which='major', length=0) # remove major ticks and keep only minor ticks
    plt.gca().invert_yaxis() # make y axis go from bottom to top

    plt.xlabel('Token',  
               labelpad=-10
               )
    plt.ylabel('Layer')
    if title is not None:
        plt.title(title, y=1.08, x=0.53, fontdict={'fontsize': 16})
    
    # plt.yticks(np.arange(len(scores.T))+0.5, range(1, len(scores.T)+1)) # add 1 to every y tick without changing its position
    plt.yticks(np.arange(len(scores.T)), range(1, len(scores.T)+1)) # add 1 to every y tick without changing its position

    # thin grid lines
    plt.grid(which='major', color='gray', linestyle='-', linewidth=0.3)

    # leave only every 4th y tick       
    for i, label in enumerate(plt.gca().yaxis.get_ticklabels()):
        if (i+1) % 8 != 0:
            label.set_visible(False)
    
    for label in plt.gca().yaxis.get_ticklabels():
        # label.set_verticalalignment('bottom')
        label.set_verticalalignment('center')
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
    
    # save the plot to a file
    plt_path = 'plots/linear_probes'
    plt_format = 'pdf'
    pathlib.Path(plt_path).mkdir(parents=True, exist_ok=True)
    n = 1
    while pathlib.Path(f'{plt_path}/{plot_name}_{n}.{plt_format}').exists():  # Check if file already exists and increment n if so
        n += 1
    fig.savefig(f'{plt_path}/{plot_name}_{n}.{plt_format}', bbox_inches='tight')
    plt.show()
    

def main():
    """Example usage of run_q_type and plot_score_grid"""
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed = 600
    seed_stage2 = 0
    np.random.seed(seed)
    
    # TODO just load data from a config file using this fn: "from data_generation.load_data_from_config import generate_data_from_experiment_folder"
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

    # NOTE: by default, transformerlens does not support custom models. As a workaround, we modify transformerlens code, which is cursed (TODO: submit a PR to transformerlens).
    # In particular, we modify transformer_lens.loading_from_pretrained.get_official_model_name to return model_name if official_model_name is None, instead of raising an error.
    try:
        model = HookedTransformer.from_pretrained(model_path, device=device)
    except Exception as e:
        print(e)
        print('Loading the model failed. Make sure to modify transformer_lens code as described above.')
        return
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


def entropy_of_generated_answers(model, data, data_subsets=None, answer_tokens_only=True, temperature=0):
    data_subsets = ['ent_assoc_name_qd1consis', 
                'ent_assoc_name_qd2incons', 
                'ent_assoc_name_q', 
                'ent_assoc_name_d1consis',
                'ent_assoc_name_d2consis',]
    # data_subsets = ['ent_assoc_who_qd1consis', 
    #                 'ent_assoc_who_qd2incons', 
    #                 'ent_assoc_who_q', 
    #                 'ent_assoc_who_d1consis',
    #                 'ent_assoc_who_d2consis',]
    
    # data_subsets = ['qd1consis',
    #                 'qd2incons',
    #                 'q',
    #                 'd1consis',
    #                 'd2consis',
    #                 # 'd3consis',
    #                 'no_qd_baseline',
    #                 'q_no_replacement_baseline']

    # generate answers using the model
    qa_generated_ans_dict = {}
    for subset in data_subsets:
        qa_generated_ans_dict[subset] = [model.generate(data[subset]['question'][i], max_new_tokens=50, temperature=temperature)
                                        for i in range(len(data[subset]['question']))]
        
    # compute per-token losses of generated samples
    qa_generated_per_token_losses_dict = {}
    for k in data_subsets:
        qa_generated_per_token_losses_dict[k] = [model(qa_generated_ans_dict[k][i], return_type="loss", loss_per_token=True).cpu().numpy()
                                                 for i in range(len(data[k]['question']))]
        
    qa_generated_ans_losses_dict = {}
    for k in data_subsets:
        if answer_tokens_only:
            # the answer is everything after the last ":"
            answers = [x.split(':')[-1].strip() for x in qa_generated_ans_dict[k]]  # .replace('\n<|endoftext|>', '')
            print(answers)
            # how many tokens are in the answers?
            answer_token_lenghts = [len(model.tokenizer.tokenize(ans)) for ans in answers]
            # get the losses for the answer tokens. index 0 is there because the batch size is 1; -1 is there because the last token is EOS
            answer_per_token_losses = [qa_generated_per_token_losses_dict[k][i][0, -answer_token_lenghts[i]:-1] for i in range(len(answers))]
        else:  # loss on the entire sequence
            answer_per_token_losses = [qa_generated_per_token_losses_dict[k][i][0, :] for i in range(len(data[k]['question']))]
                                                                                
        # average over tokens
        answer_losses = [np.mean(l) for l in answer_per_token_losses]
        qa_generated_ans_losses_dict[k] = answer_losses
    return qa_generated_ans_losses_dict


def plot_entropy_of_generated_answers(qa_generated_ans_losses_dict: Dict[str, List[float]]):
    """Input is a dictionary with keys being data subsets and values being lists of per-answer losses"""
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    
    def data_subset_to_color(subset: str):
        palette = sns.color_palette()
        color2order = {'blue': 0, 'orange': 1, 'green': 2, 'red': 3, 'purple': 4, 'brown': 5, 'pink': 6, 'gray': 7, 'olive': 8, 'cyan': 9}  
        name2color = {'d1consis': 'blue', 'q': 'brown',  'qd2incons': 'pink',  'd2consis': 'red', 'qd1consis': 'purple',
                        'no_qd_baseline': 'orange', 'q_no_replacement_baseline': 'green', 'qd1incons': 'cyan', 'qd2consis': 'olive', 'd3consis': 'gray'}
        replace_dict = {'ent_assoc_meaning_': '', 'ent_assoc_who_': '', 'ent_assoc_name_': '', 'ent_assoc_standFor_': '',}
        for k, v in replace_dict.items():
            subset = subset.replace(k, v)
        return palette[color2order[name2color[subset]]]

    data_subsets = list(qa_generated_ans_losses_dict.keys())
    # plot loss histograms using sns, all in one figure
    fig, axs = plt.subplots(1, 1, figsize=(4, 4))
    for k in data_subsets:
        color = data_subset_to_color(k)
        sns.distplot(qa_generated_ans_losses_dict[k], ax=axs, label=k, bins=10, color=color)
    axs.set_title(f'Loss distribution for model-generated answers \n (TVE, ``name of xyz" test set)')
    axs.set_title(f'Loss distribution for model-generated answers \n (TVE, ``who is xyz" test set)')
    # axs.set_title(f'Loss distribution for model-generated answers \n (TVE, test questions similar to training ones)')
    axs.set_xlabel('Loss')
    axs.set_xlim(0, 2)
    axs.legend()
    handles, labels = axs.get_legend_handles_labels()
    new_labels = prettify_labels(data_subsets)
    sorted_pairs = sorted(zip(handles, new_labels), key=lambda zipped_pair: int([c for c in zipped_pair[1] if c.isdigit()][0]))
    handles, new_labels = zip(*sorted_pairs)
    axs.legend(handles, new_labels, loc='upper right')
        

def leave_unique_vars(data_in: List[str]) -> Tuple[List[str], Set[str]]:
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
    # add braces back to the variables
    unique_vars = set([f'<|{v}|>' for v in unique_vars])
    return data_out, unique_vars


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