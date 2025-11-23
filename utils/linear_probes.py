from __future__ import annotations

import os, gc
import pathlib
import warnings
os.environ["OMP_NUM_THREADS"] = "6" # export OMP_NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = "6" # export OPENBLAS_NUM_THREADS
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = "6" # export VECLIB_MAXIMUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS

import numbers
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
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, StratifiedKFold
from sklearn.utils import shuffle
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from transformer_lens import utils as tl_utils

from data_generation.define_experiment import get_questions_dataset
from utils.aggregation_utils import prettify_labels


def _compute_loss_stats_from_log_probs(
    log_probs: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-position next-token loss (NLL) and running-average variants
    from log-probabilities and input token IDs.

    Args:
        log_probs: Tensor of shape (B, T, V) with log-softmaxed logits.
        input_ids: Tensor of shape (B, T) with token IDs.
        attention_mask: Optional binary tensor of shape (B, T).

    Returns:
        Dict with keys:
            - 'next_token_loss': next-token NLL aligned to position t (last position NaN)
            - 'cumulative_avg_loss': running average NLL up to and including t
            - 'cumulative_avg_loss_prev': running average NLL up to previous token
    """
    device = log_probs.device
    B, T, V = log_probs.shape
    assert input_ids.shape[:2] == (B, T), "input_ids must have shape (B, T) matching logits"

    # Labels for the next token at each position t (valid for t in [0..T-2])
    labels_next = input_ids[:, 1:]

    # Gather logprob of the chosen next token at each position
    # Shape: (B, T-1)
    next_logprob = torch.gather(
        log_probs[:, :-1, :],
        dim=-1,
        index=labels_next.unsqueeze(-1)
    ).squeeze(-1)
    next_nll = -next_logprob

    # Validity mask for next-token loss
    if attention_mask is not None:
        if attention_mask.device != device:
            attention_mask = attention_mask.to(device)
        valid_next = (attention_mask[:, :-1] > 0) & (attention_mask[:, 1:] > 0)
    else:
        valid_next = torch.ones_like(next_nll, dtype=torch.bool, device=device)

    # Per-position next-token loss (pad last position)
    next_nll = next_nll.masked_fill(~valid_next, float('nan'))
    loss_next = torch.nn.functional.pad(next_nll, (0, 1), value=float('nan'))  # (B, T)

    # Align per-token current loss: NLL(x_t | x_<t)
    token_nll_cur = torch.nn.functional.pad(next_nll, (1, 0), value=float('nan'))  # (B, T)

    # For running averages: cumulative sum and count of valid entries
    mask_cur = (attention_mask > 0).to(token_nll_cur.dtype) if attention_mask is not None else torch.ones_like(token_nll_cur)

    nll_filled = torch.where(torch.isnan(token_nll_cur), torch.zeros_like(token_nll_cur), token_nll_cur)
    nll_filled = nll_filled * mask_cur
    cumsum = torch.cumsum(nll_filled, dim=-1)

    valid_flags = ((mask_cur > 0) & ~torch.isnan(token_nll_cur)).to(token_nll_cur.dtype)
    count = torch.cumsum(valid_flags, dim=-1)

    eps = torch.finfo(token_nll_cur.dtype).eps
    avg = cumsum / torch.clamp_min(count, eps)
    avg = torch.where(count > 0, avg, torch.tensor(float('nan'), device=device, dtype=avg.dtype))

    if attention_mask is not None:
        avg = torch.where(attention_mask > 0, avg, torch.tensor(float('nan'), device=device, dtype=avg.dtype))

    # Previous-token running average via shifted sums and counts
    cumsum_prev = torch.nn.functional.pad(cumsum[:, :-1], (1, 0), value=0)
    count_prev = torch.nn.functional.pad(count[:, :-1], (1, 0), value=0)
    avg_prev = cumsum_prev / torch.clamp_min(count_prev, eps)
    avg_prev = torch.where(count_prev > 0, avg_prev, torch.tensor(float('nan'), device=device, dtype=avg_prev.dtype))
    if attention_mask is not None:
        avg_prev = torch.where(attention_mask > 0, avg_prev, torch.tensor(float('nan'), device=device, dtype=avg_prev.dtype))

    return {
        "next_token_loss": loss_next,
        "cumulative_avg_loss": avg,
        "cumulative_avg_loss_prev": avg_prev,
    }


def load_model_to_transformerlens(
    model_path,
    base_model_name,
    device="cuda",
    torch_dtype=None,
    hf_device_map="cpu",
    convert_on_cpu=True,
):
    """
    Load an HF model (optionally with a PEFT adapter), convert to TransformerLens,
    and finally move to `device`. By default, everything is loaded and converted
    on CPU to avoid holding two GPU copies at once.
    """
    if torch_dtype is None:
        torch_dtype = torch.bfloat16

    tl_build_device = "cpu" if convert_on_cpu else device

    # Handle PEFT models
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        from peft import PeftModel, PeftConfig

        # Load base model (defaults to CPU) to keep merge off GPU
        peft_cfg = PeftConfig.from_pretrained(model_path)
        base = AutoModelForCausalLM.from_pretrained(
            peft_cfg.base_model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=hf_device_map,
            low_cpu_mem_usage=True,
        )

        # Load adapter and merge
        hf_model = PeftModel.from_pretrained(
            base,
            model_path,
            torch_dtype=torch_dtype,
            device_map=hf_device_map,
        ).merge_and_unload()

        del base
        gc.collect()

    else:
        # Load regular model
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=hf_device_map,
            low_cpu_mem_usage=True,
        )

    # Convert to TransformerLens (stays on CPU if convert_on_cpu=True)
    tl_model = HookedTransformer.from_pretrained(
        model_name=base_model_name,
        hf_model=hf_model,
        tokenizer=AutoTokenizer.from_pretrained(model_path),
        device=tl_build_device,
        move_to_device=False,
        dtype=torch_dtype,
    )

    # Clean up and move to final device
    del hf_model
    gc.collect()

    return tl_model.to(device)


def run_balancing_probe_analysis(
    layer_name: str,
    token_idx: int,
    acts_data1: dict,
    logit_stats_data1: dict,
    acts_data2: dict,
    logit_stats_data2: dict,
    n_bins_values: list,
    matched_stats: list,
    rng: np.random.Generator,
    prev_token_indices: List[int] = None,  # TODO do we want to assert that all of these are < token_idx? Or warn?
    # --- Parameters ---
    binning_strategy: str = "quantile",
    n_random_probe_runs: int = 5, # Acts as outer CV folds
    n_cross_val: int = 1,         # Value 1 disables inner CV within train_linear_probe
    test_size: float = 0.2,
):
    """
    Args:
        layer_name: Name of the layer.
        token_idx: Index of the token position.
        acts_data1: Activation data for dataset 1.
        logit_stats_data1: Logit statistics for dataset 1.
        acts_data2: Activation data for dataset 2.
        logit_stats_data2: Logit statistics for dataset 2.
        n_bins_values: List of bin counts for balancing.
        matched_stats: List of statistic names to match on.
        rng: NumPy random number generator instance.
        binning_strategy: Strategy for binning ('quantile', 'uniform').
        n_random_probe_runs: Number of outer train/test splits (folds).
        n_cross_val: Inner CV folds for train_linear_probe (use == 1).
        test_size: Fraction of the original data to use as the test set per fold.

    Returns:
        A dictionary containing lists of scores averaged across outer folds,
        average training sample sizes, and the list of n_bins processed.
    """

    assert len(acts_data1[layer_name]) == len(acts_data2[layer_name]), "The two datasets must have the same number of examples"
    scores_balanced_per_bin = defaultdict(list)
    scores_unbalanced_per_bin = defaultdict(list)
    scores_full_per_bin = defaultdict(list)
    train_samples_per_bin = defaultdict(list)

    acts1_full = acts_data1[layer_name][:, token_idx, :]
    acts2_full = acts_data2[layer_name][:, token_idx, :]
    
    if np.allclose(acts1_full, acts2_full):
        return {
            'scores_balanced': [0.5],
            'scores_unbalanced': [0.5],
            'scores_full_set': [0.5],
            'sample_sizes': [len(acts1_full)],
            'n_bins_successful': [1]
        }

    y1_full = np.zeros(len(acts1_full))
    y2_full = np.ones(len(acts2_full))
    X_full = np.concatenate((acts1_full, acts2_full), axis=0)
    y_full = np.concatenate((y1_full, y2_full), axis=0)
    indices_full = np.arange(len(X_full))

    # print(f"Starting analysis with {n_random_probe_runs} outer folds...")
    for run_idx in range(n_random_probe_runs):
        current_run_seed = rng.integers(10000)
        # print(f"--- Run {run_idx+1}/{n_random_probe_runs} (seed: {current_run_seed}) ---")

        X_train, X_test, y_train, y_test, indices_train, _ = train_test_split(
            X_full, y_full, indices_full, test_size=test_size, stratify=y_full, random_state=current_run_seed
        )

        acts1_train = X_train[y_train == 0]
        acts2_train = X_train[y_train == 1]

        # --- Corrected Stat Slicing ---
        original_indices_class0_in_train = indices_train[y_train == 0]
        original_indices_class1_in_train = indices_train[y_train == 1]
        # Adjust indices for class 1 to be relative to acts2_full / logit_stats_data2
        indices_relative_to_acts2 = original_indices_class1_in_train - len(acts1_full)

        stats1_train = {name: logit_stats_data1[name][:, token_idx][original_indices_class0_in_train] for name in matched_stats}
        stats2_train = {name: logit_stats_data2[name][:, token_idx][indices_relative_to_acts2] for name in matched_stats}
        # --- End Correction ---
        min_train_size = min(len(acts1_train), len(acts2_train))
        results_full = train_linear_probe(acts1_train[:min_train_size], acts2_train[:min_train_size], num_cross_val=n_cross_val)
        clf_full = results_full['trained_classifier']
        assert clf_full is not None, f"Full probe training failed in run {run_idx+1}"
        score_full_run = clf_full.score(X_test, y_test)
        # print(f"  Run {run_idx+1}: Full probe test score = {score_full_run:.4f}")

        for n_bins in n_bins_values:
            # print(f"    Processing n_bins = {n_bins}...")
            try:
                # TODO consider refactoring so that these two cases are handled by the same function
                if prev_token_indices is None:
                    acts1_bal_train, acts2_bal_train, _, _, _, _ = balanced_sample_by_stats(
                        acts_1=acts1_train, acts_2=acts2_train,
                        stats_1=stats1_train, stats_2=stats2_train,
                        match_on=matched_stats,
                        n_bins=n_bins,
                        rng=rng,
                        binning=binning_strategy
                    )
                else:
                    # print(f"Balancing using prev token stats for n_bins={n_bins}")
                    acts1_bal_train, acts2_bal_train, _, _ = balance_using_prev_token_stats(
                        all_acts_data1=acts_data1, all_acts_data2=acts_data2, 
                        curr_layer_name=layer_name, curr_token_idx=token_idx,
                        all_logit_stats_data1=logit_stats_data1, all_logit_stats_data2=logit_stats_data2, 
                        prev_token_indices=prev_token_indices, n_bins_for_prev_stats=n_bins, 
                        matched_stats=matched_stats, rng=rng, binning_strategy=binning_strategy)
                M = acts1_bal_train.shape[0]
                assert M > 0, f"Balancing yielded M=0"
                # print(f"      Balanced to M = {M}")

                results_bal = train_linear_probe(acts1_bal_train, acts2_bal_train, num_cross_val=n_cross_val, shuffle_seed=current_run_seed)
                clf_bal = results_bal['trained_classifier']
                assert clf_bal is not None, f"Balanced probe training failed"
                score_bal_run = clf_bal.score(X_test, y_test)
                # print(f"        Balanced probe test score = {score_bal_run:.4f}")


                idx1 = rng.choice(len(acts1_train), size=M, replace=False)
                idx2 = rng.choice(len(acts2_train), size=M, replace=False)
                results_unbal = train_linear_probe(acts1_train[idx1], acts2_train[idx2], num_cross_val=n_cross_val, shuffle_seed=current_run_seed + 1)
                clf_unbal = results_unbal['trained_classifier']
                assert clf_unbal is not None, f"Unbalanced probe training failed"
                score_unbal_run = clf_unbal.score(X_test, y_test)
                # print(f"        Unbalanced probe test score = {score_unbal_run:.4f}")


                scores_balanced_per_bin[n_bins].append(score_bal_run)
                scores_unbalanced_per_bin[n_bins].append(score_unbal_run)
                scores_full_per_bin[n_bins].append(score_full_run)
                train_samples_per_bin[n_bins].append(2 * M)

            except AssertionError as e:
                print(f"      Run {run_idx+1}, n_bins={n_bins}: Failed: {e}. Skipping bin for this run.")
                continue

    print("\n--- Averaging Results Across Runs ---")
    final_scores_balanced = []
    final_scores_unbalanced = []
    final_scores_full = []
    final_avg_sample_sizes = []
    processed_n_bins = sorted(scores_balanced_per_bin.keys())

    for n_bins in processed_n_bins:
        # Calculate means for bins that had successful runs
        bal_mean = np.mean(scores_balanced_per_bin[n_bins])
        unbal_mean = np.mean(scores_unbalanced_per_bin[n_bins])
        full_mean = np.mean(scores_full_per_bin[n_bins]) # Should have scores from all runs if bin processed once
        sample_mean = np.mean(train_samples_per_bin[n_bins])

        final_scores_balanced.append(bal_mean)
        final_scores_unbalanced.append(unbal_mean)
        final_scores_full.append(full_mean)
        final_avg_sample_sizes.append(sample_mean)

        print(f"  n_bins={n_bins}: Balanced={bal_mean:.4f}, Unbalanced={unbal_mean:.4f}, Full={full_mean:.4f}, Avg_M*2={sample_mean:.1f} (from {len(scores_balanced_per_bin[n_bins])} runs)")

    return {
        'scores_balanced': final_scores_balanced,
        'scores_unbalanced': final_scores_unbalanced,
        'scores_full_set': final_scores_full,
        'sample_sizes': final_avg_sample_sizes,
        'n_bins_successful': processed_n_bins
    }


def balance_using_prev_token_stats(
    all_acts_data1: Dict[str, np.ndarray],
    all_acts_data2: Dict[str, np.ndarray],
    curr_layer_name: str,
    curr_token_idx: int,
    all_logit_stats_data1: Dict[str, np.ndarray],
    all_logit_stats_data2: Dict[str, np.ndarray],
    prev_token_indices: List[int],
    n_bins_for_prev_stats: Union[int, Dict[str, int]],
    matched_stats: Sequence[str],
    rng: np.random.Generator,
    binning_strategy: str = "quantile",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Balances current token activations based on the statistics of previous tokens.

    This function extracts activations for a specific layer and token position,
    then subsamples them. The subsampling ensures that for each specified 
    previous token in prev_token_indices, the distributions of matched_stats 
    are similar between the two resulting subsamples.

    Args:
        all_acts_data1: Dict mapping layer_name to activation arrays (N1, T, D_model) for dataset 1.
        all_acts_data2: Dict mapping layer_name to activation arrays (N2, T, D_model) for dataset 2.
        curr_layer_name: Name of the layer for current token activations.
        curr_token_idx: Index of the current token.
        all_logit_stats_data1: Dict mapping stat_name to np.ndarray of shape (N1, T_max),
                               containing statistics for all tokens in dataset 1.
        all_logit_stats_data2: Dict mapping stat_name to np.ndarray of shape (N2, T_max),
                               for dataset 2.
        prev_token_indices: List of absolute token indices whose statistics will be used
                            for balancing.
        n_bins_for_prev_stats: Number of bins for balancing. Passed to balanced_sample_by_stats.
        matched_stats: List of statistic names to balance on (e.g., ['mean', 'std']).
        rng: NumPy random number generator.
        binning_strategy: Binning strategy ('quantile' or 'uniform').

    Returns:
        A tuple (acts_curr_tok1_balanced, acts_curr_tok2_balanced, 
                 final_keep_idxs1, final_keep_idxs2).
        - acts_curr_tok1_balanced: Subsampled activations from dataset 1.
        - acts_curr_tok2_balanced: Subsampled activations from dataset 2.
        - final_keep_idxs1: Indices used to subsample acts_curr_tok1.
        - final_keep_idxs2: Indices used to subsample acts_curr_tok2.
    """
    # --- Input Assertions & Initial Data Extraction ---
    assert isinstance(all_acts_data1, dict), "all_acts_data1 must be a dictionary."
    assert isinstance(all_acts_data2, dict), "all_acts_data2 must be a dictionary."
    assert curr_layer_name in all_acts_data1, f"curr_layer_name '{curr_layer_name}' not found in all_acts_data1."
    assert curr_layer_name in all_acts_data2, f"curr_layer_name '{curr_layer_name}' not found in all_acts_data2."

    acts_full_layer1 = all_acts_data1[curr_layer_name]
    acts_full_layer2 = all_acts_data2[curr_layer_name]

    assert isinstance(acts_full_layer1, np.ndarray) and acts_full_layer1.ndim == 3, \
        f"Activations for layer '{curr_layer_name}' in dataset 1 must be a 3D numpy array (N, T, D_model)."
    assert isinstance(acts_full_layer2, np.ndarray) and acts_full_layer2.ndim == 3, \
        f"Activations for layer '{curr_layer_name}' in dataset 2 must be a 3D numpy array (N, T, D_model)."

    num_sequences1, seq_len1, _ = acts_full_layer1.shape
    num_sequences2, seq_len2, _ = acts_full_layer2.shape

    assert -seq_len1 <= curr_token_idx < seq_len1, \
        f"curr_token_idx {curr_token_idx} is out of bounds for layer '{curr_layer_name}' in dataset 1 (seq_len: {seq_len1})."
    assert -seq_len2 <= curr_token_idx < seq_len2, \
        f"curr_token_idx {curr_token_idx} is out of bounds for layer '{curr_layer_name}' in dataset 2 (seq_len: {seq_len2})."

    acts_curr_tok1 = acts_full_layer1[:, curr_token_idx, :]
    acts_curr_tok2 = acts_full_layer2[:, curr_token_idx, :]
    
    assert isinstance(all_logit_stats_data1, dict), "all_logit_stats_data1 must be a dictionary."
    assert isinstance(all_logit_stats_data2, dict), "all_logit_stats_data2 must be a dictionary."

    assert isinstance(prev_token_indices, list) and len(prev_token_indices) > 0, \
        "prev_token_indices must be a non-empty list."
    assert all(isinstance(idx, int) for idx in prev_token_indices), \
        "All elements in prev_token_indices must be integers."

    assert isinstance(matched_stats, Sequence) and len(matched_stats) > 0, \
        "matched_stats must be a non-empty sequence."
    assert all(isinstance(stat, str) for stat in matched_stats), \
        "All elements in matched_stats must be strings."

    # Check consistency of sample counts between activations and logit stats
    # And validate prev_token_indices against logit stat sequence lengths
    for stat_name in matched_stats:
        assert stat_name in all_logit_stats_data1, f"Stat '{stat_name}' not found in all_logit_stats_data1."
        assert stat_name in all_logit_stats_data2, f"Stat '{stat_name}' not found in all_logit_stats_data2."
        
        stat_array1 = all_logit_stats_data1[stat_name]
        stat_array2 = all_logit_stats_data2[stat_name]

        assert isinstance(stat_array1, np.ndarray) and stat_array1.ndim == 2, \
            f"Stat '{stat_name}' in all_logit_stats_data1 must be a 2D array (N, T_stat)."
        assert stat_array1.shape[0] == num_sequences1, \
            f"Stat '{stat_name}' in all_logit_stats_data1 has {stat_array1.shape[0]} samples, expected {num_sequences1} (from activations)."
        
        assert isinstance(stat_array2, np.ndarray) and stat_array2.ndim == 2, \
            f"Stat '{stat_name}' in all_logit_stats_data2 must be a 2D array (N, T_stat)."
        assert stat_array2.shape[0] == num_sequences2, \
            f"Stat '{stat_name}' in all_logit_stats_data2 has {stat_array2.shape[0]} samples, expected {num_sequences2} (from activations)."

        max_stat_seq_len1 = stat_array1.shape[1]
        max_stat_seq_len2 = stat_array2.shape[1]
        for prev_idx in prev_token_indices:
            assert -max_stat_seq_len1 <= prev_idx < max_stat_seq_len1, \
                f"prev_tok_idx {prev_idx} is out of bounds for stat '{stat_name}' in dataset 1 (stat_seq_len: {max_stat_seq_len1})."
            assert -max_stat_seq_len2 <= prev_idx < max_stat_seq_len2, \
                f"prev_tok_idx {prev_idx} is out of bounds for stat '{stat_name}' in dataset 2 (stat_seq_len: {max_stat_seq_len2})."

    assert isinstance(rng, np.random.Generator), "rng must be a numpy.random.Generator instance."
    assert binning_strategy in ["quantile", "uniform"], "binning_strategy must be 'quantile' or 'uniform'."

    # --- Collect Individual Keep Indices for each Previous Token Criterion ---
    list_of_keep1_indices_arrays: List[np.ndarray] = []
    list_of_keep2_indices_arrays: List[np.ndarray] = []

    for prev_tok_idx in prev_token_indices:
        # Prepare stats for this specific previous token using dict comprehensions
        stats1_for_one_prev_tok = {
            stat_name: all_logit_stats_data1[stat_name][:, prev_tok_idx]
            for stat_name in matched_stats
        }
        stats2_for_one_prev_tok = {
            stat_name: all_logit_stats_data2[stat_name][:, prev_tok_idx]
            for stat_name in matched_stats
        }
        
        # Verify shapes after slicing (should be 1D arrays of length num_sequencesX)
        for stat_name in matched_stats:
            assert stats1_for_one_prev_tok[stat_name].shape == (num_sequences1,), \
                f"Sliced stat '{stat_name}' for prev_tok_idx {prev_tok_idx} (dataset 1) has wrong shape."
            assert stats2_for_one_prev_tok[stat_name].shape == (num_sequences2,), \
                f"Sliced stat '{stat_name}' for prev_tok_idx {prev_tok_idx} (dataset 2) has wrong shape."

        # Balance based on this single previous token's stats
        _, _, _, _, keep1_this_criterion, keep2_this_criterion = balanced_sample_by_stats(
            acts_1=acts_curr_tok1,
            acts_2=acts_curr_tok2,
            stats_1=stats1_for_one_prev_tok,
            stats_2=stats2_for_one_prev_tok,
            match_on=matched_stats,
            n_bins=n_bins_for_prev_stats,
            binning=binning_strategy,
            rng=rng,
        )
        
        assert len(keep1_this_criterion) > 0, \
            f"Balancing on prev_tok_idx {prev_tok_idx} yielded 0 samples for dataset 1. Check n_bins or data distribution."
        assert len(keep2_this_criterion) > 0, \
            f"Balancing on prev_tok_idx {prev_tok_idx} yielded 0 samples for dataset 2. Check n_bins or data distribution."

        list_of_keep1_indices_arrays.append(keep1_this_criterion)
        list_of_keep2_indices_arrays.append(keep2_this_criterion)

    # --- Intersect Keep Indices (AND mask over all criteria) ---
    keep_idxs1_multitok: np.ndarray = list_of_keep1_indices_arrays[0]
    for i in range(1, len(list_of_keep1_indices_arrays)):
        keep_idxs1_multitok = np.intersect1d(
            keep_idxs1_multitok, list_of_keep1_indices_arrays[i], assume_unique=True
        )
    assert len(keep_idxs1_multitok) > 0, \
        "Intersection of all balancing criteria resulted in 0 samples for dataset 1. Try fewer prev_tokens, different n_bins, or check data."

    keep_idxs2_multitok: np.ndarray = list_of_keep2_indices_arrays[0]
    for i in range(1, len(list_of_keep2_indices_arrays)):
        keep_idxs2_multitok = np.intersect1d(
            keep_idxs2_multitok, list_of_keep2_indices_arrays[i], assume_unique=True
        )
    assert len(keep_idxs2_multitok) > 0, \
        "Intersection of all balancing criteria resulted in 0 samples for dataset 2. Try fewer prev_tokens, different n_bins, or check data."

    # --- Downsample to Equal Length ---
    min_final_len = min(len(keep_idxs1_multitok), len(keep_idxs2_multitok))

    assert min_final_len > 0, \
        "After intersection, one of the datasets has 0 eligible samples (min_final_len is 0). Cannot proceed."

    final_keep_idxs1 = rng.choice(keep_idxs1_multitok, size=min_final_len, replace=False)
    final_keep_idxs2 = rng.choice(keep_idxs2_multitok, size=min_final_len, replace=False)
    
    # --- Subsample Current Activations ---
    acts_curr_tok1_balanced = acts_curr_tok1[final_keep_idxs1]
    acts_curr_tok2_balanced = acts_curr_tok2[final_keep_idxs2]

    return acts_curr_tok1_balanced, acts_curr_tok2_balanced, final_keep_idxs1, final_keep_idxs2


# --------------------------------------------------------------------------- #
#  Helpers for balanced_sample_by_stats                                       #
# --------------------------------------------------------------------------- #
def _compute_edges_uniform(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-width edges spanning [min, max].  length = n_bins+1"""
    return np.linspace(values.min(), values.max(), n_bins + 1, dtype=np.float32)


def _compute_edges_quantile_sklearn(values: np.ndarray, n_bins: int) -> np.ndarray:
    est = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    # est.fit(values.reshape(-1, 1))
    # return est.bin_edges_[0]

    with warnings.catch_warnings():
         # Disable occassional warnings about not being able to make enough bins and returning fewer than requested
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.preprocessing._discretization")
        est.fit(values.reshape(-1, 1))
    return est.bin_edges_[0]


def _digitise(values: np.ndarray,
              edges: np.ndarray) -> np.ndarray:
    """Bin values using np.digitize with *left-closed, right-open* bins."""
    return np.digitize(values, edges[:-1], right=False).astype(np.int32)


# --------------------------------------------------------------------------- #
#  Main balancing routine                                                     #
# --------------------------------------------------------------------------- #
def balanced_sample_by_stats(
    acts_1: np.ndarray,
    acts_2: np.ndarray,
    *,
    stats_1: Dict[str, np.ndarray],
    stats_2: Dict[str, np.ndarray],
    match_on: Optional[Sequence[str]] = None,
    n_bins: int | Dict[str, int] = 40,
    binning: str = "uniform",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray,
           Dict[str, np.ndarray], Dict[str, np.ndarray],
           np.ndarray, np.ndarray]:
    """
    Balance two activation sets so their joint histogram over the chosen
    statistics is identical (up to binning granularity).

    Parameters
    ----------
    binning
        *"uniform"* (default) → equal-width bins.  
        *"quantile"*          → equal-frequency bins (percentiles).

        The strategy is applied *per statistic* on the **combined** samples from
        both classes so that edges are shared.
    """
    assert binning in {"uniform", "quantile"}, "binning must be 'uniform' or 'quantile'"
    if rng is None:
        rng = np.random.default_rng()

    # --- select stats ------------------------------------------------------
    if match_on is None:
        match_on = list(stats_1.keys())
    for name in match_on:
        assert name in stats_1 and name in stats_2, f"stat '{name}' missing"

    # --- n_bins per stat ---------------------------------------------------
    if isinstance(n_bins, numbers.Integral):  # any int type
        n_bins = {name: n_bins for name in match_on}
    assert isinstance(n_bins, dict)
    
    # ---------------------------------------------------------------------- #
    # 1  Compute bin edges & digitise each statistic                         #
    # ---------------------------------------------------------------------- #
    def choose_edges(all_vals: np.ndarray, m: int) -> np.ndarray:
        return (_compute_edges_uniform if binning == "uniform"
                else _compute_edges_quantile_sklearn)(all_vals, m)

    def to_bins(stats_dict: Dict[str, np.ndarray]) -> np.ndarray:
        labels = []
        for name in match_on:
            all_vals = np.concatenate([stats_1[name], stats_2[name]])
            edges    = choose_edges(all_vals, n_bins[name])

            if len(edges) - 1 != n_bins[name]:
                print(f"binning {name} -- requested {n_bins[name]}, got {len(edges) - 1}")

            labels.append(_digitise(stats_dict[name], edges))
        return np.stack(labels, axis=1)           # (N_examples, N_stats)

    bins_1 = to_bins(stats_1)
    bins_2 = to_bins(stats_2)

    # ---------------------------------------------------------------------- #
    # 2  Collapse multi-dim bin -> integer code                              #
    # ---------------------------------------------------------------------- #
    radix_sizes = np.array([n_bins[name] for name in match_on], dtype=np.int64)
    multipliers = np.concatenate([[1], np.cumprod(radix_sizes[:-1])])
    codes_1 = (bins_1 * multipliers).sum(axis=1)
    codes_2 = (bins_2 * multipliers).sum(axis=1)

    # ---------------------------------------------------------------------- #
    # 3  Sample equal counts per code                                        #
    # ---------------------------------------------------------------------- #
    common = np.intersect1d(codes_1, codes_2)
    assert common.size, "No overlap; lower n_bins or reduce #stats"

    keep1, keep2 = [], []
    for c in common:
        idx1, idx2 = np.where(codes_1 == c)[0], np.where(codes_2 == c)[0]
        m = min(idx1.size, idx2.size)
        keep1.append(rng.choice(idx1, m, replace=False))
        keep2.append(rng.choice(idx2, m, replace=False))
    keep1, keep2 = np.concatenate(keep1), np.concatenate(keep2)
    rng.shuffle(keep1), rng.shuffle(keep2)

    # ---------------------------------------------------------------------- #
    # 4  Slice and return                                                    #
    # ---------------------------------------------------------------------- #
    def slc(a, idx): 
        return {k: v[idx] for k, v in a.items()}

    return acts_1[keep1], acts_2[keep2], slc(stats_1, keep1), slc(stats_2, keep2), keep1, keep2


def calculate_logit_stats(
    logits: torch.Tensor,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    """
    Calculates various statistics over the vocabulary dimension of a logit batch.

    Performs most calculations (mean, std, min/max, percentiles, norms, entropy)
    on the GPU using PyTorch, then transfers results to CPU NumPy arrays.
    Calculates skewness and kurtosis on the CPU using SciPy after transferring
    the raw logits.

    Args:
        logits: A PyTorch tensor of shape (B, T, V), potentially on GPU.
        input_ids: Optional int tensor of shape (B, T). If provided, per-position
            loss (next-token NLL) and cumulative loss stats are computed.
        attention_mask: Optional binary tensor of shape (B, T). If provided, it
            is used to mask padded positions for loss-related stats.

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
    stats_torch["logsumexp"] = torch.logsumexp(logits_float, dim=-1).to(dtype_torch)

    # Percentiles using torch.quantile
    quantiles_tensor = PERCENTILES_TO_CALCULATE_TORCH.to(device)
    percentile_values = torch.quantile(logits_float, q=quantiles_tensor, dim=-1) # Shape: (num_quantiles, B, T)
    for i, q in enumerate(PERCENTILES_TO_CALCULATE_TORCH):
        percentile_key = f"percentile_{(q * 100):.0f}"
        stats_torch[percentile_key] = percentile_values[i].to(dtype_torch)

    # Norms using torch.linalg.norm
    stats_torch["norm_l1"] = torch.linalg.norm(logits_float, ord=1, dim=-1).to(dtype_torch)
    stats_torch["norm_l2"] = torch.linalg.norm(logits_float, ord=2, dim=-1).to(dtype_torch)

    # Entropy using torch.log_softmax (also reused for loss)
    log_probs = torch.log_softmax(logits_float, dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    stats_torch["entropy"] = entropy.to(dtype_torch)
    
    # Helper function to create backward-looking versions
    def make_backward_looking(tensor, fill_value=0):
        """Shift tensor right by 1 and fill first position with fill_value."""
        # pad = (left, right) for last dimension
        return torch.nn.functional.pad(tensor[:, :-1], (1, 0), value=fill_value)
    
    # Cumulative entropy
    cumulative_entropy = torch.cumsum(entropy, dim=-1)
    stats_torch["cumulative_entropy"] = cumulative_entropy.to(dtype_torch)
    stats_torch["cumulative_entropy_prev"] = make_backward_looking(cumulative_entropy, 0).to(dtype_torch)
    
    # Cumulative max log prob
    max_log_probs = log_probs.max(dim=-1).values
    cumulative_max_log_prob = torch.cumsum(max_log_probs, dim=-1)
    stats_torch["cumulative_max_log_prob"] = cumulative_max_log_prob.to(dtype_torch)
    stats_torch["cumulative_max_log_prob_prev"] = make_backward_looking(cumulative_max_log_prob, 0).to(dtype_torch)
    
    # Running min/max entropy
    running_min_entropy = torch.cummin(entropy, dim=-1).values
    running_max_entropy = torch.cummax(entropy, dim=-1).values
    
    stats_torch["running_min_entropy"] = running_min_entropy.to(dtype_torch)
    stats_torch["running_min_entropy_prev"] = make_backward_looking(running_min_entropy, float('inf')).to(dtype_torch)
    
    stats_torch["running_max_entropy"] = running_max_entropy.to(dtype_torch)
    stats_torch["running_max_entropy_prev"] = make_backward_looking(running_max_entropy, float('-inf')).to(dtype_torch)

    # --- Loss-related stats (optional) ------------------------------------
    if input_ids is not None:
        if input_ids.device != logits.device:
            input_ids = input_ids.to(logits.device)
        if attention_mask is not None and attention_mask.device != logits.device:
            attention_mask = attention_mask.to(logits.device)

        loss_stats = _compute_loss_stats_from_log_probs(log_probs, input_ids, attention_mask)
        for k, v in loss_stats.items():
            stats_torch[k] = v.to(dtype_torch)

    # --- Move PyTorch results to CPU NumPy arrays ---
    for name, tensor in stats_torch.items():
        stats_np[name] = tensor.cpu().numpy().astype(dtype_np) # Ensure float32 just in case

    # --- Calculate Skewness & Kurtosis on CPU using SciPy ---
    # Move raw logits to CPU NumPy *once* for these calculations
    logits_np = logits_float.cpu().numpy()

    # Calculate unbiased skewness & unbiased Fisher's kurtosis (excess kurtosis)
    stats_np["skewness"] = scipy.stats.skew(logits_np, axis=-1, bias=False).astype(dtype_np)
    stats_np["kurtosis"] = scipy.stats.kurtosis(logits_np, axis=-1, fisher=True, bias=False).astype(dtype_np)

    return stats_np


def calculate_activation_stats(activations: torch.Tensor) -> Dict[str, np.ndarray]:
    """
    Calculates statistics over the d_model dimension of activation tensors.
    
    Args:
        activations: A PyTorch tensor of shape (B, T, d_model), potentially on GPU.
    
    Returns:
        A dictionary where keys are stat names (str) and values are NumPy arrays
        of shape (B, T) containing the calculated statistics (as np.float32).
    """
    stats_torch = {}
    dtype_torch = torch.float32
    dtype_np = np.float32
    
    # Ensure activations are float for calculations
    acts_float = activations.float()
    d_model = acts_float.shape[-1]
    
    # --- Calculate stats on original device (GPU potentially) using PyTorch ---
    stats_torch["mean"] = acts_float.mean(dim=-1).to(dtype_torch)
    stats_torch["std"] = acts_float.std(dim=-1, unbiased=True).to(dtype_torch)
    stats_torch["max"] = acts_float.max(dim=-1).values.to(dtype_torch)
    stats_torch["min"] = acts_float.min(dim=-1).values.to(dtype_torch)
    
    # Normalized norms
    stats_torch["l1_norm"] = torch.linalg.norm(acts_float, ord=1, dim=-1).to(dtype_torch) / d_model
    stats_torch["l2_norm"] = torch.linalg.norm(acts_float, ord=2, dim=-1).to(dtype_torch) / (d_model ** 0.5)
    
    # --- Move PyTorch results to CPU NumPy arrays ---
    stats_np = {}
    for name, tensor in stats_torch.items():
        stats_np[name] = tensor.cpu().numpy().astype(dtype_np)
    
    # --- Calculate Skewness & Kurtosis on CPU using SciPy ---
    acts_np = acts_float.cpu().numpy()
    stats_np["skewness"] = scipy.stats.skew(acts_np, axis=-1, bias=False).astype(dtype_np)
    stats_np["kurtosis"] = scipy.stats.kurtosis(acts_np, axis=-1, fisher=True, bias=False).astype(dtype_np)
    
    return stats_np


def get_activations_and_logit_stats(
    model: HookedTransformer,
    data: Sequence[str],
    batch_size: int = 128,
    hook_substr: str = "hook_resid_post",
    device: str | torch.device | None = None,
    keep_every: int = 1,
    keep_from_layer: int | None = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
    """
    Collect activations, logit statistics, and activation statistics for every example.

    Parameters
    ----------
    keep_every
        If > 1, store activations only for layers whose index is a multiple of
        `keep_every`.  Default 1 keeps every layer (original behaviour).
    keep_from_layer
        If given, store activations only for layers with index ≥ this value.
        Can be combined with `keep_every`.

    Returns
    -------
    activations : Dict[str, np.ndarray]
        Each array has shape (N_examples, T, d_model), dtype float32. Contains
        residual stream activations specified by `hook_substr`.
    logit_stats_dict : Dict[str, np.ndarray]
        Each array has shape (N_examples, T), dtype float32. Contains statistics
        computed over the vocabulary dimension of the logits.
    activation_stats_dict : Dict[str, Dict[str, np.ndarray]]
        Nested dict where activation_stats_dict[layer_name][stat_name] is an
        array of shape (N_examples, T), dtype float32.
    """
    acts_batches: Dict[str, List[torch.Tensor]] = defaultdict(list)
    stats_batches_np: Dict[str, List[np.ndarray]] = defaultdict(list)
    act_stats_batches_np: Dict[str, Dict[str, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))

    model.eval()
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            batch = data[start : start + batch_size]

            # Run model and get logits and cache
            logits, cache = model.run_with_cache(batch, device=device)
            # logits shape: (B, T, V), on specified device or model's device

            # Calculate logit stats with loss always enabled
            tokens = model.to_tokens(batch, move_to_device=True)
            # Use same BOS behavior as model defaults
            prepend_bos = model.cfg.default_prepend_bos
            attention_mask = tl_utils.get_attention_mask(model.tokenizer, tokens, prepend_bos)
            batch_logit_stats_np: Dict[str, np.ndarray] = calculate_logit_stats(
                logits, input_ids=tokens, attention_mask=attention_mask
            )
            for stat_name, array_np in batch_logit_stats_np.items():
                stats_batches_np[stat_name].append(array_np)

            # Process activations
            for layer_name, tensor in cache.items():
                if hook_substr not in layer_name:
                    continue

                # Extract numeric index:  "blocks.{idx}.hook_resid_post" → idx
                idx = int(layer_name.split(".")[1])

                passes_every = (idx % keep_every) == 0
                passes_cutoff = (keep_from_layer is None) or (idx >= keep_from_layer)
                if not (passes_every and passes_cutoff):
                    continue

                # Calculate activation stats before moving to CPU
                batch_act_stats_np = calculate_activation_stats(tensor)
                for stat_name, array_np in batch_act_stats_np.items():
                    act_stats_batches_np[layer_name][stat_name].append(array_np)

                # Store activations (move tensor to CPU)
                acts_batches[layer_name].append(tensor.cpu())

            # Housekeeping
            del cache, logits, batch_logit_stats_np
            torch.cuda.empty_cache()
            
    # Concatenate batches for activations
    activations = {
        name: torch.concat(t_list, dim=0).to(torch.float32).numpy()
        for name, t_list in acts_batches.items()
    }

    # Concatenate batches for logit statistics
    logit_stats_dict = {
        stat_name: np.concatenate(arr_list, axis=0)
        for stat_name, arr_list in stats_batches_np.items()
    }
    
    # Concatenate batches for activation statistics
    activation_stats_dict = {}
    for layer_name, stat_dict in act_stats_batches_np.items():
        activation_stats_dict[layer_name] = {
            stat_name: np.concatenate(arr_list, axis=0)
            for stat_name, arr_list in stat_dict.items()
        }

    return activations, logit_stats_dict, activation_stats_dict


def train_linear_probe(
    x1,
    x2,
    num_cross_val: int = 5,
    shuffle_seed: int = 0,
    *,
    probe_type: str = "logreg",  # "logreg" | "lda"
    # Logistic Regression parameters
    penalty: str = "l2",
    C: float = 1.0,
    max_iter: int = 1000,
    solver: str = 'lbfgs',
    # LDA parameters
    lda_shrinkage: str | float | None = "auto",
    lda_solver: str = "lsqr",
):
    if np.allclose(x1, x2):
        return {
            "cv_scores": [0.5] * max(num_cross_val, 1), # dummy score
            "trained_classifier": None,
            "cv_estimators": None,
        }

    # --------------------------- prepare inputs ---------------------------
    # g designates the group axis (two groups: class 0 and class 1).
    X = rearrange([x1, x2], "g n d -> (g n) d")
    y = rearrange([np.zeros(len(x1)), np.ones(len(x2))], "g n -> (g n)")
    X, y = shuffle(X, y, random_state=shuffle_seed)
    # ---------------------------- choose model ---------------------------
    
    cv_scores, cv_estimators = None, None
    assert probe_type in ["logreg", "lda"], "probe_type must be 'logreg' or 'lda'"
    if probe_type == "logreg":
        clf = LogisticRegression(random_state=0, max_iter=max_iter, penalty=penalty, C=C, solver=solver)

        if num_cross_val > 1:
            cv = StratifiedKFold(n_splits=num_cross_val, shuffle=True, random_state=shuffle_seed)
            cv_res = cross_validate(clf, X, y, cv=cv, scoring="accuracy", 
                                    n_jobs=num_cross_val, return_estimator=True, return_train_score=True)
            cv_scores = cv_res["test_score"]
            cv_estimators = cv_res["estimator"]

    elif probe_type == "lda":
        clf = LDA(n_components=1, shrinkage=lda_shrinkage, solver=lda_solver, store_covariance=False)

        if num_cross_val > 1:
            cv = StratifiedKFold(n_splits=num_cross_val, shuffle=True, random_state=shuffle_seed)
            cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
            cv_estimators = None

    # ----------------------------- final fit -----------------------------
    clf.fit(X, y)

    return {
        "cv_scores": cv_scores,
        "cv_estimators": cv_estimators,
        "trained_classifier": clf,
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
    
    acts_data1, logit_stats_data1 = get_activations_and_logit_stats(model, data1)
    acts_data2, logit_stats_data2 = get_activations_and_logit_stats(model, data2)
    
    n_examples, n_tokens, d_model = acts_data1[list(acts_data1.keys())[0]].shape
    # print(f'n_examples: {n_examples} \t n_tokens: {n_tokens} \t d_model: {d_model}')
    
    score_grid, clf_grid = train_probes_per_layer_and_token(acts_data1, acts_data2)
    
    return score_grid, clf_grid, data1, data2, acts_data1, acts_data2


def train_probes_per_layer_and_token(acts1, acts2, C=1.0):
     # TODO check that the shapes are the same and that the keys are the same
     # TODO option to skip some layers
     # TODO pass some params to the linear probe
    n_examples, n_tokens, d_model = acts1[list(acts1.keys())[0]].shape
    layer_names = list(acts1.keys())
    score_grid = np.zeros((n_tokens, len(layer_names)))
    clf_grid = [[None for _ in layer_names] for _ in range(n_tokens)]  # classifier grid

    for layer in layer_names:
        print(f'training probes for layer {layer}')
        for token_idx in range(n_tokens):
            # train linear probe on activations for token i
            result = train_linear_probe(
                acts1[layer][:, token_idx, :], 
                acts2[layer][:, token_idx, :],
                C=C
            )
            
            score_grid[token_idx, layer_names.index(layer)] = np.mean(result['cv_scores'])
            clf_grid[token_idx][layer_names.index(layer)] = result['trained_classifier']  # store classifier

    score_grid = score_grid[1:, :]  # remove BOS token that transformerlens adds automatically
    clf_grid = clf_grid[1:]         # also remove BOS classifiers for consistency
    return score_grid, clf_grid


def plot_score_grid(scores, tokens: List[str], title=None, vmin=0.49, vmax=1.01, cmap='Blues', figsize=(6, 2.2),
                    plot_name='linear_probe', plt_path='plots', plt_format='pdf'):
    """
    Plot a grid of scores, with tokens on the x axis and layers on the y axis.
    scores: np array with shape (num_tokens, num_layers)
    """
    # larger font size and times new roman font
    plt.rc('font', size=14)#, family='Times New Roman')
    plt.rc('text', usetex=False)
    
    fig, ax = plt.subplots(figsize=figsize)

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
