# %%
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from transformer_lens import HookedTransformer

# Project utilities you already have
from data_generation.load_data_from_config import generate_data_from_experiment_folder
from utils.linear_probes import (
    leave_unique_q_type,
    get_activations_and_logit_stats,
    load_model_to_transformerlens,
)
from utils.io_utils import save_activation_summary_npz, unique_path
from utils.plotting_training_data_order import train_stage_pair_probes

# ------------------------------------------------------------
# seed assurance helpers (strict)
# ------------------------------------------------------------

def _find_seeds_in_path(path: str) -> List[int]:
    """Return all seeds found from path components ending in "_s{seed}" or "s{seed}".
    Matches your stage naming like: stage1_s42, stage6_s7, etc.
    """
    seeds: set[int] = set()
    for part in Path(path).parts:
        m = re.search(r"(?:^|_)s(\d+)$", part)
        if m:
            seeds.add(int(m.group(1)))
    return sorted(seeds)


def _assure_seed_from_path(model_path: str, seed: int) -> int:
    """Ensure the caller-provided seed matches the seed encoded in the path.
    - Raises if no seed suffix is present in the path.
    - Raises if multiple (conflicting) seeds appear in the path.
    - Raises if the path seed != provided seed.
    Returns the assured seed.
    """
    seeds = _find_seeds_in_path(model_path)
    if not seeds:
        raise ValueError(
            f"No 's{{seed}}' suffix found anywhere in model path: {model_path}. "
            "Your stage dirs must end with s{seed} (e.g., stage3_s42)."
        )
    if len(seeds) > 1:
        raise ValueError(
            f"Multiple seed suffixes found in path ({seeds}). "
            "Make sure only one stage dir with s{seed} appears in the path."
        )
    path_seed = seeds[0]
    if path_seed != seed:
        raise ValueError(
            f"Seed mismatch: path implies seed {path_seed} (…_s{path_seed}), "
            f"but you passed {seed}."
        )
    return path_seed


# ------------------------------------------------------------
# small helpers
# ------------------------------------------------------------

def _prompt_key_prefix(prompt_type: str) -> str:
    # matches your dataset naming e.g. 'ent_assoc_who_qd1consis'
    return f"ent_assoc_{prompt_type}_"


def _ordered_subset_keys(data_keys: Sequence[str], prefix: str) -> List[str]:
    # stable subset order used before
    order = ["qd1consis", "qd1incons", "qd2consis", "qd2incons", "qd4consis", "q"]
    print(data_keys)
    print(prefix)
    picked = [k for k in data_keys if k.startswith(prefix)]
    main = [prefix + o for o in order if (prefix + o) in picked]
    extras = sorted([k for k in picked if k not in main])
    print(f'Returning {main}, skipping {extras}')
    return main


def _filter_for_prompt(texts: List[str], model: HookedTransformer, prompt_type: str, natural_style_vars: bool) -> List[str]:
    # replicate earlier q_type normalisation for the filter util
    q = {"standFor": "stand for", "who": "Who", "meaning": "mean"}.get(prompt_type, prompt_type)
    var_len = 5 if natural_style_vars else 3
    return leave_unique_q_type(texts, model, q, var_len)


# ------------------------------------------------------------
# core runner
# ------------------------------------------------------------

def collect_for_model(
    model_path: str,
    seed: int,
    *,
    prompt_types: Sequence[str] = ("who", "name", "standFor", "mean"),
    batch_size: int = 256,
    keep_every: int = 1,
    probing_C_logreg: float = 0.1,
    probing_token_indices: list[int] = list(range(-1, -9, -1)), # [-1, -2, ..., -8]
    keep_from_layer: int | None = None,
    train_and_save_probes: bool = False,
) -> None:
    """For one finetuned model path + seed: regenerate data, collect, and save
    a single NPZ *per prompt type* with centroids/percentiles only, using
    `save_activation_summary_npz` from your utils module.

    Strong seed assurance:
    - Requires the model_path (or one of its parent components) to end with `_s{seed}`.
    - Raises if the encoded seed != the provided `seed`.
    - Uses this assured seed for all data generation stages.
    """
    assert isinstance(prompt_types, Sequence) and all(isinstance(pt, str) for pt in prompt_types), \
        "prompt_types must be a sequence of strings"
    
    # --- seed assurance ---------------------------------------------------
    assured_seed = _assure_seed_from_path(model_path, seed)

    # --- config + data ----------------------------------------------------
    config_folder = Path(model_path).parent.as_posix() if not 'checkpoint' in model_path else Path(model_path).parent.parent.as_posix()
    
    data, params_used, cfg = generate_data_from_experiment_folder(
        folder_path=config_folder,
        seed=assured_seed,
        seed_stage2=0,  # always use 0 for stage2 seed
        train_subset="full",
    )

    # Also assert if config exposes a seed and it disagrees
    cfg_seed = None
    for attr in ("seed", "data_seed", "train_seed"):
        if hasattr(cfg, attr):
            try:
                cfg_seed = int(getattr(cfg, attr))
                break
            except Exception:
                pass
    if cfg_seed is not None and cfg_seed != assured_seed:
        raise ValueError(f"Config seed {cfg_seed} != assured seed {assured_seed} from path")

    # --- model ------------------------------------------------------------
    base_model_name = getattr(getattr(cfg, "model_arguments", None), "model_name_or_path", None)
    if not base_model_name or os.path.exists(str(base_model_name)):
        base_model_name = "meta-llama/Llama-3.2-1B"
    model = load_model_to_transformerlens(model_path, base_model_name)

    natural_style_vars = bool(getattr(params_used, "natural_style_vars", False) if hasattr(params_used, "__dict__") else params_used.get("natural_style_vars", False))

    # --- per prompt type --------------------------------------------------
    for prompt in prompt_types:
        prefix = _prompt_key_prefix(prompt)
        subset_keys = _ordered_subset_keys(list(data.keys()), prefix)
        assert subset_keys, f"No data found for prompt '{prompt}' in {model_path}"

        # gather text lists per subset
        raw_groups = [data[k]['question'] for k in subset_keys]
        print(f"Found {len(raw_groups)} subsets for prompt '{prompt}': {subset_keys}")
        
        # two example texts from first group
        print(f"Example texts from first group: {raw_groups[0][:2] if raw_groups else 'N/A'}")
        
        # filter for uniqueness/length (mirrors notebook behaviour)
        groups = [
            _filter_for_prompt(texts, model, prompt, natural_style_vars) if texts else []
            for texts in raw_groups
        ]
        # trim to same size for fair comparison
        n = min((len(g) for g in groups if len(g) > 0), default=0)
        groups = [g[:n] for g in groups]
        if n == 0:
            print(f"[skip] empty after filtering for prompt '{prompt}'")
            continue

        print(f'Data lens: {[len(g) for g in groups]}')

        # collect activations per subset (not saved; only summaries are written)
        acts_per_subset: List[Dict[str, np.ndarray]] = []
        sample_text = groups[0][0]
        for texts in groups:
            acts, _, _ = get_activations_and_logit_stats(
                model,
                texts,
                batch_size=batch_size,
                keep_every=keep_every,
                keep_from_layer=keep_from_layer,
            )
            acts_per_subset.append(acts)

        # Save via shared utility (handles token window + unique naming)
        # out_dir = Path(model_path).parent
        basename = f"activation-centroids-and-percentiles-{prompt}-seed{assured_seed}"               
        
        save_activation_summary_npz(
            datasets=acts_per_subset,
            out_dir=model_path,
            basename=basename,
            seed=assured_seed,
            prompt_type=prompt,
            model_path=model_path,
            model=model,
            sample_text=sample_text,
            dataset_names=[f"D{i+1}" for i in range(len(acts_per_subset))],
            percentiles=(0.05, 0.50, 0.95),
            dtype="float32",
            version=1,
        )
        
        # ------------------------------------------------------------
        # PROBE TRAINING & SAVING 
        # ------------------------------------------------------------
        if train_and_save_probes:
            layer_names = sorted(
                list(acts_per_subset[0].keys()),
                key=lambda n: int(re.search(r"blocks\.(\d+)\.", n).group(1)), # numeric sort by block index (will crash if name doesn't match pattern)
            )
            token_indices = probing_token_indices

            perf_dict = {L: {} for L in layer_names}
            weights_dict = {L: {} for L in layer_names}

            for L in layer_names:
                print(f"Training probes for layer {L} for {len(token_indices)} token positions")
                for tok in token_indices:
                    W, stage_idxs, perf = train_stage_pair_probes(
                        acts_per_subset,
                        layer_name=L,
                        token_idx=tok,
                        pairs="all",
                        C=probing_C_logreg,
                    )
                    perf_dict[L][tok] = np.asarray(perf, dtype=np.float32)
                    weights_dict[L][tok] = np.asarray(W, dtype=np.float32)

            meta = dict(
                layer_names=np.asarray(layer_names, dtype=object),
                token_indices=np.asarray(token_indices, dtype=np.int64),
                subset_keys=np.asarray(subset_keys, dtype=object),
                prompt=np.asarray(prompt),
                seed=np.asarray(assured_seed),
                model_path=np.asarray(str(model_path)),
                stage_idxs=np.asarray(stage_idxs, dtype=np.int64),
            )
            perf_out = unique_path(model_path, f"probe-perf-{prompt}-seed{assured_seed}", suffix='.npz')
            np.savez_compressed(perf_out, perf=np.array(perf_dict, dtype=object), **meta)
            print(f"[saved] {perf_out}")

            weights_out = unique_path(model_path, f"probe-weights-{prompt}-seed{assured_seed}", suffix='.npz')
            np.savez_compressed(weights_out, weights=np.array(weights_dict, dtype=object), **meta)
            print(f"[saved] {weights_out}")
        # ------------------------------------------------------------
        # PROBE STUFF DONE -------------------------------------------
        # ------------------------------------------------------------

        # drop heavy refs ASAP
        del acts_per_subset
        torch.cuda.empty_cache()
    

# ------------------------------------------------------------
# convenience: run many
# ------------------------------------------------------------

def collect_many(
    jobs: Sequence[tuple[str, int]],
    *,
    prompt_types: Sequence[str] = ("who", "name", "standFor", "meaning"),
    batch_size: int = 256,
    keep_every: int = 1,
    keep_from_layer: int | None = None,
    probing_C_logreg: float = 0.1,
    probing_token_indices: list[int] = list(range(-1, -9, -1)),
) -> None:
    for model_path, seed in jobs:
        print()
        print(f"=== {model_path} — seed s{seed} ===")
        collect_for_model(
            model_path,
            seed,
            prompt_types=prompt_types,
            batch_size=batch_size,
            keep_every=keep_every,
            keep_from_layer=keep_from_layer,
            probing_C_logreg=probing_C_logreg,
            probing_token_indices=probing_token_indices,
        )

# %%
seed = 600

stage1_path = f'stage1_s{seed}'
stage2_path = f'stage2_s{seed}'
stage3_path = f'stage3_s{seed}'
stage4_path = f'stage4_s{seed}'
stage5_path = f'stage5_s{seed}'
stage6_path = f'stage6_s{seed}'

# d1->d2->d3->d2
model_path = f'experiments/data_order_qa_cvdb_tveDefs_nEnts16000_eps5and5and5and5_bs256and256and256and256_Llama_3.2_1B_ADAFACTOR_4stage/{stage4_path}'  

# d1->d2->d3->d1
model_path = f'experiments/qd1_last_qa_cvdb_tveDefs_nEnts16000_eps5and5and5and5_bs256and256and256and256_Llama_3.2_1B_ADAFACTOR_4stage/{stage4_path}'

# 6 distinct datasets in a sequence LLAMA
model_path = f'experiments/qd1_last_qa_cvdb_tveDefs_nEnts16000_eps5and5and5and5and5and5_bs256and256and256and256and256and256_Llama_3.2_1B_ADAFACTOR_6stage/{stage6_path}'

# # 6 distinct configs LORA
# model_path = f'experiments/qa_cvdb_tveDefs_nEnts16000_eps5-5-5-5-5-5_bs256-256-256-256-256-256_Llama_3.2_1B_ADAFACTOR_loraR32_loraAlpha32_6stage/{stage6_path}'
# model_path = f'experiments/qa_cvdb_tveDefs_nEnts16000_eps5-5-5-5-5-5_bs256-256-256-256-256-256_Llama_3.2_1B_ADAFACTOR_loraR128_loraAlpha128_6stage/{stage6_path}'

# # 6 stage llama 8b
# model_path = f'experiments/qa_cvdb_tveDefs_nEnts16000_eps5-5-5-5-5-5_bs256-256-256-256-256-256_Llama_3.1_8B_ADAFACTOR_loraR128_loraAlpha128_6stage/{stage6_path}'
# # # 6 stage qwen3 8b
# model_path = f'experiments/qa_cvdb_tveDefs_nEnts16000_eps5-5-5-5-5-5_bs256-256-256-256-256-256_Qwen3_8B_ADAFACTOR_loraR128_loraAlpha128_6stage/{stage6_path}'

# model_path = f'experiments/qd1_last_qa_cvdb_tveDefs_nEnts16000_eps10_bs256_stage6_s600_ADAFACTOR_single_stage/s{seed}/checkpoint-141'
# model_path = f'experiments/qd1_last_qa_cvdb_tveDefs_nEnts16000_eps10_bs256_stage6_s600_ADAFACTOR_single_stage/s{seed}'  # same thing above but finetuned on stage2 data for a while after the original six stages

# # natural vars and questions
# model_path = f'experiments/qd1_last_qa_cvdb_tveDefs_nEnts16000_eps5and5and5and5and5and5_bs256and256and256and256and256and256_Llama_3.2_1B_ADAFACTOR_naturTrainQs_naturVars_6stage/{stage6_path}'


# 32k entities llama 3.2 1b
# model_path = f'experiments/qa_cvdb_tveDefs_nEnts32000_eps5and5and5and5and5and5_bs256and256and256and256and256and256_Llama_3.2_1B_ADAFACTOR_naturTrainQs_naturVars_6stage/{stage6_path}'

# # qwen
# model_path = f'experiments/qa_cvdb_tveDefs_nEnts16000_eps5and5and5and5and5and5_bs256and256and256and256and256and256_Qwen2.5_0.5B_ADAFACTOR_6stage/{stage6_path}'

# # qwen 1.5b
# model_path = f'experiments/qa_cvdb_tveDefs_nEnts16000_eps5and5and5and5and5and5_bs256and256and256and256and256and256_Qwen2.5_1.5B_ADAFACTOR_6stage/{stage6_path}'

# qwen 3b
# model_path = f'experiments/qa_cvdb_tveDefs_nEnts16000_eps5and5and5and5and5and5_bs256and256and256and256and256and256_Qwen2.5_3B_ADAFACTOR_6stage/{stage6_path}'

# # qwen 3 0.6b
# model_path = f'experiments/qa_cvdb_tveDefs_nEnts16000_eps5-5-5-5-5-5_bs256-256-256-256-256-256_Qwen3_0.6B_ADAFACTOR_6stage/{stage6_path}'

# # qwen 3 1.7b
# model_path = f'experiments/qa_cvdb_tveDefs_nEnts16000_eps5-5-5-5-5-5_bs256-256-256-256-256-256_Qwen3_1.7B_ADAFACTOR_6stage/{stage6_path}'

# 15 epochs instead of 5 for one of the stages
# model_path = f'experiments/qa_cvdb_tveDefs_nEnts16000_eps5-15-5-5-5-5_bs256-256-256-256-256-256_Llama_3.2_1B_ADAFACTOR_6stage/{stage6_path}'
# model_path = f'experiments/qa_cvdb_tveDefs_nEnts16000_eps5-5-15-5-5-5_bs256-256-256-256-256-256_Llama_3.2_1B_ADAFACTOR_6stage/{stage6_path}'
# model_path = f'experiments/qa_cvdb_tveDefs_nEnts16000_eps5-5-5-15-5-5_bs256-256-256-256-256-256_Llama_3.2_1B_ADAFACTOR_6stage/{stage6_path}'
# model_path = f'experiments/qa_cvdb_tveDefs_nEnts16000_eps5-5-5-5-15-5_bs256-256-256-256-256-256_Llama_3.2_1B_ADAFACTOR_6stage/{stage6_path}'

# %%
# collect_for_model(
#     model_path=model_path,
#     seed=600,
#     prompt_types=(["meaning"]),#"who", "name", "standFor",
#     batch_size=256,
#     keep_every=2,
#     keep_from_layer=8,
# )

# %%
path_load_checkpoints = Path('experiments/mixed-training_qa_cvdb_tveDefs_nEnts16000_eps30_bs256_stage6_s600_ADAFACTOR_single_stage/s600')
checkpoint_dirs = sorted(
    [p for p in path_load_checkpoints.iterdir()
     if p.is_dir() and p.name.startswith("checkpoint-")],
    key=lambda p: int(p.name.split("-")[1])           # numeric sort by step
)
assert checkpoint_dirs

jobs = [(str(p), 600) for p in checkpoint_dirs]
jobs

# %%
seed_one_ep = 600
path_one_epoch = f'experiments/qd1_last_qa_cvdb_tveDefs_nEnts16000_eps1-1-1-1-1-1_bs256-256-256-256-256-256_Llama_3.2_1B_ADAFACTOR_6stage/stage6_s{seed_one_ep}'
jobs = [(path_one_epoch, seed_one_ep)]

# %%
seed = 602
path = f"experiments/qd1_last_qa_cvdb_tveDefs_nEnts16000_eps5-5-5-5-5-5_bs256-256-256-256-256-256_Llama_3.2_1B_ADAFACTOR_naturTrainQs_naturVars_6stage/stage6_s{seed}"
jobs = [(path, seed)]

# %%
seed = 600
path = f'experiments/trainQmult_qa_cvdb_tveDefs_nEnts16000_eps1-1-1-1-1-1_bs256-256-256-256-256-256_Llama_3.2_1B_ADAFACTOR_naturTrainQs_naturVars_trainQsMult5_6stage/stage6_s{seed}'
jobs = [(path, seed)]

# %%
seed = 600
path = f'experiments/Instruct_qa_cvdb_tveDefs_nEnts16000_eps5-5-5-5-5-5_bs256-256-256-256-256-256_Llama_3.2_1B_Instruct_ADAFACTOR_6stage/stage6_s{seed}'
jobs = [(path, seed)]

# %%
seed = 600
path = f'experiments/Rev_step108000_qa_cvdb_tveDefs_nEnts16000_eps5-5-5-5-5-5_bs256-256-256-256-256-256_pythia_1b_deduped_ADAFACTOR_6stage/stage6_s{seed}'
path = f'experiments/Rev_step36000_qa_cvdb_tveDefs_nEnts16000_eps5-5-5-5-5-5_bs256-256-256-256-256-256_pythia_1b_deduped_ADAFACTOR_6stage/stage6_s{seed}'
jobs = [(path, seed)]

# %%
seed = 600
path = f'experiments/qa_cvdb_tveDefs_nEnts16000_eps5-5-5-5-5-5_bs256-256-256-256-256-256_Qwen3_1.7B_ADAFACTOR_6stage/stage6_s{seed}'
jobs = [(path, seed)]

# %%
collect_many(
    jobs=jobs,
    prompt_types=("meaning", "who", "name", "standFor"), # , 
    batch_size=256,
    keep_every=2,
    keep_from_layer=8,
)

# %%
# import numpy as np
# import itertools
# import matplotlib.pyplot as plt

# def list_layers_tokens(perf_npz_path: str):
#     """Quick helper to see what's inside."""
#     d = np.load(perf_npz_path, allow_pickle=True)
#     perf = d["perf"].item()
#     layers = list(perf.keys())
#     tokens_by_layer = {L: sorted(perf[L].keys()) for L in layers}
#     return layers, tokens_by_layer

# def _vector_to_matrix(perf_vec: np.ndarray, n_subsets: int) -> np.ndarray:
#     """Map pairwise perf vector to an n×n symmetric matrix."""
#     m = np.full((n_subsets, n_subsets), np.nan, dtype=np.float32)
#     k_expected = n_subsets * (n_subsets - 1) // 2
#     assert perf_vec.shape[0] == k_expected, f"expected {k_expected} pair entries, got {perf_vec.shape[0]}"
#     for idx, (i, j) in enumerate(itertools.combinations(range(n_subsets), 2)):
#         m[i, j] = perf_vec[idx]
#         m[j, i] = perf_vec[idx]
#     return m

# def visualize_perf_grid(perf_npz_path: str, layer: str, token: int, annotate: bool = True):
#     """
#     Load perf file and show a square grid (subsets × subsets) for the given (layer, token).
#     """
#     d = np.load(perf_npz_path, allow_pickle=True)
#     perf = d["perf"].item()                       # dict[layer][token] -> (num_pairs,)
#     subset_keys = [str(x) for x in d["subset_keys"].tolist()]
#     n = len(subset_keys)

#     perf_vec = np.asarray(perf[layer][token], dtype=np.float32)  # (num_pairs,)
#     grid = _vector_to_matrix(perf_vec, n)

#     fig, ax = plt.subplots(figsize=(0.8*n + 2, 0.8*n + 2))
#     im = ax.imshow(grid, interpolation="nearest", aspect="equal", vmin=0.0, vmax=1.0)
#     plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Probe accuracy")

#     ax.set_xticks(range(n))
#     ax.set_yticks(range(n))
#     ax.set_xticklabels(subset_keys, rotation=45, ha="right")
#     ax.set_yticklabels(subset_keys)

#     ax.set_title(f"Pairwise probe accuracy — layer='{layer}', token={token}")
#     ax.set_xlabel("Subset")
#     ax.set_ylabel("Subset")

#     if annotate:
#         for i in range(n):
#             for j in range(n):
#                 if i == j or np.isnan(grid[i, j]):
#                     txt = "–"
#                 else:
#                     txt = f"{grid[i, j]:.2f}"
#                 ax.text(j, i, txt, ha="center", va="center", fontsize=8)

#     plt.tight_layout()
#     plt.show()


# perf_path = f"{model_path}/probe-perf-meaning-seed600_1.npz"

# # 1) See what's available
# layers, tokens_by_layer = list_layers_tokens(perf_path)
# layers = sorted(layers)  # sort layers for consistent output
# print(layers)
# print("Example layer:", layers[-1])
# print("Tokens for that layer:", tokens_by_layer[layers[0]])

# # 2) Plot a grid for your chosen (layer, token)
# visualize_perf_grid(perf_path, layer=layers[0], token=-1)
# # or explicitly:
# # visualize_perf_grid(perf_path, layer="blocks.12.hook_resid_post", token=-1)


# %%


# %%



