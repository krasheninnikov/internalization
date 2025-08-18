# -----------------------------------------------------------------------------
# Save + load activation centroids (and optional percentiles) to .npz bundles.
# Keeps your existing schema:
#   centroids:       (K, L, T, d)
#   percentiles:     (K, L, T, d, Q)  [omitted if you pass percentiles=()]
#   percentiles_q:   (Q,)
#   layer_names:     list[str]
#   token_indices:   (T,)
#   token_labels:    list[str]
#   dataset_names:   list[str]
#   dataset_sizes:   (K,)
#   prompt_type:     str
#   model_path:      str
#   seed:            int
#   d:               int
#   version:         int
#
# Input acts format (per dataset):
#   dict[str, np.ndarray] with arrays shaped (N, T, d)
# -----------------------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union, List
import numpy as np

Array = np.ndarray
ActsDict = Mapping[str, Array]  # layer_name -> (N, T, d)


# -----------------------------------------------------------------------------
# filename helper (underscores, no dot before the counter)
# -----------------------------------------------------------------------------
def unique_path(out_dir: Union[str, Path], stem: str, *, suffix: str = ".npz") -> Path:
    """
    Return a non-existing path like: dir/stem.npz, dir/stem_1.npz, dir/stem_2.npz, ...
    Also cleans the stem (no slashes; collapse whitespace).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean = stem.strip().replace("/", "_").replace("\\", "_")
    clean = "_".join(clean.split()) or "activation_summary"

    base = out_dir / f"{clean}{suffix}"
    if not base.exists():
        return base

    n = 1
    while True:
        candidate = out_dir / f"{clean}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


# -----------------------------------------------------------------------------
# token window helper
# -----------------------------------------------------------------------------
def infer_token_window_from_sample(
    *,
    model,
    sample_text: str,
    expected_tokens: int,
) -> Tuple[List[int], List[str]]:
    """
    Pick positions so the tokenized length matches the activations' T.
    Drops BOS/EOS when present. Returns (token_indices, token_labels).
    """
    ids_bos   = model.to_tokens(sample_text, prepend_bos=True )[0].tolist()
    ids_nobos = model.to_tokens(sample_text, prepend_bos=False)[0].tolist()

    if len(ids_bos) == expected_tokens:
        token_ids = ids_bos
        has_bos = True
    elif len(ids_nobos) == expected_tokens:
        token_ids = ids_nobos
        has_bos = False
    else:
        raise ValueError(
            f"Expected T={expected_tokens}, got len(with_bos)={len(ids_bos)} "
            f"and len(no_bos)={len(ids_nobos)}."
        )

    bos_id = getattr(model.tokenizer, "bos_token_id", None)
    eos_id = getattr(model.tokenizer, "eos_token_id", None)
    start = 1 if (has_bos and token_ids and token_ids[0] == bos_id) else 0
    end   = len(token_ids) - (1 if (token_ids and token_ids[-1] == eos_id) else 0)

    indices = list(range(start, end))
    toks    = model.tokenizer.convert_ids_to_tokens(token_ids)
    labels  = [t.replace("Ġ", " ").replace("Ċ", "\\n") for t in toks[start:end]]
    return indices, labels


# -----------------------------------------------------------------------------
# public API
# -----------------------------------------------------------------------------
def save_activation_summary_npz(
    datasets: Sequence[Optional[ActsDict]],
    *,
    out_dir: Union[str, Path],
    basename: str,
    seed: int,
    prompt_type: str,
    model_path: str,
    # token window: either give (token_indices + token_labels) OR (model + sample_text)
    token_indices: Optional[Sequence[int]] = None,
    token_labels: Optional[Sequence[str]] = None,
    model=None,
    sample_text: Optional[str] = None,
    # names & percentiles
    dataset_names: Optional[Sequence[str]] = None,
    percentiles: Sequence[float] = (0.05, 0.50, 0.95),
    dtype: str = "float32",
    version: int = 1,
) -> Path:
    """
    Save per-dataset × per-layer × per-token centroids (and optional percentiles).
    Returns the saved Path.
    """
    # Discover layout from the first non-None dataset
    first = next(a for a in datasets if a is not None)
    layer_names: List[str] = list(first.keys())
    any_layer = layer_names[0]
    _, T_expected, d = first[any_layer].shape

    # Token window
    if token_indices is None or token_labels is None:
        if model is None or sample_text is None:
            raise ValueError("Provide either (token_indices & token_labels) or (model & sample_text).")
        token_indices, token_labels = infer_token_window_from_sample(
            model=model, sample_text=sample_text, expected_tokens=T_expected
        )
    token_indices = list(token_indices)
    token_labels  = list(token_labels)
    T = len(token_indices)

    # Dataset meta
    K = len(datasets)
    if dataset_names is None:
        dataset_names = [f"D{i+1}" for i in range(K)]
    dataset_names = list(dataset_names)

    qs = np.array(percentiles, dtype=np.float32)
    Q  = int(len(qs))

    # Allocate
    centroids = np.full((K, len(layer_names), T, d), np.nan, dtype=dtype)
    percentiles_arr = np.full((K, len(layer_names), T, d, Q), np.nan, dtype=dtype) if Q else None
    dataset_sizes = np.zeros(K, dtype=np.int32)

    # Compute stats
    for k, acts_dict in enumerate(datasets):
        if acts_dict is None:
            continue
        for li, layer in enumerate(layer_names):
            A = acts_dict[layer][:, token_indices, :]  # (N, T, d)
            dataset_sizes[k] = A.shape[0]              # same N across layers
            centroids[k, li] = np.nanmean(A, axis=0).astype(dtype)
            if Q:
                qs_vals = np.nanquantile(A, qs, axis=0)           # (Q, T, d)
                percentiles_arr[k, li] = np.moveaxis(qs_vals, 0, -1).astype(dtype)  # → (T, d, Q)

    # Save
    outfile = unique_path(out_dir, basename, suffix=".npz")
    save_items = {
        "centroids": centroids,
        "percentiles_q": qs,
        "layer_names": np.array(layer_names, dtype=object),
        "token_indices": np.array(token_indices, dtype=int),
        "token_labels": np.array(token_labels, dtype=object),
        "dataset_names": np.array(dataset_names, dtype=object),
        "dataset_sizes": dataset_sizes,
        "prompt_type": np.array(prompt_type, dtype=object),
        "model_path": np.array(model_path, dtype=object),
        "seed": np.int32(seed),
        "d": np.int32(d),
        "version": np.int32(version),
    }
    if percentiles_arr is not None:
        save_items["percentiles"] = percentiles_arr

    np.savez_compressed(outfile, **save_items)
    print(f"[✓] saved → {outfile}")
    return outfile


def load_activation_summary_npz(path: Union[str, Path]) -> "np.lib.npyio.NpzFile":
    """Thin wrapper so list-like metadata loads cleanly with allow_pickle."""
    return np.load(str(path), allow_pickle=True)
