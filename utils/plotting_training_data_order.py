from __future__ import annotations
import itertools
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import MDS
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from utils.linear_probes import train_linear_probe

# TODO consider making COLOR_MAP and _fallback_colours into function arguments / defining them inside _get_colour

# Fixed mapping so a dataset keeps the same colour in every plot
COLOR_MAP = {
    "D1": "tab:blue",
    "D2": "tab:orange",
    "D3": "tab:green",
    "D4": "tab:red",
    "D5": "tab:purple",
    "D6": "tab:brown",
}

# If an unknown key shows up we cycle through the default Tableau palette
_fallback_colours = itertools.cycle(
    plt.rcParams["axes.prop_cycle"].by_key()["color"]
)


def _get_colour(label: str, colour_map=COLOR_MAP):
    """Return deterministic colour for a dataset.

    We assume the dataset name is the **first** whitespace‑separated part
    of *label* (e.g. ``'D1 (Train)'`` → ``'D1'``)."""
    dataset_key = label.split()[0]
    return colour_map.get(dataset_key, next(_fallback_colours))

# ------------------------------------------------------------------
# Pre‑processing helpers
# ------------------------------------------------------------------

def standardize_data(train_acts1, train_acts2, project_acts=None):
    """Z‑score features using statistics from *training* activations only."""
    scaler = StandardScaler()
    scaler.fit(np.vstack([train_acts1, train_acts2]))

    s_train1 = scaler.transform(train_acts1)
    s_train2 = scaler.transform(train_acts2)
    s_proj   = scaler.transform(project_acts) if project_acts is not None else None
    return s_train1, s_train2, s_proj, scaler

# ------------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------------

def _latex_labels(label: str) -> str:
    """Convert D+integer patterns to LaTeX format (e.g., 'D1' → '$D_1$')."""
    return re.sub(r'\bD(\d+)\b', r'$D_{\1}$', label)

def _hist(ax, data, label, bins=50, **kwargs):
    ax.hist(data, bins=bins, density=True, alpha=0.6,
            label=_latex_labels(label), color=_get_colour(label), **kwargs)


def _scatter(ax, x, y, label, **kwargs):
    ax.scatter(x, y, alpha=0.4, s=15, label=_latex_labels(label),
               color=_get_colour(label), **kwargs)

# ------------------------------------------------------------------
# Analyses – now axis‑aware & silent by default
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 1. Pair‑&‑project configuration helper
# ------------------------------------------------------------------

def _make_configs(
    datasets: list[str] | tuple[str, ...],
    *,
    pair_mode: str = "consecutive",  # "consecutive" | "all"
    custom_pairs: list[tuple[str, str]] | None = None,
):
    """Return a list of (train1, train2, [proj_names ...]) tuples.

    *pair_mode* is ignored when *custom_pairs* is provided.
    """
    ds = list(datasets)
    if len(ds) < 2:
        raise ValueError("Need at least two datasets to train a probe.")

    if custom_pairs is not None:
        pair_list = custom_pairs
    elif pair_mode == "all":
        pair_list = list(itertools.combinations(ds, 2))
    else:  # default is consecutive
        pair_list = [(ds[i], ds[i + 1]) for i in range(len(ds) - 1)]

    configs: list[tuple[str, str, list[str]]] = []
    for t1, t2 in pair_list:
        projects = [d for d in ds if d not in (t1, t2)]
        configs.append((t1, t2, projects))
    return configs

# ------------------------------------------------------------------
# 2. Analysis helpers upgraded to multi‑project aware variants
# ------------------------------------------------------------------

# ——— PCA ————————————————————————————————————————————————————————

def perform_pca_analysis(
    acts_train1: np.ndarray,
    acts_train2: np.ndarray,
    *,
    acts_projects: dict[str, np.ndarray] | None = None,
    n_components: int = 2,
    group1_name: str = "Train 1",
    group2_name: str = "Train 2",
    title: str = "PCA Projection",
    ax=None,
    show: bool = True,
):
    """Project *train1*, *train2* and any number of *acts_projects* sets with PCA.

    Returns (pca, proj_train1, proj_train2, proj_dict, evr).
    """
    acts_projects = acts_projects or {}

    # Fit on *training* activations only
    train = np.vstack([acts_train1, acts_train2])
    pca = PCA(n_components=max(10, n_components))
    pca.fit(train)

    proj_t1 = pca.transform(acts_train1)[:, :n_components]
    proj_t2 = pca.transform(acts_train2)[:, :n_components]
    proj_dict = {
        name: pca.transform(act)[:, :n_components] for name, act in acts_projects.items()
    }
    evr = pca.explained_variance_ratio_[:n_components]

    created_fig = False
    if ax is None:
        created_fig = True
        fig, ax = plt.subplots(figsize=(10, 8) if n_components == 2 else (8, 5))

    # ---------- plotting ------------------------------------------
    if n_components == 2:
        _scatter(ax, proj_t1[:, 0], proj_t1[:, 1], group1_name)
        _scatter(ax, proj_t2[:, 0], proj_t2[:, 1], group2_name)
        for name, proj in proj_dict.items():
            _scatter(ax, proj[:, 0], proj[:, 1], f"{name} (Proj)")
        ax.set(
            xlabel=f"PC1 ({evr[0]*100:.2f}% var)",
            ylabel=f"PC2 ({evr[1]*100:.2f}% var)",
        )
    else:
        _hist(ax, proj_t1[:, 0], group1_name)
        _hist(ax, proj_t2[:, 0], group2_name)
        for name, proj in proj_dict.items():
            _hist(ax, proj[:, 0], f"{name} (Proj)")
        ax.set(xlabel=f"PC1 ({evr[0]*100:.2f}% var)", ylabel="Density")

    ax.set(title=title)
    ax.legend(); ax.grid(ls="--", alpha=0.6)
    if n_components == 2:
        ax.axhline(0, lw=.5, c="grey"); ax.axvline(0, lw=.5, c="grey")

    if created_fig and show:
        plt.show()
    elif show:
        plt.draw()

    return pca, proj_t1, proj_t2, proj_dict, evr

# ——— LDA ————————————————————————————————————————————————————————

def perform_lda_analysis(
    acts_train1: np.ndarray,
    acts_train2: np.ndarray,
    *,
    acts_projects: dict[str, np.ndarray] | None = None,
    group1_name: str = "Train 1",
    group2_name: str = "Train 2",
    title: str = "LDA Projection",
    num_cross_val: int = 1,
    plot_train_mean_lines: bool = False,
    plot_proj_mean_lines: bool = False,
    ax=None,
    show: bool = True,
):
    """1‑D LDA histogram with any number of projected datasets."""

    res = train_linear_probe(
        acts_train1,
        acts_train2,
        num_cross_val=num_cross_val,
        probe_type="lda",
        lda_solver="eigen",
    )

    lda = res["trained_classifier"]
    if lda is None:
        return res, None, None, {}

    proj_t1 = lda.transform(acts_train1).ravel()
    proj_t2 = lda.transform(acts_train2).ravel()
    acts_projects = acts_projects or {}
    proj_dict = {name: lda.transform(act).ravel() for name, act in acts_projects.items()}

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(8, 5))

    # --------------- histograms -----------------------------------------
    _hist(ax, proj_t1, group1_name)
    _hist(ax, proj_t2, group2_name)
    for name, proj in proj_dict.items():
        _hist(ax, proj, f"{name} (Proj)")

    # --------------- optional mean markers -----------------------------
    if plot_train_mean_lines:
        m1, m2 = proj_t1.mean(), proj_t2.mean()
        ax.axvline(m1, color=_get_colour(group1_name), ls="--", lw=2)
        ax.axvline(m2, color=_get_colour(group2_name), ls="--", lw=2)

    if plot_proj_mean_lines:
        for name, proj in proj_dict.items():
            mu = proj.mean()
            ax.axvline(mu, color=_get_colour(name), ls="--", lw=2)

    ax.set(title=title, xlabel="LDA Component 1", ylabel="Density")
    ax.legend(); ax.grid(axis="y", ls="--", alpha=0.6)

    if created and show:
        plt.show()
    elif show:
        plt.draw()

    return res, proj_t1, proj_t2, proj_dict

# ——— Logistic‑Regression projection ———————————————————————————

def perform_lr_projection_analysis(
    acts_train1: np.ndarray,
    acts_train2: np.ndarray,
    *,
    acts_projects: dict[str, np.ndarray] | None = None,
    group1_name: str = "Train 1",
    group2_name: str = "Train 2",
    title: str = "LR Projection",
    num_cross_val: int = 5,
    plot_train_mean_lines: bool = False,
    plot_proj_mean_lines: bool = False,
    ax=None,
    show: bool = True,
):
    """Project onto LR probe axis; plot histograms + optional mean markers."""

    probe_results = train_linear_probe(
        acts_train1, acts_train2, num_cross_val=num_cross_val
    )
    clf = probe_results["trained_classifier"]
    if clf is None:
        return probe_results, None, None, {}

    direction = clf.coef_.ravel()
    proj_t1 = acts_train1 @ direction
    proj_t2 = acts_train2 @ direction
    acts_projects = acts_projects or {}
    proj_dict = {name: acts @ direction for name, acts in acts_projects.items()}

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(8, 5))

    _hist(ax, proj_t1, group1_name)
    _hist(ax, proj_t2, group2_name)
    for name, proj in proj_dict.items():
        _hist(ax, proj, f"{name} (Proj)")

    # Optional mean lines
    if plot_train_mean_lines:
        m1, m2 = proj_t1.mean(), proj_t2.mean()
        ax.axvline(m1, color=_get_colour(group1_name), ls="--", lw=2)
        ax.axvline(m2, color=_get_colour(group2_name), ls="--", lw=2)

    if plot_proj_mean_lines:
        for name, proj in proj_dict.items():
            mu = proj.mean()
            ax.axvline(mu, color=_get_colour(name), ls="--", lw=2)

    ax.set(title=title, xlabel="Projection Score ⟨x, w⟩", ylabel="Density")
    ax.legend(); ax.grid(axis="y", ls="--", alpha=0.6)

    if created and show:
        plt.show()
    elif show:
        plt.draw()

    return probe_results, direction, proj_t1, proj_t2, proj_dict

# ——— Difference‑of‑Means projection —————————————————————————

def perform_diffmean_analysis(
    acts_train1: np.ndarray,
    acts_train2: np.ndarray,
    *,
    acts_projects: Dict[str, np.ndarray] | None = None,
    group1_name: str = "Train 1",
    group2_name: str = "Train 2",
    title: str = "Difference of Means Projection",
    rescale_projections: bool = True,
    plot_train_mean_lines: bool = False,
    plot_proj_mean_lines: bool = False,
    ax=None,
    show: bool = True,
):
    """Project activations on the diff‑of‑means axis; optional mean markers."""

    acts_projects = acts_projects or {}

    # --- Standardisation
    scaler = StandardScaler().fit(np.vstack([acts_train1, acts_train2]))
    s_train1 = scaler.transform(acts_train1)
    s_train2 = scaler.transform(acts_train2)
    s_projects = {name: scaler.transform(act) for name, act in acts_projects.items()}

    mean_s1, mean_s2 = s_train1.mean(0), s_train2.mean(0)
    diff_vec = mean_s1 - mean_s2
    norm = np.linalg.norm(diff_vec)
    if np.isclose(norm, 0):
        raise RuntimeError("Means of the two training sets are identical; cannot define direction.")

    w = diff_vec / norm
    proj_t1 = s_train1 @ w
    proj_t2 = s_train2 @ w
    proj_dict = {name: s @ w for name, s in s_projects.items()}

    # Optional scaling so means map to –1/+1
    scaling_params = None
    if rescale_projections:
        mu1, mu2 = proj_t1.mean(), proj_t2.mean()
        denom = mu2 - mu1
        if not np.isclose(denom, 0):
            a = 2.0 / denom
            b = -1.0 - a * mu1
            proj_t1 = a * proj_t1 + b
            proj_t2 = a * proj_t2 + b
            proj_dict = {k: a * v + b for k, v in proj_dict.items()}
            scaling_params = {"a": a, "b": b}

    # --- Plotting
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(8, 5))

    _hist(ax, proj_t1, group1_name)
    _hist(ax, proj_t2, group2_name)
    for name, proj in proj_dict.items():
        _hist(ax, proj, f"{name} (Proj)")

    # --- Mean lines --------------------------------------------------------
    if plot_train_mean_lines:
        ax.axvline(proj_t1.mean(), color=_get_colour(group1_name), ls="--", lw=2)
        ax.axvline(proj_t2.mean(), color=_get_colour(group2_name), ls="--", lw=2)

    if plot_proj_mean_lines:
        for name, proj in proj_dict.items():
            mu = proj.mean()
            ax.axvline(mu, color=_get_colour(name), ls="--", lw=2)

    ax.set(title=title, xlabel="Scaled Projection Score" if scaling_params else "Projection Score", ylabel="Density")
    ax.legend(); ax.grid(axis="y", ls="--", alpha=0.6)

    if created and show:
        plt.show()
    elif show:
        plt.draw()

    return w, proj_t1, proj_t2, proj_dict, scaling_params

# ------------------------------------------------------------------
# 3. Generalised plotting wrapper – any² → many
# ------------------------------------------------------------------

def plot_pairwise_experiment(
    analysis_type: str,
    names_to_acts: Dict[str, np.ndarray],
    *,
    datasets: List[str] | Tuple[str, ...] | None = None,
    pair_mode: str = "consecutive",        # "consecutive" | "all"
    custom_pairs: List[Tuple[str, str]] | None = None,
    n_components: int = 2,
    num_cross_val: int = 5,
    rescale_projections: bool = True,
    layer_name: str = "",
    figsize: Tuple[int, int] = (18, 5),
    save_pdf: bool = True,
    plot_train_mean_lines: bool = True,
    plot_proj_mean_lines: bool = True,
):
    """Plot every requested train‑pair with all remaining datasets projected.

    analysis_type ∈ {"pca", "lda", "logreg", "diffmean"}.
    """

    # Validate & prepare dataset list
    datasets = list(datasets) if datasets is not None else list(names_to_acts.keys())
    if not set(datasets).issubset(names_to_acts.keys()):
        raise ValueError("datasets contains unknown keys")

    analysis_type = analysis_type.lower()
    analysis_dispatch = {
        "pca": perform_pca_analysis,
        "lda": perform_lda_analysis,
        "logreg": perform_lr_projection_analysis,
        "diffmean": perform_diffmean_analysis,
    }
    if analysis_type not in analysis_dispatch:
        raise ValueError("analysis_type must be one of 'pca', 'lda', 'logreg', 'diffmean'")
    analysis_fn = analysis_dispatch[analysis_type]

    configs = _make_configs(datasets, pair_mode=pair_mode, custom_pairs=custom_pairs)
    n_panels = len(configs)
    n_cols = min(3, n_panels)
    n_rows = math.ceil(n_panels / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for ax, (t1_name, t2_name, proj_names) in zip(axes, configs):
        t1 = names_to_acts[t1_name]
        t2 = names_to_acts[t2_name]
        acts_projects = {name: names_to_acts[name] for name in proj_names}

        title = (
            f"Train: {_latex_labels(t1_name)}/{_latex_labels(t2_name)}, "
            f"project: {', '.join(map(_latex_labels, proj_names)) or '—'} "
            f"({analysis_type})"
        )

        common = dict(
            acts_train1=t1,
            acts_train2=t2,
            acts_projects=acts_projects,
            group1_name=f"{t1_name}",
            group2_name=f"{t2_name}",
            # group1_name=f"{t1_name} (Train)",
            # group2_name=f"{t2_name} (Train)",
            ax=ax,
            title=title,
            show=False,
            plot_train_mean_lines=plot_train_mean_lines,
            plot_proj_mean_lines=plot_proj_mean_lines,
        )

        if analysis_type == "pca":
            del common["plot_train_mean_lines"]
            del common["plot_proj_mean_lines"]
            analysis_fn(**common, n_components=n_components)
        elif analysis_type == "lda":
            analysis_fn(**common, num_cross_val=num_cross_val)
        elif analysis_type == "logreg":
            analysis_fn(**common, num_cross_val=num_cross_val)
        elif analysis_type == "diffmean":
            analysis_fn(**common, rescale_projections=rescale_projections)

    # Remove unused axes when n_panels < len(axes)
    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle(
        f"{analysis_type.upper()} – Pairwise Experiment @ {layer_name}", fontsize=14
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    if save_pdf:
        plots_dir = Path("plots"); plots_dir.mkdir(exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_layer = layer_name.replace("/", "_").replace(".", "_")
        fname = plots_dir / f"{analysis_type}_{safe_layer}_{date_str}.pdf"
        fig.savefig(fname, format="pdf", bbox_inches="tight", dpi=300)
        print(f"Saved plot to: {fname}")

    return fig


# # # -----------------------------------------------------------------------------------------
# # Example workflow for the above functions (assuming *names_to_acts* is prepared elsewhere)
# # -----------------------------------------------------------------------------------------
# if __name__ == "__main__":
#     # *names_to_acts* expected to hold the 3 datasets' activations
#     # e.g. names_to_acts = {"D1": acts1, "D2": acts2, "D3": acts3}

#     layer_name        = "blocks.12.hook_resid_post"
#     last_token_offset = 2  # index from end (e.g. ‑2 for last ‑ 1)
    
#     # layer_name = 'blocks.9.hook_resid_post'
#     # last_token_offset = 6

#     last_token_index = acts_data1_filtered[layer_name].shape[1] - last_token_offset
#     names_to_acts = {
#             "D1": acts_data1_filtered[layer_name][:, last_token_index, :],
#             "D2": acts_data2_filtered[layer_name][:, last_token_index, :],
#             "D3": acts_data3_filtered[layer_name][:, last_token_index, :]
#         }

#     layer = "blocks.12.hook_resid_post"
#     plot_pairwise_experiment("pca", names_to_acts, layer_name=layer)
#     plot_pairwise_experiment("lda", names_to_acts, layer_name=layer)
#     plot_pairwise_experiment("logreg", names_to_acts, layer_name=layer, num_cross_val=5)


# ────────────────────────── train_stage_pair_probes ──────────────────────────

def train_stage_pair_probes(
    acts_all,
    layer_name: str,
    token_idx: int,
    *,
    probe_type: str = "logreg",
    pairs="all",                         #  "consecutive" | "all" | list[tuple[int,int]]
    num_cv: int = 5,
    # LDA parameters
    lda_shrinkage: str | float | None = "auto",
    lda_solver: str = "lsqr",
    # Logistic Regression parameters
    penalty: str = "l2",
    C: float = 1.0,
    max_iter: int = 1000,
    solver: str = "lbfgs",
    # Normalisation and sign alignment
    normalise: bool = False,
    align_sign: bool = True,             # flip so w • (μ_j – μ_i) > 0
):
    """
    Train a binary probe for each requested (stage_i , stage_j) pair
    and stack the resulting directions.
    
    Parameters
    ----------
    acts_all : list[dict[str, np.ndarray]]
        Each element is {layer_name : activations (N × T × d)} for one stage.
        We assume these are in the order of finetuning stages.
    
    Returns
    -------
    W     : (N_pairs, d)   probe directions
    pairs : list[(i,j)]    stage indices for each row of W
    perf  : list[float]    mean CV accuracy of that probe
    """

    # --------------------- decide which pairs to do ------------------------
    n_stages = len(acts_all)
    if pairs == "consecutive":
        pair_list = [(i, i + 1) for i in range(n_stages - 1)]
    elif pairs == "all":
        pair_list = list(itertools.combinations(range(n_stages), 2))
    else:
        pair_list = list(pairs)

    directions, out_pairs, perf = [], [], []

    # ----------------------- loop over pairs ------------------------------
    for i, j in pair_list:
        a_i = acts_all[i][layer_name][:, token_idx, :]
        a_j = acts_all[j][layer_name][:, token_idx, :]

        if probe_type == "logreg":
            res = train_linear_probe(
                a_i,
                a_j,
                num_cross_val=num_cv,
                penalty=penalty,
                C=C,
                max_iter=max_iter,
                solver=solver,
            )
        elif probe_type == "lda":
            res = train_linear_probe(
                a_i,
                a_j,
                num_cross_val=num_cv,
                probe_type="lda",
                lda_shrinkage=lda_shrinkage,
                lda_solver=lda_solver,
            )
        else:
            raise ValueError("probe_type must be 'logreg' or 'lda'")

        clf = res["trained_classifier"]
        if clf is None:
            continue

        w = getattr(clf, "coef_", None)
        if w is not None:
            w = w.ravel()
        else:
            w = clf.scalings_.T.ravel()

        # --- optional sign alignment -------------------------------------
        # --- ensures that the mean of the later stage projects to a larger value 
        # --- on the axis defined by w than the mean of the earlier stage
        if align_sign and (a_j.mean(0) - a_i.mean(0)) @ w < 0:
            w = -w
        if normalise:
            w /= np.linalg.norm(w) + 1e-12

        directions.append(w)
        out_pairs.append((i, j))
        if res["cv_scores"] is not None:
            perf.append(float(np.mean(res["cv_scores"])))
        else:
            perf.append(np.nan)

    if not directions:
        raise RuntimeError("No probes were trained successfully.")

    W = np.vstack(directions)
    return W, out_pairs, perf
