import numpy as np
import itertools
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from utils.linear_probes import train_linear_probe


# TODO consider making COLOR_MAP and _fallback_colours into function arguments / defining them inside _get_colour

# Fixed mapping so a dataset keeps the same colour in every plot
COLOR_MAP = {
    "D1": "tab:blue",
    "D2": "tab:orange",
    "D3": "tab:green",
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

def _hist(ax, data, label, bins=50, **kwargs):
    ax.hist(data, bins=bins, density=True, alpha=0.6,
            label=label, color=_get_colour(label), **kwargs)


def _scatter(ax, x, y, label, **kwargs):
    ax.scatter(x, y, alpha=0.4, s=15, label=label,
               color=_get_colour(label), **kwargs)

# ------------------------------------------------------------------
# Analyses – now axis‑aware & silent by default
# ------------------------------------------------------------------

def perform_pca_analysis(
    acts_train1, acts_train2, acts_project=None,
    *, n_components=2,
    group1_name="Train 1", group2_name="Train 2", project_name="Project",
    title="PCA Projection", ax=None, show=True
):
    """Run PCA using *train1+train2* only; optionally project third set."""

    # Fit on *training* activations only
    train   = np.vstack([acts_train1, acts_train2])
    pca     = PCA(n_components=n_components)
    pca.fit(train)

    proj_t1 = pca.transform(acts_train1)
    proj_t2 = pca.transform(acts_train2)
    proj_pr = pca.transform(acts_project) if acts_project is not None else None
    evr     = pca.explained_variance_ratio_

    # Handle figure / axis management
    created_fig = False
    if ax is None:
        created_fig = True
        fig, ax = plt.subplots(figsize=(10, 8) if n_components == 2 else (8, 5))

    if n_components == 2:
        _scatter(ax, proj_t1[:, 0], proj_t1[:, 1], group1_name)
        _scatter(ax, proj_t2[:, 0], proj_t2[:, 1], group2_name)
        if proj_pr is not None:
            _scatter(ax, proj_pr[:, 0], proj_pr[:, 1], project_name)
        ax.set(
            xlabel=f"PC1 ({evr[0]*100:.2f}% var)",
            ylabel=f"PC2 ({evr[1]*100:.2f}% var)"
        )
    else:  # 1‑D histogram
        _hist(ax, proj_t1[:, 0], group1_name)
        _hist(ax, proj_t2[:, 0], group2_name)
        if proj_pr is not None:
            _hist(ax, proj_pr[:, 0], project_name)
        ax.set(
            xlabel=f"PC1 ({evr[0]*100:.2f}% var)",
            ylabel="Density"
        )

    ax.set(title=title)
    ax.legend(); ax.grid(ls="--", alpha=0.6)
    if n_components == 2:
        ax.axhline(0, lw=.5, c="grey"); ax.axvline(0, lw=.5, c="grey")

    if created_fig and show:
        plt.show()
    elif show:
        # Caller created external fig; defer plt.draw, they'll call plt.show().
        plt.draw()

    return pca, proj_t1, proj_t2, proj_pr, evr


def perform_lda_analysis(
    acts_train1, acts_train2, acts_project=None,
    *, group1_name="Train 1", group2_name="Train 2", project_name="Project",
    title="LDA Projection", ax=None, show=True
):
    """Train LDA and optionally project *acts_project*."""

    train  = np.vstack([acts_train1, acts_train2])
    labels = np.concatenate([
        np.zeros(len(acts_train1), dtype=int),
        np.ones(len(acts_train2),  dtype=int)
    ])

    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(train, labels)

    proj_t1 = lda.transform(acts_train1).ravel()
    proj_t2 = lda.transform(acts_train2).ravel()
    proj_pr = lda.transform(acts_project).ravel() if acts_project is not None else None

    created_fig = False
    if ax is None:
        created_fig = True
        fig, ax = plt.subplots(figsize=(8, 5))

    _hist(ax, proj_t1, group1_name)
    _hist(ax, proj_t2, group2_name)
    if proj_pr is not None:
        _hist(ax, proj_pr, project_name)

    ax.set(title=title, xlabel="LDA Component 1", ylabel="Density")
    ax.legend(); ax.grid(axis="y", ls="--", alpha=0.6)

    if created_fig and show:
        plt.show()
    elif show:
        plt.draw()

    return lda, proj_t1, proj_t2, proj_pr


def perform_lr_projection_analysis(
    acts_train1, acts_train2, acts_project=None,
    *, group1_name="Train 1", group2_name="Train 2", project_name="Project",
    title="LR Projection", num_cross_val=5, ax=None, show=True
):
    """Project activations onto the direction learnt by *train_linear_probe*."""

    # --- Probe training ----------------------------------------------------
    probe_results = train_linear_probe(
        acts_train1, acts_train2, num_cross_val=num_cross_val
    )

    clf = probe_results["trained_classifier"]
    if clf is None:
        # Degenerate case handled inside train_linear_probe
        return probe_results, None, None, None

    direction = clf.coef_.ravel()

    # --- Projection --------------------------------------------------------
    proj_t1 = acts_train1 @ direction
    proj_t2 = acts_train2 @ direction
    proj_pr = acts_project @ direction if acts_project is not None else None

    created_fig = False
    if ax is None:
        created_fig = True
        fig, ax = plt.subplots(figsize=(8, 5))

    _hist(ax, proj_t1, group1_name)
    _hist(ax, proj_t2, group2_name)
    if proj_pr is not None:
        _hist(ax, proj_pr, project_name)

    ax.set(title=title, xlabel="Projection Score ⟨x, w⟩", ylabel="Density")
    ax.legend(); ax.grid(axis="y", ls="--", alpha=0.6)

    if created_fig and show:
        plt.show()
    elif show:
        plt.draw()

    return probe_results, direction, proj_t1, proj_t2, proj_pr

# ------------------------------------------------------------------
# Composite helper – 3‑way train/project sweep
# ------------------------------------------------------------------

def plot_three_way_experiment(
    analysis_type: str,
    names_to_acts: dict,
    *,
    datasets=("D1", "D2", "D3"),
    n_components=2,             # for PCA
    num_cross_val=5,            # for LR
    layer_name="",
    figsize=(18, 5),
):
    """Draw a 1×3 grid cycling over the three possible train/project splits.

    *analysis_type* one of ``'pca' | 'lda' | 'logreg'``.
    """

    assert set(datasets).issubset(names_to_acts.keys()), "Unknown dataset key(s)"
    analysis_type = analysis_type.lower()
    if analysis_type not in {"pca", "lda", "logreg"}:
        raise ValueError("analysis_type must be 'pca', 'lda', or 'logreg'")

    # Map string → callable --------------------------------------------------
    analysis_dispatch = {
        "pca": perform_pca_analysis,
        "lda": perform_lda_analysis,
        "logreg": perform_lr_projection_analysis,
    }
    analysis_fn = analysis_dispatch[analysis_type]

    # List all (train1, train2, project) permutations
    configs = [
        (datasets[0], datasets[1], datasets[2]),
        (datasets[0], datasets[2], datasets[1]),
        (datasets[1], datasets[2], datasets[0]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)

    for ax, (t1_name, t2_name, pr_name) in zip(axes, configs):
        # --- Data ----------------------------------------------------------
        t1 = names_to_acts[t1_name]
        t2 = names_to_acts[t2_name]
        pr = names_to_acts[pr_name]

        s_t1, s_t2, s_pr, _ = standardize_data(t1, t2, pr)

        title = f"{analysis_type.upper()}: {t1_name}/{t2_name} → {pr_name}"

        if analysis_type == "pca":
            analysis_fn(
                s_t1, s_t2, s_pr,
                n_components=n_components,
                group1_name=f"{t1_name} (Train)",
                group2_name=f"{t2_name} (Train)",
                project_name=f"{pr_name} (Proj)",
                title=title,
                ax=ax,
                show=False,
            )
        elif analysis_type == "lda":
            analysis_fn(
                s_t1, s_t2, s_pr,
                group1_name=f"{t1_name} (Train)",
                group2_name=f"{t2_name} (Train)",
                project_name=f"{pr_name} (Proj)",
                title=title,
                ax=ax,
                show=False,
            )
        else:  # logreg
            analysis_fn(
                s_t1, s_t2, s_pr,
                group1_name=f"{t1_name} (Train)",
                group2_name=f"{t2_name} (Train)",
                project_name=f"{pr_name} (Proj)",
                title=title,
                num_cross_val=num_cross_val,
                ax=ax,
                show=False,
            )

    fig.suptitle(f"{analysis_type.upper()} – Three‑Way Experiment @ {layer_name}", fontsize=14)
    fig.tight_layout()
    plt.show()

    return fig  # for further tweaking / saving

# # ------------------------------------------------------------------
# # Example workflow (assuming *names_to_acts* is prepared elsewhere)
# # ------------------------------------------------------------------
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
#     plot_three_way_experiment("pca", names_to_acts, layer_name=layer)
#     plot_three_way_experiment("lda", names_to_acts, layer_name=layer)
#     plot_three_way_experiment("logreg", names_to_acts, layer_name=layer, num_cross_val=5)
