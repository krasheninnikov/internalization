# %% [markdown]
# Unified plotting with shared helpers
# - Single figure example
# - 2x2 grid with shared x but per-subplot y (w1 shared, w2 per-subplot residual PCA)
# - 2x2 grid with fully shared W (both x & y shared)

# %%
import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh, norm
from pathlib import Path
import re

# ---------- helpers kept intact ----------
def pick_last_layer_and_token(bundle):
    """Extract centroids from last layer/token."""
    C = bundle["centroids"]
    layers = bundle["layer_names"].tolist()
    tokens = bundle["token_labels"].tolist()
    li, ti = len(layers) - 1, len(tokens) - 1
    X = C[:, li, ti, :]
    names = bundle["dataset_names"].tolist() if "dataset_names" in bundle else [f"D{i+1}" for i in range(X.shape[0])]
    prompt = str(bundle["prompt_type"].item()) if "prompt_type" in bundle else "?"
    seed = int(bundle["seed"]) if "seed" in bundle else -1
    return X, names, prompt, seed, layers[li], tokens[ti]

def latexify_D_labels(names, use_mathrm=False):
    """Convert 'D1' to '$D_{1}$' for LaTeX rendering."""
    D = r"\mathrm{D}" if use_mathrm else "D"
    def repl_token(m):
        return rf"${D}_{{{m.group(1)}}}$"
    out = []
    for s in names:
        m = re.fullmatch(r"\s*D[_\s]?(\d+)\s*", s)
        if m:
            out.append(repl_token(m))
        else:
            s2 = re.sub(r"(?<!\w)D[_\s]?(\d+)(?!\w)", repl_token, s)
            out.append(s2)
    return out

# ============= Data Loading =============
def load_runs(paths):
    """Load centroid arrays from NPZ files, dropping NaNs."""
    runs = []
    for path in paths:
        Z = np.load(path, allow_pickle=True)
        X, names, *_ = pick_last_layer_and_token(Z)
        mask = ~np.isnan(X).any(axis=1)
        runs.append(X[mask].astype(np.float64))
    return runs

def load_runs_with_meta(paths):
    """Load centroid arrays with metadata for plotting."""
    runs, meta = [], []
    for path in paths:
        Z = np.load(path, allow_pickle=True)
        X, names, prompt, seed, layer, tok = pick_last_layer_and_token(Z)
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask].astype(np.float64)
        names = [n for n, m in zip(names, mask) if m]
        runs.append(X)
        meta.append((names, prompt, seed, layer, tok))
    return runs, meta

# ============= Projection Computation =============
def compute_w1(runs_x):
    """Average endpoint direction across runs."""
    vec = np.zeros(runs_x[0].shape[1], dtype=np.float64)
    for X in runs_x:
        vec += (X[-1] - X[0])
    return vec / (norm(vec) + 1e-12)

def compute_w2_residual(runs_y, w1):
    """Top PC in subspace orthogonal to w1."""
    C = np.vstack(runs_y)
    resid = C - (C @ w1)[:, None] * w1[None, :]
    cov = np.cov(resid, rowvar=False)
    _, eigvecs = eigh(cov)
    w2 = eigvecs[:, -1]
    w2 = w2 - (w2 @ w1) * w1  # ensure orthogonal
    return w2 / (norm(w2) + 1e-12)

def compute_projection_matrix(paths_for_x_axis, paths_for_y_axis=None, scale_by_std=False):
    """
    Build projection matrix W = [w1, w2] from paths.
    Returns: W, scaler
    """
    runs_x = load_runs(paths_for_x_axis)
    runs_y = load_runs(paths_for_y_axis) if paths_for_y_axis else runs_x

    scaler = None
    if scale_by_std:
        all_data = np.vstack(runs_x + runs_y)
        scaler = np.std(all_data, axis=0)
        scaler = np.where(scaler == 0, 1.0, scaler)
        runs_x = [X / scaler for X in runs_x]
        runs_y = [X / scaler for X in runs_y]

    w1 = compute_w1(runs_x)
    w2 = compute_w2_residual(runs_y, w1)
    return np.column_stack([w1, w2]), scaler

# ============= Plotting =============
def plot_centroids_on_ax(
    ax,
    paths_to_plot,
    *,
    # Projection options (in order of precedence):
    W=None,                      # Option 1: Full projection matrix [d,2]
    w1=None,                     # Option 2: Just x-axis (computes w2 from data)
    paths_for_x_axis=None,       # Option 3: Paths to compute w1 (and w2)
    paths_for_y_axis=None,       # Paths for w2 computation (defaults to paths_to_plot)
    # Scaling:
    scaler=None,                 # Pre-computed std vector
    scale_by_std=False,          # Whether to compute scaling (ignored if scaler provided)
    # Visualization:
    legend_labels=None,
    xlabel=None, ylabel=None, title=None,
    connect_runs=True,
    text_x_offset=0.0, text_y_offset=0.2,
    palette=None,                # Custom color mapping
    markersize=50,
    xlim=None, ylim=None,
    pad_frac=0.10
):
    """
    Plot centroid trajectories on given axis.
    Projection precedence:
      1) W provided; 2) w1 provided (compute w2); 3) compute both from paths_for_x_axis (+ paths_for_y_axis).
    Returns: W used
    """
    runs_plot, meta_plot = load_runs_with_meta(paths_to_plot)

    # Determine projection matrix
    if W is not None:
        W_used = W
        if scaler is not None:
            runs_plot = [X / scaler for X in runs_plot]

    elif w1 is not None:
        runs_y = load_runs(paths_for_y_axis) if paths_for_y_axis else runs_plot
        if scaler is not None:
            runs_plot = [X / scaler for X in runs_plot]
            runs_y = [X / scaler for X in runs_y] if paths_for_y_axis else runs_plot
        elif scale_by_std:
            pool = runs_plot + (runs_y if paths_for_y_axis else [])
            all_data = np.vstack(pool)
            scaler = np.std(all_data, axis=0)
            scaler = np.where(scaler == 0, 1.0, scaler)
            runs_plot = [X / scaler for X in runs_plot]
            runs_y = [X / scaler for X in runs_y]
        w2 = compute_w2_residual(runs_y, w1)
        W_used = np.column_stack([w1, w2])

    else:
        if paths_for_x_axis is None:
            raise ValueError("Must provide either W, w1, or paths_for_x_axis")
        runs_x = load_runs(paths_for_x_axis)
        runs_y = load_runs(paths_for_y_axis) if paths_for_y_axis else runs_plot
        if scaler is not None:
            runs_plot = [X / scaler for X in runs_plot]
            runs_x = [X / scaler for X in runs_x]
            runs_y = [X / scaler for X in runs_y] if paths_for_y_axis else runs_plot
        elif scale_by_std:
            pool = runs_plot + runs_x + (runs_y if paths_for_y_axis else [])
            all_data = np.vstack(pool)
            scaler = np.std(all_data, axis=0)
            scaler = np.where(scaler == 0, 1.0, scaler)
            runs_plot = [X / scaler for X in runs_plot]
            runs_x = [X / scaler for X in runs_x]
            runs_y = [X / scaler for X in runs_y]
        w1 = compute_w1(runs_x)
        w2 = compute_w2_residual(runs_y, w1)
        W_used = np.column_stack([w1, w2])

    # Colors & markers
    if palette is None:
        all_names = []
        for names, *_ in meta_plot:
            for n in names:
                if n not in all_names:
                    all_names.append(n)
        palette = {n: f"C{i % 10}" for i, n in enumerate(all_names)}

    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "*"]

    # Plot runs
    all_projected = []
    for run_idx, (X, meta) in enumerate(zip(runs_plot, meta_plot)):
        names, prompt, seed, layer, tok = meta
        pts = X @ W_used
        all_projected.append(pts)
        marker = markers[run_idx % len(markers)]
        for (x, y), name in zip(pts, names):
            ax.scatter(x, y, s=markersize, marker=marker, color=palette[name],
                       edgecolors="black", linewidths=0.5, alpha=0.95)
        if connect_runs and len(pts) >= 2:
            ax.plot(pts[:, 0], pts[:, 1], lw=1.0, alpha=0.85, color="0.35")
        if run_idx == 0:
            for (x, y), name, latex_name in zip(pts, names, latexify_D_labels(names)):
                ax.text(x + text_x_offset, y + text_y_offset, latex_name,
                        fontsize=8, weight="bold", color=palette[name])

    # Axes
    ax.axhline(0, lw=.5, c="grey")
    ax.axvline(0, lw=.5, c="grey")
    ax.grid(ls="--", alpha=.3)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)
    if title: ax.set_title(title, fontsize=10, pad=5)

    # Legend
    if legend_labels:
        handles = []
        for run_idx in range(len(meta_plot)):
            marker = markers[run_idx % len(markers)]
            label = legend_labels[run_idx] if run_idx < len(legend_labels) else f"Run {run_idx}"
            h = ax.scatter([], [], marker=marker, color="gray", edgecolors="black", label=label)
            handles.append(h)
        ax.legend(handles=handles, fontsize=7, frameon=True, fancybox=True,
                  framealpha=0.9, loc='best', markerscale=0.8)

    # Limits
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is None or ylim is None:
        all_pts = np.vstack(all_projected)
        if xlim is None:
            xmin, xmax = all_pts[:, 0].min(), all_pts[:, 0].max()
            x_pad = pad_frac * (xmax - xmin)
            ax.set_xlim(xmin - x_pad, xmax + x_pad)
        if ylim is None:
            ymin, ymax = all_pts[:, 1].min(), all_pts[:, 1].max()
            y_pad = pad_frac * (ymax - ymin)
            ax.set_ylim(ymin - y_pad, ymax + y_pad)

    return W_used

def plot_centroids(paths_to_plot, paths_for_x_axis=None, paths_for_y_axis=None,
                   figsize=(8.4, 4.7), save_path=None, **kwargs):
    """Convenience wrapper that creates a figure and uses plot_centroids_on_ax."""
    fig, ax = plt.subplots(figsize=figsize)
    kwargs.setdefault('xlabel', 'Avg endpoint difference')
    kwargs.setdefault('ylabel', 'PC-1 (residual PCA)')
    kwargs.setdefault('markersize', 72)  # Larger for standalone

    # Default title from metadata (if not provided)
    if 'title' not in kwargs:
        _, meta = load_runs_with_meta(paths_to_plot[:1])
        if meta:
            layer, tok = meta[0][3], meta[0][4]
            kwargs['title'] = f"Centroid trajectories\nlast token '{tok}' @ {layer}"

    W = plot_centroids_on_ax(ax, paths_to_plot,
                             paths_for_x_axis=paths_for_x_axis,
                             paths_for_y_axis=paths_for_y_axis,
                             **kwargs)

    # Standalone legend placement
    if ax.get_legend():
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
                  frameon=False, handletextpad=0.1, markerscale=2.0)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", format="pdf")
        print(f"Saved to {save_path}")
    plt.show()
    return fig, ax, W

# %% [markdown]
# === Single-figure example ===

# %%
legend_labels = None

# ---- X-axis definition (Option A: different prompts same seed=600) ----
seed = 600
base_path_A = f'experiments/qd1_last_qa_cvdb_tveDefs_nEnts16000_eps5and5and5and5and5and5_bs256and256and256and256and256and256_Llama_3.2_1B_ADAFACTOR_6stage/stage6_s{seed}/'
paths_for_x_A = [
    base_path_A + f"activation-centroids-and-percentiles-who-seed{seed}.npz",
    base_path_A + f"activation-centroids-and-percentiles-standFor-seed{seed}.npz",
    base_path_A + f"activation-centroids-and-percentiles-name-seed{seed}.npz",
    base_path_A + f"activation-centroids-and-percentiles-meaning-seed{seed}.npz",
]

# ---- X-axis definition (Option B: different prompts and seeds 602..605) ----
base_path_B = "experiments/qa_cvdb_tveDefs_nEnts16000_eps5-5-5-5-5-5_bs256-256-256-256-256-256_Llama_3.2_1B_ADAFACTOR_6stage"
paths_for_x_B = [
    f"{base_path_B}/stage6_s602/activation-centroids-and-percentiles-who-seed602.npz",
    f"{base_path_B}/stage6_s603/activation-centroids-and-percentiles-standFor-seed603.npz",
    f"{base_path_B}/stage6_s604/activation-centroids-and-percentiles-name-seed604.npz",
    f"{base_path_B}/stage6_s605/activation-centroids-and-percentiles-mean-seed605_1.npz",
]

# ---- Natural vars (defined even when not used) ----
base_path_natural_vars = 'experiments/qd1_last_qa_cvdb_tveDefs_nEnts16000_eps5and5and5and5and5and5_bs256and256and256and256and256and256_Llama_3.2_1B_ADAFACTOR_naturTrainQs_naturVars_6stage/stage6_s600/activation-centroids-and-percentiles'
paths_natural_vars = [f"{base_path_natural_vars}-{p}-seed{seed}.npz" for p in ["who", "standFor", "name", "meaning"]]

# Mimic your later overrides (final choice = natural vars)
paths_for_x = paths_for_x_B
paths_for_x = paths_for_x + paths_natural_vars
paths_for_x = paths_natural_vars  # final

# ---- Single-figure 'BASE diff prompts' (same composition as before) ----
paths_to_plot = paths_for_x_B + [paths_natural_vars[0]]  # add one extra for illustration
legend_labels = ["Who (602)", "Stand for (603)", "Name (604)", "Meaning (605)", "Who (600, natural vars)"]

print(paths_to_plot)

fig, ax, W_single = plot_centroids(
    paths_to_plot=paths_to_plot,
    paths_for_x_axis=paths_for_x,
    figsize=(10.4, 4.7),
    save_path="plots/simplified_centroids.pdf",
    legend_labels=legend_labels
)

# %% [markdown]
# === 2x2 grid — shared x, NOT shared y (i.e., shared w1; per-subplot does residual PCA for w2) ===

# %%
# Common x-axis paths (same as earlier; you can switch to paths_for_x_A if desired)
paths_for_x_grid = [base_path_A + f"activation-centroids-and-percentiles-{p}-seed{seed}.npz"
                    for p in ["who", "standFor", "name", "meaning"]]
paths_for_x_grid = paths_natural_vars

# Precompute shared w1
runs_x_grid = load_runs(paths_for_x_grid)
w1_shared = compute_w1(runs_x_grid)

fig, axes = plt.subplots(2, 2, figsize=(16, 8))

# ---- (a) Sequential Stages ----
prompt_type = "who"
ax = axes[0, 0]
seed_seq = 605
all_stage_paths = [f'stage{stage}_s{seed_seq}' for stage in range(1, 7)]
paths_subplot1 = [
    f'experiments/qa_cvdb_tveDefs_nEnts16000_eps5-5-5-5-5-5_bs256-256-256-256-256-256_Llama_3.2_1B_ADAFACTOR_6stage/{sp}/activation-centroids-and-percentiles-{prompt_type}-seed{seed_seq}.npz'
    for sp in all_stage_paths
][::-1]

## 600-variant
# seed_seq = 600
# all_stage_paths = [f'stage{stage}_s{seed_seq}' for stage in range(1, 7)]
# paths_subplot1 = [
#     f"experiments/qd1_last_qa_cvdb_tveDefs_nEnts16000_eps5and5and5and5and5and5_bs256and256and256and256and256and256_Llama_3.2_1B_ADAFACTOR_6stage/{sp}/activation-centroids-and-percentiles-{prompt_type}-seed{seed_seq}.npz"
#     for sp in all_stage_paths
# ][::-1]

legend_labels_1 = [f"Stage {i+1}" for i in range(len(all_stage_paths))][::-1]
plot_centroids_on_ax(
    ax, paths_subplot1,
    w1=w1_shared,  # shared x
    title="(a) Sequential Stages",
    xlabel="Avg endpoint difference",
    ylabel="PC-1 (residual)",
    legend_labels=legend_labels_1
)

# ---- (b) Re-exposure ----
## 605-variant
ax = axes[0, 1]
seed_reexp = 605
path_orig = f"experiments/qa_cvdb_tveDefs_nEnts16000_eps5-5-5-5-5-5_bs256-256-256-256-256-256_Llama_3.2_1B_ADAFACTOR_6stage/stage6_s{seed_reexp}/activation-centroids-and-percentiles-{prompt_type}-seed{seed_reexp}.npz"
path_rexs = [
    f"experiments/re-expose-stage{i}_qa_cvdb_tveDefs_nEnts16000_eps5_bs256_stage6_s{seed_reexp}_ADAFACTOR_single_stage/s{seed_reexp}/activation-centroids-and-percentiles-{prompt_type}-seed{seed_reexp}.npz"
    for i in range(1, 6)
]
# ## 600-variant
# seed_reexp = 600
# path_orig = f"experiments/qd1_last_qa_cvdb_tveDefs_nEnts16000_eps5and5and5and5and5and5_bs256and256and256and256and256and256_Llama_3.2_1B_ADAFACTOR_6stage/stage6_s600/activation-centroids-and-percentiles-{prompt_type}-seed600.npz"
# path_rexs = [
#     f"experiments/re-expose-stage{i}_qa_cvdb_tveDefs_nEnts16000_eps5_bs256_stage6_s{seed_reexp}_ADAFACTOR_single_stage/s{seed_reexp}/activation-centroids-and-percentiles-{prompt_type}-seed{seed_reexp}.npz"
#     for i in range(1, 6)
# ]

paths_subplot2 = [path_orig] + path_rexs
legend_labels_2 = ["Original"] + [f"Re-exp {i}" for i in range(1, 6)]
plot_centroids_on_ax(
    ax, paths_subplot2,
    w1=w1_shared,
    title="(b) Re-exposure to Earlier Stages",
    xlabel="Avg endpoint difference",
    ylabel="PC-1 (residual)",
    legend_labels=legend_labels_2
)

# ---- (c) Extra Epochs ----
ax = axes[1, 0]
prompt_type = "who"
path_orig = f"experiments/qd1_last_qa_cvdb_tveDefs_nEnts16000_eps5and5and5and5and5and5_bs256and256and256and256and256and256_Llama_3.2_1B_ADAFACTOR_6stage/stage6_s600/activation-centroids-and-percentiles-{prompt_type}-seed600.npz"
paths_extra_eps = [
    f'experiments/qa_cvdb_tveDefs_nEnts16000_eps5-15-5-5-5-5_bs256-256-256-256-256-256_Llama_3.2_1B_ADAFACTOR_6stage/stage6_s600/activation-centroids-and-percentiles-{prompt_type}-seed600.npz',
    f'experiments/qa_cvdb_tveDefs_nEnts16000_eps5-5-15-5-5-5_bs256-256-256-256-256-256_Llama_3.2_1B_ADAFACTOR_6stage/stage6_s600/activation-centroids-and-percentiles-{prompt_type}-seed600.npz',
    f'experiments/qa_cvdb_tveDefs_nEnts16000_eps5-5-5-15-5-5_bs256-256-256-256-256-256_Llama_3.2_1B_ADAFACTOR_6stage/stage6_s600/activation-centroids-and-percentiles-{prompt_type}-seed600.npz',
    f'experiments/qa_cvdb_tveDefs_nEnts16000_eps5-5-5-5-15-5_bs256-256-256-256-256-256_Llama_3.2_1B_ADAFACTOR_6stage/stage6_s600/activation-centroids-and-percentiles-{prompt_type}-seed600.npz'
]
paths_subplot3 = [path_orig] + paths_extra_eps
legend_labels_3 = ["Original"] + [f"15ep stage {i}" for i in range(2, 6)]
plot_centroids_on_ax(
    ax, paths_subplot3,
    w1=w1_shared,
    title="(c) Extra Epochs Mid-Training",
    xlabel="Avg endpoint difference",
    ylabel="PC-1 (residual)",
    legend_labels=legend_labels_3
)

# ---- (d) Mixed Training Checkpoints ----
ax = axes[1, 1]
prompt_one = "who"
def checkpoint_step(p: Path) -> int:
    m = re.search(r"checkpoint-(\d+)", p.name)
    return int(m.group(1)) if m else -1

ckpt_files = []
path_load_checkpoints = Path('experiments/mixed-training_qa_cvdb_tveDefs_nEnts16000_eps30_bs256_stage6_s600_ADAFACTOR_single_stage/s600')
if path_load_checkpoints.exists():
    checkpoint_dirs = sorted(
        [p for p in path_load_checkpoints.iterdir()
         if p.is_dir() and p.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[1])
    )
    for d in checkpoint_dirs:
        f = d / f"activation-centroids-and-percentiles-{prompt_one}-seed{seed}.npz"
        if f.exists():
            ckpt_files.append((checkpoint_step(d), str(f)))

ckpt_files.sort(key=lambda t: t[0])
paths_subplot4 = [f'experiments/qd1_last_qa_cvdb_tveDefs_nEnts16000_eps5and5and5and5and5and5_bs256and256and256and256and256and256_Llama_3.2_1B_ADAFACTOR_6stage/stage6_s600/activation-centroids-and-percentiles-{prompt_one}-seed600.npz']
paths_subplot4.extend([fpath for step, fpath in ckpt_files[::2]])
legend_labels_4 = ["Original"] + [f"{i}ep mixed" for i in list(range(2, 31, 2))[::2]]

plot_centroids_on_ax(
    ax, paths_subplot4,
    w1=w1_shared,
    title="(d) Mixed Training Checkpoints",
    xlabel="Avg endpoint difference",
    ylabel="PC-1 (residual)",
    legend_labels=legend_labels_4
)

fig.suptitle("Centroid Evolution Under Different Training Regimes", fontsize=14, y=1.02)
plt.tight_layout()
Path("plots").mkdir(parents=True, exist_ok=True)
plt.savefig("plots/centroid_grid_comparison.pdf", bbox_inches="tight", dpi=150)
plt.show()

# %% [markdown]
# === 2x2 grid — fully SHARED projection (shared x & y, i.e., shared W) ===

# %%
# Gather ALL paths across all subplots (same as before)
all_subplot_paths = []

# (a) sequential stages (use 605 variant we plotted)
all_subplot_paths.extend(paths_subplot1)

# (b) re-exposure
all_subplot_paths.extend(paths_subplot2)

# (c) extra epochs
all_subplot_paths.extend(paths_subplot3)

# (d) mixed training checkpoints
all_subplot_paths.extend(paths_subplot4)

# Compute a shared W using shared x-paths (paths_for_x_grid) and all subplot paths as y
print(f"Computing shared projection from {len(all_subplot_paths)} total paths")
W_shared, scaler_shared = compute_projection_matrix(
    paths_for_x_axis=paths_for_x_grid,
    paths_for_y_axis=all_subplot_paths,
    scale_by_std=False
)

# Make the grid with W_shared
fig, axes = plt.subplots(2, 2, figsize=(16, 8))

# (a)
ax = axes[0, 0]
plot_centroids_on_ax(
    ax, paths_subplot1,
    W=W_shared, scaler=scaler_shared,
    title="(a) Sequential Stages",
    xlabel="Avg endpoint difference",
    ylabel="PC-1 (all data)",
    legend_labels=legend_labels_1
)

# (b)
ax = axes[0, 1]
plot_centroids_on_ax(
    ax, paths_subplot2,
    W=W_shared, scaler=scaler_shared,
    title="(b) Re-exposure to Earlier Stages",
    xlabel="Avg endpoint difference",
    ylabel="PC-1 (all data)",
    legend_labels=legend_labels_2
)

# (c)
ax = axes[1, 0]
plot_centroids_on_ax(
    ax, paths_subplot3,
    W=W_shared, scaler=scaler_shared,
    title="(c) Extra Epochs Mid-Training",
    xlabel="Avg endpoint difference",
    ylabel="PC-1 (all data)",
    legend_labels=legend_labels_3
)

# (d)
ax = axes[1, 1]
plot_centroids_on_ax(
    ax, paths_subplot4,
    W=W_shared, scaler=scaler_shared,
    title="(d) Mixed Training Checkpoints",
    xlabel="Avg endpoint difference",
    ylabel="PC-1 (all data)",
    legend_labels=legend_labels_4
)

fig.suptitle("Centroid Evolution Under Different Training Regimes (Shared Projection)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("plots/centroid_grid_shared_projection.pdf", bbox_inches="tight", dpi=150)
plt.show()

# %%
def is_orthonormal(W):
    return np.allclose(W.T @ W, np.eye(W.shape[1])) and np.allclose(np.linalg.norm(W, axis=0), 1)

print("W_shared is orthonormal:", is_orthonormal(W_shared))

# %%
