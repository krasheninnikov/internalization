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
paths_natural_vars_s600 = [f"{base_path_natural_vars}-{p}-seed600.npz" for p in ["who", "standFor", "name", "meaning"]]

base_path_natural_vars_s601 = 'experiments/qd1_last_qa_cvdb_tveDefs_nEnts16000_eps5-5-5-5-5-5_bs256-256-256-256-256-256_Llama_3.2_1B_ADAFACTOR_naturTrainQs_naturVars_6stage/stage6_s601/activation-centroids-and-percentiles'
paths_natural_vars_s601 = [f"{base_path_natural_vars_s601}-{p}-seed601.npz" for p in ["who", "standFor", "name", "meaning"]]

# overrides
paths_for_x = paths_for_x_B
# paths_for_x = paths_for_x + paths_natural_vars_s600 + paths_natural_vars_s601
paths_for_x = paths_natural_vars_s600 +   paths_natural_vars_s601 + paths_for_x_A

# ---- Single-figure 'BASE diff prompts' (same composition as before) ----
paths_to_plot = paths_for_x_B + [paths_natural_vars_s600[0], paths_natural_vars_s601[1], 
                                #  paths_natural_vars_s600[2], paths_natural_vars_s601[3]
                                 ]
legend_labels = ["Who (602)", "Stand for (603)", "Name (604)", "Meaning (605)", 
                 "Who (600, natural)", "StandFor (601, natural)",
                #  "Name (600, natural)", "Meaning (601, natural)"
                 ]

print(paths_to_plot)

fig, ax, W_single = plot_centroids(
    paths_to_plot=paths_to_plot,
    paths_for_x_axis=paths_for_x,
    figsize=(10.4, 4.7),
    save_path="plots/simplified_centroids.pdf",
    legend_labels=legend_labels,
    text_x_offset=0.0, text_y_offset=-0.6,
)

# %%
# %%  KDE — (1) Collect activations only (no projection)
from typing import NamedTuple
import numpy as np
from pathlib import Path

class ActsBundle(NamedTuple):
    subsets: list[np.ndarray]   # each (N_k, d)
    names: list[str]
    prompt: str
    seed: int
    layer_name: str
    tok_label: str
    d: int

def collect_activations_for_npz(
    npz_path: str,
    *,
    token_idx: int = -1,        # last token (matches your plots)
    batch_size: int = 256,
    keep_every: int = 1,
    keep_from_layer: int | None = None,
) -> ActsBundle:
    """Rebuild data/model for this NPZ and return raw per-subset activations (no scaling, no projection)."""
    Z = np.load(npz_path, allow_pickle=True)
    centroid_matrix, names, prompt, seed, layer_name, tok_label = pick_last_layer_and_token(Z)
    print(f"Loaded {npz_path}: prompt={prompt}, seed={seed}, layer={layer_name}, token='{tok_label}', names={names}")

    # --- rebuild data & model --------------------------
    model_dir = Path(npz_path).parent.as_posix()
    config_folder = Path(model_dir).parent.as_posix() if 'checkpoint' in model_dir else Path(model_dir).parent.as_posix()

    from data_generation.load_data_from_config import generate_data_from_experiment_folder
    data, params_used, cfg = generate_data_from_experiment_folder(
        folder_path=config_folder,
        seed=int(seed),
        seed_stage2=0,
        train_subset="full",
    )
    natural_style_vars = bool(getattr(params_used, "natural_style_vars", False)
                              if hasattr(params_used, "__dict__") else params_used.get("natural_style_vars", False))

    prompt_dataset_str = {'mean': 'meaning', "stand for": "standFor", "Who": "who"}.get(prompt, prompt) # TODO alternative get this from npz str
    prefix = f"ent_assoc_{prompt_dataset_str}_"
    print(f"Prefix: {prefix}")
    order = ["qd1consis", "qd1incons", "qd2consis", "qd2incons", "qd4consis", "q"]
    data_keys = list(data.keys())
    picked = [k for k in data_keys if k.startswith(prefix)]
    subset_keys = [prefix + o for o in order if (prefix + o) in picked]
    raw_groups = [data[k]['question'] for k in subset_keys]  # NOTE using the question subset here
    print(f'Data keys: {data_keys}')
    print(f"Used subset keys: {subset_keys}")
    print(f"Lengths of each subset: {[len(g) for g in raw_groups]}")

    # --- model + filtering to mirror centroid pipeline ---------------------
    from utils.linear_probes import load_model_to_transformerlens, leave_unique_q_type, get_activations_and_logit_stats
    base_model_name = getattr(getattr(cfg, "model_arguments", None), "model_name_or_path", None)
    model = load_model_to_transformerlens(model_dir, base_model_name)

    q = {"standFor": "stand for", "who": "Who", "meaning": "mean"}.get(prompt, prompt)
    var_len = 5 if natural_style_vars else 3
    groups = [leave_unique_q_type(texts, model, q, var_len) if texts else [] for texts in raw_groups]
    n = min((len(g) for g in groups if len(g) > 0), default=0)
    groups = [g[:n] for g in groups]

    # --- collect activations per subset (slice layer/token) ----------------
    subsets = []
    d = centroid_matrix.shape[-1]
    for texts in groups:
        if not texts:
            raise ValueError("Empty data subset.")
        acts, _, _ = get_activations_and_logit_stats(
            model,
            texts,
            batch_size=batch_size,
            keep_every=keep_every,
            keep_from_layer=keep_from_layer,
        )
        arr = acts[layer_name][:, token_idx, :]  # (N, d)
        subsets.append(arr)

    assert len(names) == len(subsets), f"Mismatch in number of subsets: npz has {len(names)} but collected {len(subsets)}"
    return ActsBundle(subsets=subsets, names=names, prompt=prompt, seed=seed,
                      layer_name=layer_name, tok_label=tok_label, d=d)

# --- Example: collect once and keep in memory --------------------------------
acts_cache = {}
run_idx = 2
npz_path = paths_to_plot[run_idx]
acts_bundle = collect_activations_for_npz(
    npz_path,
    token_idx=-1,
    batch_size=256,
    keep_every=2,
    keep_from_layer=8,
)
acts_cache[npz_path] = acts_bundle
print(f"Collected activations for {npz_path} → "
      f"{[a.shape for a in acts_bundle.subsets]} (d={acts_bundle.d})")

# %%  KDE — (2) Project + plot (reuse any W/scaler; fast iteration)
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt

def project_activations_to_2d(
    subsets: list[np.ndarray],
    W: np.ndarray,                # [d, 2]
    scaler: np.ndarray | None = None,  # divide-only std vector used for centroids (or None)
):
    """Return (list of (N_k,2) arrays, (K,2) centroids2d)."""
    proj = []
    if scaler is not None:
        safe = np.where(scaler == 0, 1.0, scaler)
    for arr in subsets:
        if arr is None or len(arr) == 0:
            proj.append(np.zeros((0, 2), dtype=np.float64))
            continue
        A = (arr / safe) if scaler is not None else arr
        proj.append(A @ W)
    cents2d = np.vstack([p.mean(axis=0) if len(p) else np.zeros(2) for p in proj])
    return proj, cents2d

def overlay_kde_contours(
    ax: plt.Axes,
    Z2d_subsets: list[np.ndarray],
    names: list[str],
    palette: dict[str, str],
    *,
    ref_points: np.ndarray | None = None,
    mass: float = 0.68,
    grid_n: int = 220,
    pad_frac: float = 0.12,
    alpha: float = 0.6,
    linewidth: float = 1.6,
    draw_centroids: bool = False,
    centroids2d: np.ndarray | None = None,
    centroid_markersize: float = 56.0,
):
    """Draw iso-mass KDE contours for each subset cloud in Z2d_subsets."""
    # initial bounds
    if ref_points is not None and len(ref_points):
        xmin, xmax = ref_points[:, 0].min(), ref_points[:, 0].max()
        ymin, ymax = ref_points[:, 1].min(), ref_points[:, 1].max()
    else:
        xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
    dx, dy = (xmax - xmin), (ymax - ymin)
    xmin -= pad_frac * (dx + 1e-12); xmax += pad_frac * (dx + 1e-12)
    ymin -= pad_frac * (dy + 1e-12); ymax += pad_frac * (dy + 1e-12)

    def isomass_threshold(dens2d: np.ndarray, p: float) -> float:
        flat = dens2d.ravel()
        order = np.argsort(flat)[::-1]
        cdf = np.cumsum(flat[order]); cdf /= cdf[-1] if cdf[-1] > 0 else 1.0
        return flat[order][np.searchsorted(cdf, p)]

    # expand grid if a contour touches borders
    for _ in range(3):
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, grid_n), np.linspace(ymin, ymax, grid_n))
        grid = np.vstack([xx.ravel(), yy.ravel()])
        touches = {'L': False, 'R': False, 'B': False, 'T': False}
        dens_list, thr_list = [], []

        for Z in Z2d_subsets:
            if Z is None or len(Z) == 0:
                dens_list.append(None); thr_list.append(None); continue
            kde = gaussian_kde(Z.T, bw_method="scott")
            dens = kde(grid).reshape(xx.shape); dens_list.append(dens)
            thr = isomass_threshold(dens, mass); thr_list.append(thr)
            if dens[:,  0].max() >= thr: touches['L'] = True
            if dens[:, -1].max() >= thr: touches['R'] = True
            if dens[ 0, :].max() >= thr: touches['B'] = True
            if dens[-1, :].max() >= thr: touches['T'] = True

        if any(touches.values()):
            if touches['L']: xmin -= 0.20 * (xmax - xmin)
            if touches['R']: xmax += 0.20 * (xmax - xmin)
            if touches['B']: ymin -= 0.20 * (ymax - ymin)
            if touches['T']: ymax += 0.20 * (ymax - ymin)
            continue
        break

    xsegs, ysegs = [], []
    for lab, dens, thr in zip(names, dens_list, thr_list):
        if dens is None or thr is None: continue
        cs = ax.contour(xx, yy, dens, levels=[thr],
                        colors=[palette.get(lab, "k")], linewidths=[linewidth],
                        zorder=2, alpha=alpha)
        if cs.allsegs and cs.allsegs[0]:
            for seg in cs.allsegs[0]:
                xsegs.append(seg[:, 0]); ysegs.append(seg[:, 1])

    if draw_centroids and centroids2d is not None and len(centroids2d):
        for (x, y), lab in zip(centroids2d, names):
            ax.scatter(x, y, s=centroid_markersize, facecolors="none",
                       edgecolors=palette.get(lab, "k"), linewidths=1.8, zorder=3)

    # pad limits to include contours + reference
    cxmin, cxmax = ax.get_xlim(); cymin, cymax = ax.get_ylim()
    xmin_f, xmax_f, ymin_f, ymax_f = cxmin, cxmax, cymin, cymax
    if ref_points is not None and len(ref_points):
        xmin_f = min(xmin_f, ref_points[:, 0].min()); xmax_f = max(xmax_f, ref_points[:, 0].max())
        ymin_f = min(ymin_f, ref_points[:, 1].min()); ymax_f = max(ymax_f, ref_points[:, 1].max())
    if xsegs:
        xs = np.concatenate(xsegs); ys = np.concatenate(ysegs)
        xmin_f = min(xmin_f, xs.min()); xmax_f = max(xmax_f, xs.max())
        ymin_f = min(ymin_f, ys.min()); ymax_f = max(ymax_f, ys.max())
    pad_x = 0.05 * (xmax_f - xmin_f + 1e-12); pad_y = 0.05 * (ymax_f - ymin_f + 1e-12)
    ax.set_xlim(xmin_f - pad_x, xmax_f + pad_x)
    ax.set_ylim(ymin_f - pad_y, ymax_f + pad_y)

# --- Build / reuse a projection & draw scatter -----------------------------
W_single, scaler_single = compute_projection_matrix(
    paths_for_x_axis=paths_for_x,     # whatever you used for x-axis
    paths_for_y_axis=paths_to_plot,   # include plotted runs
    scale_by_std=False
)

fig, ax = plt.subplots(figsize=(10.4, 4.7))
plot_centroids_on_ax(
    ax, paths_to_plot,
    W=W_single, scaler=scaler_single,
    legend_labels=legend_labels,
    xlabel='Avg endpoint difference',
    ylabel='PC-1 (residual PCA)'
)

# palette + bounds consistent with scatter
_, meta = load_runs_with_meta(paths_to_plot)
all_names = []
for names, *_ in meta:
    for n in names:
        if n not in all_names: all_names.append(n)
palette = {n: f"C{i % 10}" for i, n in enumerate(all_names)}
ref_points = np.vstack([X @ W_single for X in load_runs(paths_to_plot)])

# --- Project previously collected activations and overlay KDE --------------
npz_path = paths_to_plot[run_idx]
acts_bundle = acts_cache[npz_path]   # fast: no recollection
Z2d_subsets, cents2d = project_activations_to_2d(
    acts_bundle.subsets, W_single, scaler=scaler_single
)

overlay_kde_contours(
    ax, Z2d_subsets, acts_bundle.names, palette,
    ref_points=ref_points,
    mass=0.68,
    grid_n=220,
    draw_centroids=True,
    centroids2d=cents2d
)

plt.tight_layout()
plt.show()

# --- Tip: re-run just from here after changing W/scaler to iterate quickly.


# %% [markdown]
# === 2x2 grid — shared x, NOT shared y (i.e., shared w1; per-subplot does residual PCA for w2) ===

# %%
# Common x-axis paths (same as earlier; you can switch to paths_for_x_A if desired)
paths_for_x_grid = [base_path_A + f"activation-centroids-and-percentiles-{p}-seed{seed}.npz"
                    for p in ["who", "standFor", "name", "meaning"]]
paths_for_x_grid = paths_natural_vars_s600

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
seed_ckpt = 600
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
    print(f"Found {len(checkpoint_dirs)} checkpoint dirs, e.g. {checkpoint_dirs[0]}")
    for d in checkpoint_dirs:
        f = d / f"activation-centroids-and-percentiles-{prompt_one}-seed{seed_ckpt}.npz"
        if f.exists():
            ckpt_files.append((checkpoint_step(d), str(f)))
        else:
            print(f"File not found: {f}")

ckpt_files.sort(key=lambda t: t[0])
print(f"Found {len(ckpt_files)} checkpoint files")
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
