# %% [markdown]
# # 3D Interactive Plot of Centroids
# - Loads centroids from NPZ files (no raw activations needed)
# - Projects onto first 3 principal components of the centroids themselves
# - Interactive 3D plot with plotly (full rotation/zoom/pan)

# %%
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from numpy.linalg import eigh, norm

# %% [markdown]
# ## Helper Functions

# %%
def normalize_rows(X, eps=1e-12):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)

def pick_last_layer_and_token(bundle):
    """Extract centroids from last layer/token."""
    C = bundle["centroids"]
    # Shape: (n_datasets, n_layers, n_tokens, d)
    n_datasets, n_layers, n_tokens, d = C.shape

    # Use last layer and last token
    li, ti = n_layers - 1, n_tokens - 1
    X = C[:, li, ti, :]

    # Try to load metadata, but handle numpy version mismatch gracefully
    try:
        names = bundle["dataset_names"].tolist()
    except Exception:
        names = [f"D{i+1}" for i in range(n_datasets)]

    try:
        prompt = str(bundle["prompt_type"].item())
    except Exception:
        prompt = "?"

    try:
        seed = int(bundle["seed"])
    except Exception:
        seed = -1

    try:
        layer_name = bundle["layer_names"].tolist()[li]
    except Exception:
        layer_name = f"layer_{li}"

    try:
        tok_label = bundle["token_labels"].tolist()[ti]
    except Exception:
        tok_label = f"tok_{ti}"

    return X, names, prompt, seed, layer_name, tok_label

def load_centroids_with_meta(path):
    """Load centroids with metadata from a single NPZ file."""
    Z = np.load(path, allow_pickle=True)
    X, names, prompt, seed, layer, tok = pick_last_layer_and_token(Z)
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask].astype(np.float64)
    X = normalize_rows(X)
    names = [n for n, m in zip(names, mask) if m]
    return X, names, prompt, seed, layer, tok

def compute_pca_3d(centroids, center=False):
    """Compute top 3 principal components from centroids."""
    if center:
        mean = centroids.mean(axis=0)
        data = centroids - mean
    else:
        mean = np.zeros(centroids.shape[1])
        data = centroids

    # Covariance and eigen decomposition
    cov = np.cov(data, rowvar=False)
    eigenvalues, eigenvectors = eigh(cov)

    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Top 3 components
    W = eigenvectors[:, :3]

    # Explained variance
    total_var = eigenvalues.sum()
    explained = eigenvalues[:3] / total_var * 100

    return W, mean, explained

# %% [markdown]
# ## Main Plotting Function

# %%
# Plotly color palette (similar to matplotlib's C0-C9)
PLOTLY_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

# Plotly marker symbols for different prompts
PLOTLY_MARKERS = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'star']

# Line styles (dash patterns) for different prompts
PLOTLY_LINE_DASHES = ['solid', 'dash', 'dot', 'dashdot']

# Prompt types available
PROMPT_TYPES = ['who', 'standFor', 'name', 'meaning']

def plot_centroids_3d_multi_prompt(base_path, seed, title=None, prompts=None):
    """
    Plot centroids in 3D using PCA on the centroids themselves,
    with all 4 prompts shown on the same plot.

    Args:
        base_path: Base path to NPZ files (without prompt-seed suffix)
        seed: Seed value used in filenames
        title: Optional plot title
        prompts: List of prompts to plot (defaults to all 4)
    """
    if prompts is None:
        prompts = PROMPT_TYPES

    # Build paths for all prompts
    paths = [base_path + f"{prompt}-seed{seed}.npz" for prompt in prompts]

    # Load all centroids
    all_runs = []
    all_meta = []
    valid_prompts = []
    for path, prompt in zip(paths, prompts):
        try:
            X, names, _, seed_loaded, layer, tok = load_centroids_with_meta(path)
            all_runs.append(X)
            all_meta.append((names, prompt, seed_loaded, layer, tok))
            valid_prompts.append(prompt)
        except FileNotFoundError:
            print(f"Warning: File not found: {path}")
            continue

    if not all_runs:
        raise ValueError("No valid centroid files found!")

    # Pool centroids for PCA (use all prompts)
    pooled = np.vstack(all_runs)
    W, mean, explained = compute_pca_3d(pooled)

    print(f"PCA explained variance: PC1={explained[0]:.1f}%, PC2={explained[1]:.1f}%, PC3={explained[2]:.1f}%")
    print(f"Total explained: {sum(explained):.1f}%")

    # Color palette based on dataset names (D1, D2, etc.)
    all_names = []
    for names, *_ in all_meta:
        for n in names:
            if n not in all_names:
                all_names.append(n)
    palette = {n: PLOTLY_COLORS[i % len(PLOTLY_COLORS)] for i, n in enumerate(all_names)}
    print(f"DEBUG palette: {palette}")

    fig = go.Figure()

    # Plot each prompt as a separate "run" with different marker/line style
    for prompt_idx, (X, meta) in enumerate(zip(all_runs, all_meta)):
        names, prompt, _, layer, tok = meta

        # Project onto PCA space
        centered = X - mean
        pts = centered @ W

        marker_symbol = PLOTLY_MARKERS[prompt_idx % len(PLOTLY_MARKERS)]
        line_dash = PLOTLY_LINE_DASHES[prompt_idx % len(PLOTLY_LINE_DASHES)]

        # Get colors for each point based on dataset name
        colors = [palette[name] for name in names]

        # Add individual scatter points for each centroid (colored by dataset)
        for i, (pt, name, color) in enumerate(zip(pts, names, colors)):
            fig.add_trace(go.Scatter3d(
                x=[pt[0]], y=[pt[1]], z=[pt[2]],
                mode='markers+text',
                marker=dict(
                    size=8,
                    color=color,
                    symbol=marker_symbol,
                    line=dict(width=1, color='black')
                ),
                text=name if prompt_idx == 0 else '',  # Only label first prompt
                textposition='top center',
                textfont=dict(size=10, color=color),
                name=f'{name} ({prompt})',
                showlegend=False,  # Don't show individual points in legend
            ))

        # Connect points with lines (using segments colored by dataset)
        for i in range(len(pts) - 1):
            fig.add_trace(go.Scatter3d(
                x=pts[i:i+2, 0], y=pts[i:i+2, 1], z=pts[i:i+2, 2],
                mode='lines',
                line=dict(width=4, color=colors[i], dash=line_dash),
                showlegend=False,
            ))

    # Add dummy traces for legend (one per prompt type showing marker style)
    for prompt_idx, prompt in enumerate(valid_prompts):
        marker_symbol = PLOTLY_MARKERS[prompt_idx % len(PLOTLY_MARKERS)]
        line_dash = PLOTLY_LINE_DASHES[prompt_idx % len(PLOTLY_LINE_DASHES)]
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers+lines',
            marker=dict(size=8, color='gray', symbol=marker_symbol),
            line=dict(width=4, color='gray', dash=line_dash),
            name=f'{prompt}',
            showlegend=True,
        ))

    # Layout
    if title is None and all_meta:
        layer, tok = all_meta[0][3], all_meta[0][4]
        title = f"Centroid trajectories (3D PCA) - All prompts<br>last token '{tok}' @ {layer}"

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        scene=dict(
            xaxis_title=f'PC1 ({explained[0]:.1f}%)',
            yaxis_title=f'PC2 ({explained[1]:.1f}%)',
            zaxis_title=f'PC3 ({explained[2]:.1f}%)',
        ),
        width=900,
        height=700,
        showlegend=True,
        legend=dict(
            title="Prompt type",
            itemsizing='constant'
        ),
    )

    return fig, W, mean


def plot_centroids_3d(paths, title=None):
    """
    Plot centroids in 3D using PCA on the centroids themselves.
    Uses plotly for full interactivity (rotate, zoom, pan).

    Args:
        paths: List of paths to NPZ files (each is a "run")
        title: Optional plot title
    """
    # Load all centroids
    all_runs = []
    all_meta = []
    for path in paths:
        X, names, prompt, seed, layer, tok = load_centroids_with_meta(path)
        all_runs.append(X)
        all_meta.append((names, prompt, seed, layer, tok))

    # Pool centroids for PCA (use all runs)
    pooled = np.vstack(all_runs)
    W, mean, explained = compute_pca_3d(pooled)

    print(f"PCA explained variance: PC1={explained[0]:.1f}%, PC2={explained[1]:.1f}%, PC3={explained[2]:.1f}%")
    print(f"Total explained: {sum(explained):.1f}%")

    # Color palette
    all_names = []
    for names, *_ in all_meta:
        for n in names:
            if n not in all_names:
                all_names.append(n)
    palette = {n: PLOTLY_COLORS[i % len(PLOTLY_COLORS)] for i, n in enumerate(all_names)}

    fig = go.Figure()

    # Plot each run
    for run_idx, (X, meta) in enumerate(zip(all_runs, all_meta)):
        names, prompt, seed, layer, tok = meta

        # Project onto PCA space
        centered = X - mean
        pts = centered @ W

        marker_symbol = PLOTLY_MARKERS[run_idx % len(PLOTLY_MARKERS)]

        # Add scatter points for each centroid
        for pt, name in zip(pts, names):
            fig.add_trace(go.Scatter3d(
                x=[pt[0]], y=[pt[1]], z=[pt[2]],
                mode='markers+text',
                marker=dict(
                    size=10,
                    color=palette[name],
                    symbol=marker_symbol,
                    line=dict(width=1, color='black')
                ),
                text=name if run_idx == 0 else '',  # Only label first run
                textposition='top center',
                textfont=dict(size=12, color=palette[name]),
                name=f'{name} (run {run_idx+1})',
                showlegend=(run_idx == 0),  # Only show legend for first run
                legendgroup=name,
            ))

        # Connect points with lines
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='lines',
            line=dict(width=3, color='gray'),
            name=f'Run {run_idx+1} trajectory',
            showlegend=False,
        ))

    # Layout
    if title is None and all_meta:
        layer, tok = all_meta[0][3], all_meta[0][4]
        title = f"Centroid trajectories (3D PCA)<br>last token '{tok}' @ {layer}"

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        scene=dict(
            xaxis_title=f'PC1 ({explained[0]:.1f}%)',
            yaxis_title=f'PC2 ({explained[1]:.1f}%)',
            zaxis_title=f'PC3 ({explained[2]:.1f}%)',
        ),
        width=900,
        height=700,
        showlegend=True,
    )

    return fig, W, mean

# %% [markdown]
# ## Configure and Plot
# Edit the paths below to plot different runs

# %% [markdown]
# ## SGD 5 epochs per stage

# %%
seed = 600
base_path_sgd_5eps = f'experiments/qa_cvdb_tveDefs_nEnts16000_eps5-5-5-5-5-5_bs256-256-256-256-256-256_Llama_3.2_1B_SGD_6stage/stage6_s{seed}/activation-centroids-and-percentiles-'

print("Loading centroids for all prompts from:")
print(f"  {base_path_sgd_5eps}[prompt]-seed{seed}.npz")

# %%
fig, W, mean = plot_centroids_3d_multi_prompt(
    base_path_sgd_5eps,
    seed=seed,
    title="SGD 5 eps/stage: Centroid trajectories in 3D PCA space (all prompts)"
)
fig.show()

# %% [markdown]
# ## SGD 10 epochs per stage

# %%
seed = 600
base_path_sgd_10eps = f'experiments/qa_cvdb_tveDefs_nEnts16000_eps10-10-10-10-10-10_bs256-256-256-256-256-256_Llama_3.2_1B_SGD_6stage/stage6_s{seed}/activation-centroids-and-percentiles-'

print("Loading centroids for all prompts from:")
print(f"  {base_path_sgd_10eps}[prompt]-seed{seed}.npz")

# %%
fig, W, mean = plot_centroids_3d_multi_prompt(
    base_path_sgd_10eps,
    seed=seed,
    title="SGD 10 eps/stage: Centroid trajectories in 3D PCA space (all prompts)"
)
fig.show()

# %% [markdown]
# ## SGD 25 epochs per stage

# %%
seed = 600
base_path_sgd_25eps = f'experiments/qa_cvdb_tveDefs_nEnts16000_eps25-25-25-25-25-25_bs256-256-256-256-256-256_Llama_3.2_1B_SGD_6stage/stage6_s{seed}/activation-centroids-and-percentiles-'

print("Loading centroids for all prompts from:")
print(f"  {base_path_sgd_25eps}[prompt]-seed{seed}.npz")

# %%
fig, W, mean = plot_centroids_3d_multi_prompt(
    base_path_sgd_25eps,
    seed=seed,
    title="SGD 25 eps/stage: Centroid trajectories in 3D PCA space (all prompts)"
)
fig.show()

# %% [markdown]
# ## LLaMA 3.2 1B Adafactor Run

# %%
seed = 600
# Using qd1_last version which has seed 600 with all prompts
base_path_llama_adafactor = f'experiments/qd1_last_qa_cvdb_tveDefs_nEnts16000_eps5and5and5and5and5and5_bs256and256and256and256and256and256_Llama_3.2_1B_ADAFACTOR_6stage/stage6_s{seed}/activation-centroids-and-percentiles-'

print("Loading centroids for all prompts from:")
print(f"  {base_path_llama_adafactor}[prompt]-seed{seed}.npz")

# %%
fig, W, mean = plot_centroids_3d_multi_prompt(
    base_path_llama_adafactor,
    seed=seed,
    title="LLaMA 3.2 1B Adafactor: Centroid trajectories in 3D PCA space (all prompts)"
)
fig.show()

# %%
