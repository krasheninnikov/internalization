# %%
from __future__ import annotations
"""
Vision staged-training + centroid-PCA visualization (idempotent training).

What's new vs your previous scripts
-----------------------------------
• Training is **idempotent**: it skips stages that already have checkpoints.
• Always writes a stage checkpoint (keeps best-val snapshot; failsafe save).
• Centroid-PCA viz includes:
  – Robust handling when some stages have 0 test samples (kept_stages)
  – Uses torch.inference_mode()
  – Configurable pooling size (--pool-hw)
  – Degenerate PCA checks + equal-aspect plotting
• Single entry point with subcommands: `train` and `centroid-pca`.

Example usage
-------------
# 1) Train 10 stages split by class (ImageNet-32) and cache checkpoints
python vision_stages_idempotent_and_centroid_pca.py \
    train --dataset imagenet32 --stage-count 10 --epochs-per-stage 5 \
    --split-strategy classes_as_entities --work-dir ./vision_experiment_outputs

# 2) Plot stage centroids for last-group block2 with PCA-on-centroids
python vision_stages_idempotent_and_centroid_pca.py \
    centroid-pca --dataset imagenet32 --stage-count 10 \
    --work-dir ./vision_experiment_outputs --layer-regex "^g4_b2$" \
    --pool-hw 4 --save-dir ./vision_experiment_outputs/centroid_pca_plots

Notes
-----
• This file is self-contained. It mirrors your model/data code and adds the
  requested enhancements. If you prefer importing from an existing module,
  you can delete the duplicated parts and import them instead.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────
import math
import re
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
import sys

# ──────────────────────────────────────────────────────────────────────────────
# 1) Model definition (ResNet-26 variant)
# ──────────────────────────────────────────────────────────────────────────────
from einops import rearrange


class ChannelLayerNorm(nn.Module):
    """Layer-norm over channel dimension of a 4D tensor (B, C, H, W)."""
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b c h w -> b h w c")
        x = self.layer_norm(x)
        return rearrange(x, "b h w c -> b c h w")


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.norm1 = ChannelLayerNorm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.activation = nn.GELU()
        self.norm2 = ChannelLayerNorm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        if stride == 1 and in_channels == out_channels:
            self.skip_connection: nn.Module = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip_connection(x)
        x = self.conv1(self.norm1(x))
        x = self.activation(x)
        x = self.conv2(self.norm2(x))
        return x + residual


class CifarResNet26(nn.Module):
    def __init__(self, num_classes: int = 100):
        super().__init__()
        widths: List[int] = [48, 96, 192, 384]
        self.stem = nn.Conv2d(3, widths[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.residual_group1 = self._make_group(widths[0], widths[0], num_blocks=3, stride=1)
        self.residual_group2 = self._make_group(widths[0], widths[1], num_blocks=3, stride=2)
        self.residual_group3 = self._make_group(widths[1], widths[2], num_blocks=3, stride=2)
        self.residual_group4 = self._make_group(widths[2], widths[3], num_blocks=3, stride=2)
        self.head = nn.Sequential(ChannelLayerNorm(widths[3]), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(widths[3], num_classes, bias=False)

    @staticmethod
    def _make_group(in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        blocks = [ResidualBlock(in_channels, out_channels, stride=stride)]
        blocks += [ResidualBlock(out_channels, out_channels) for _ in range(num_blocks - 1)]
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.residual_group1(x)
        x = self.residual_group2(x)
        x = self.residual_group3(x)
        x = self.residual_group4(x)
        x = self.head(x).flatten(1)
        return self.classifier(x)


# ──────────────────────────────────────────────────────────────────────────────
# 2) Data loading (CIFAR-100 & ImageNet-32)
# ──────────────────────────────────────────────────────────────────────────────

class HFImageNet32(Dataset):
    HF_ID = "benjamin-paine/imagenet-1k-32x32"
    LABEL_KEY = "label"

    def __init__(self, split: str, transform=None, *, cache_dir: Path | None = None):
        from datasets import load_dataset  # lazy import to avoid mandatory HF when unused
        self.ds = load_dataset(self.HF_ID, split=split, cache_dir=str(cache_dir) if cache_dir else None, streaming=False)
        self.tfm = transform
        self._targets_cache: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        ex = self.ds[idx]
        img = ex["image"]
        label = ex[self.LABEL_KEY]
        if self.tfm:
            img = self.tfm(img)
        return img, label

    @property
    def targets(self) -> np.ndarray:
        if self._targets_cache is None:
            self._targets_cache = np.array(self.ds[self.LABEL_KEY])
        return self._targets_cache

    @classmethod
    def raw_instance(cls, split: str, cache_dir: Path | None = None) -> "HFImageNet32":
        return cls(split=split, transform=None, cache_dir=cache_dir)


def make_transforms(mean: Tuple[float, float, float], std: Tuple[float, float, float], *, size: int = 32, aug: bool = True):
    train_ops: List = []
    if aug:
        train_ops += [transforms.RandomCrop(size, padding=4, padding_mode="reflect"),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandAugment(num_ops=2, magnitude=9)]
    train_ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if aug:
        train_ops.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), value="random"))
    train_tf = transforms.Compose(train_ops)
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return train_tf, test_tf


def get_cifar_loaders(batch_size: int, data_root: Path, use_augmentation: bool = True):
    CIFAR100_MEAN, CIFAR100_STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)
    tf_train, tf_test = make_transforms(CIFAR100_MEAN, CIFAR100_STD, aug=use_augmentation)
    train_ds = datasets.CIFAR100(data_root, train=True, download=True, transform=tf_train)
    test_ds = datasets.CIFAR100(data_root, train=False, download=True, transform=tf_test)
    return (
        DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True),
        DataLoader(test_ds, batch_size, shuffle=False, num_workers=4, pin_memory=True),
    )


def get_imagenet32_loaders(batch_size: int, data_root: Path, use_augmentation: bool = True):
    IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    tf_train, tf_val = make_transforms(IMAGENET_MEAN, IMAGENET_STD, aug=use_augmentation)
    train_ds = HFImageNet32("train", tf_train, cache_dir=data_root)
    val_ds = HFImageNet32("validation", tf_val, cache_dir=data_root)
    return (
        DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True),
        DataLoader(val_ds, batch_size, shuffle=False, num_workers=4, pin_memory=True),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3) Train helpers + idempotent staged training
# ──────────────────────────────────────────────────────────────────────────────

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    top1 = (logits.argmax(dim=1) == targets).float().mean().item()
    top5 = ((logits.topk(5, dim=1).indices == targets.unsqueeze(1)).any(dim=1)).float().mean().item()
    return top1, top5


def run_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device, is_training: bool) -> Tuple[float, float, float]:
    model.train(is_training)
    num_samples = 0
    loss_total = 0.0
    accuracy_total = 0.0
    accuracy_total_top5 = 0.0
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if is_training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if is_training:
            loss.backward()
            optimizer.step()
        batch_size = labels.size(0)
        num_samples += batch_size
        loss_total += loss.item() * batch_size
        top1, top5 = accuracy(logits.detach(), labels)
        accuracy_total += top1 * batch_size
        accuracy_total_top5 += top5 * batch_size
    return loss_total / num_samples, accuracy_total / num_samples, accuracy_total_top5 / num_samples


@dataclass
class ExperimentConfig:
    # Staging
    stage_count: int = 10
    epochs_per_stage: int = 5
    split_strategy: str = "classes_as_entities"  # or "even_per_class"
    split_seed: int = 42
    init_load_ckpt: Path | None = None
    skip_if_ckpt_exists: bool = True

    # Optim
    batch_size: int = 512
    lr: float = 3e-3
    weight_decay: float = 1e-3

    # Paths
    data_root: Path = Path("./data")
    work_dir: Path = Path("./vision_experiment_outputs")

    # Dataset choice
    dataset: str = "imagenet32"  # or "cifar"
    num_classes: int = field(init=False)

    def __post_init__(self):
        assert self.dataset in ["cifar", "imagenet32"]
        self.num_classes = 100 if self.dataset == "cifar" else 1000

    def stage_ckpt(self, stage_id: int) -> Path:
        return self.work_dir / f"ckpt_{self.dataset}_stage{stage_id}.pt"


# Dataset splitting helpers

def _indices_by_class(ds, num_classes: int) -> Dict[int, np.ndarray]:
    labels = np.array(ds.targets)
    return {c: np.where(labels == c)[0] for c in range(num_classes)}


def _split_even_per_class(ds, num_classes: int, stage_count: int, seed: int):
    rng = np.random.default_rng(seed)
    per_stage: Dict[int, List[int]] = {s: [] for s in range(stage_count)}
    for c, idxs in _indices_by_class(ds, num_classes).items():
        rng.shuffle(idxs)
        chunks = np.array_split(idxs, stage_count)
        for s, chunk in enumerate(chunks):
            per_stage[s].extend(chunk.tolist())
    return per_stage


def _split_by_class(ds, num_classes: int, stage_count: int, seed: int):
    rng = np.random.default_rng(seed)
    class_perm = rng.permutation(num_classes)
    classes_per_stage = np.array_split(class_perm, stage_count)
    per_stage = {s: [] for s in range(stage_count)}
    labels = np.array(ds.targets)
    for s, cls_ids in enumerate(classes_per_stage):
        per_stage[s] = np.where(np.isin(labels, cls_ids))[0].tolist()
    return per_stage


def make_stage_split(ds, cfg: ExperimentConfig):
    if cfg.split_strategy == "even_per_class":
        return _split_even_per_class(ds, cfg.num_classes, cfg.stage_count, cfg.split_seed)
    if cfg.split_strategy == "classes_as_entities":
        return _split_by_class(ds, cfg.num_classes, cfg.stage_count, cfg.split_seed)
    raise ValueError("unknown split_strategy")


def _train_one_stage(model: nn.Module, loaders_dict: Dict[str, DataLoader], cfg: ExperimentConfig, device: torch.device, resume: Path | None, save_path: Path):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimiser = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warm = max(1, int(0.05 * cfg.epochs_per_stage))
    def lr_schedule(epoch: int) -> float:
        if epoch < warm:
            return (epoch + 1) / warm
        progress = (epoch - warm) / max(1, (cfg.epochs_per_stage - warm))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    lr_sched = optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=lr_schedule)

    # Optional resume of weights
    if resume and Path(resume).exists():
        state = torch.load(resume, map_location="cpu")
        model.load_state_dict(state["model"])
        print("Resumed from", resume)

    best_val = -float("inf")
    best_state = None
    for ep in range(cfg.epochs_per_stage):
        _ = run_epoch(model, loaders_dict["train"], criterion, optimiser, device, True)
        _, val_acc_top1, _ = run_epoch(model, loaders_dict["val"], criterion, optimiser, device, False)
        lr_sched.step()
        print(f"  ep {ep+1}/{cfg.epochs_per_stage}  val_acc={val_acc_top1:.3f}")
        if val_acc_top1 >= best_val:
            best_val = val_acc_top1
            best_state = {"model": model.state_dict(), "acc": float(best_val)}
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, save_path)
    # Failsafe: ensure checkpoint exists
    if not Path(save_path).exists() and best_state is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, save_path)


def run_train(cfg: ExperimentConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.work_dir.mkdir(parents=True, exist_ok=True)

    # Build raw train set for splitting
    if cfg.dataset == "cifar":
        raw_train = datasets.CIFAR100(cfg.data_root, train=True, download=True, transform=None)
    else:
        raw_train = HFImageNet32.raw_instance("train", cache_dir=cfg.data_root)

    stage_indices = make_stage_split(raw_train, cfg)
    pickle.dump(stage_indices, open(cfg.work_dir / "stage_indices.pkl", "wb"))

    # Find first missing checkpoint
    start_stage = 0
    if cfg.skip_if_ckpt_exists:
        while start_stage < cfg.stage_count and cfg.stage_ckpt(start_stage).exists():
            start_stage += 1
        if start_stage >= cfg.stage_count:
            print("All stage checkpoints already exist — nothing to do.")
            print("Final checkpoint:", cfg.stage_ckpt(cfg.stage_count - 1))
            return

    # Create model and optionally resume from previous stage
    model = CifarResNet26(num_classes=cfg.num_classes).to(device)
    if start_stage > 0:
        print(f"Resuming from stage {start_stage-1} checkpoint")
        state = torch.load(cfg.stage_ckpt(start_stage - 1), map_location="cpu")
        model.load_state_dict(state["model"])

    # Loader factory
    get_loaders_fn = get_cifar_loaders if cfg.dataset == "cifar" else get_imagenet32_loaders

    # Train remaining stages
    for stage_id in range(start_stage, cfg.stage_count):
        print(f"=== Stage {stage_id}/{cfg.stage_count-1} ===")
        base_train_loader, val_loader = get_loaders_fn(cfg.batch_size, cfg.data_root, use_augmentation=True)
        subset = Subset(base_train_loader.dataset, stage_indices[stage_id])
        train_loader = DataLoader(subset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        loaders_dict = {"train": train_loader, "val": val_loader}
        resume = cfg.stage_ckpt(stage_id - 1) if stage_id > 0 else cfg.init_load_ckpt
        _train_one_stage(model, loaders_dict, cfg, device, resume, cfg.stage_ckpt(stage_id))

    print("Training complete. Final checkpoint:", cfg.stage_ckpt(cfg.stage_count - 1))


# ──────────────────────────────────────────────────────────────────────────────
# 4) Centroid PCA on test/val (PCA on centroids, not activations)
# ──────────────────────────────────────────────────────────────────────────────

class _CentroidAccumulator:
    def __init__(self):
        self.sums: Dict[str, np.ndarray] = {}
        self.counts: Dict[str, int] = {}
        self._dims: Dict[str, int] = {}

    def update(self, name: str, batch_flat_cpu: torch.Tensor):
        b, d = batch_flat_cpu.shape
        arr = batch_flat_cpu.numpy()
        if name not in self.sums:
            self.sums[name] = arr.sum(axis=0, dtype=np.float64)
            self.counts[name] = b
            self._dims[name] = d
        else:
            self.sums[name] += arr.sum(axis=0, dtype=np.float64)
            self.counts[name] += b

    def finalize(self) -> Dict[str, np.ndarray]:
        centroids = {}
        for name in self.sums:
            cent = (self.sums[name] / max(1, self.counts[name])).astype(np.float32, copy=False)
            centroids[name] = cent
        return centroids


def _register_centroid_hooks(model: nn.Module, pool_hw: int = 4):
    acc = _CentroidAccumulator()
    handles = []
    pool = nn.AdaptiveAvgPool2d(pool_hw)

    def make_hook(name: str):
        def _hook(_, __, out):
            flat = pool(out).flatten(start_dim=1).to("cpu", dtype=torch.float32)
            acc.update(name, flat)
        return _hook

    for g in range(1, 5):
        group = getattr(model, f"residual_group{g}")
        for b, block in enumerate(group):
            h = block.register_forward_hook(make_hook(f"g{g}_b{b}"))
            handles.append(h)

    return acc, handles


def _load_test_dataset(cfg: ExperimentConfig):
    if cfg.dataset == "cifar":
        CIFAR100_MEAN, CIFAR100_STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
        return datasets.CIFAR100(cfg.data_root, train=False, download=True, transform=tf)
    else:
        IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        return HFImageNet32("validation", transform=tf, cache_dir=cfg.data_root)


def _infer_stage_class_sets_from_pickle(cfg: ExperimentConfig) -> Dict[int, List[int]]:
    pkl_path = cfg.work_dir / "stage_indices.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Could not find {pkl_path}. Train first with the same work-dir/dataset."
        )
    stage_indices: Dict[int, List[int]] = pickle.load(open(pkl_path, "rb"))
    if cfg.dataset == "cifar":
        raw_train = datasets.CIFAR100(cfg.data_root, train=True, download=True, transform=None)
        train_targets = np.array(raw_train.targets)
    else:
        raw_train = HFImageNet32.raw_instance("train", cache_dir=cfg.data_root)
        train_targets = np.array(raw_train.targets)
    stage_to_classes: Dict[int, List[int]] = {}
    for s, idxs in stage_indices.items():
        cls_ids = np.unique(train_targets[idxs])
        stage_to_classes[s] = list(map(int, np.sort(cls_ids)))
    return stage_to_classes


def _make_test_indices_per_stage(test_ds, stage_to_classes: Dict[int, List[int]]) -> Dict[int, List[int]]:
    if hasattr(test_ds, "targets"):
        labels = np.array(getattr(test_ds, "targets"))
    else:
        labels = np.array(test_ds.targets)
    indices_per_stage = {}
    for s, cls_list in stage_to_classes.items():
        mask = np.isin(labels, np.array(cls_list))
        indices_per_stage[s] = np.nonzero(mask)[0].tolist()
    return indices_per_stage


def _compute_centroids_for_stage(model: nn.Module, dataset, stage_indices: List[int], batch_size: int, pool_hw: int = 4, num_workers: int = 4) -> Dict[str, np.ndarray]:
    loader = DataLoader(Subset(dataset, stage_indices), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    acc, handles = _register_centroid_hooks(model, pool_hw=pool_hw)
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        for imgs, _labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            _ = model(imgs)
    for h in handles:
        h.remove()
    return acc.finalize()


def _pca_project_centroids(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if C.shape[0] < 2:
        raise ValueError("Need >= 2 centroids for PCA projection.")
    Cc = C - C.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Cc, full_matrices=False)
    pcs = Vt[:2]
    proj = Cc @ pcs.T
    ev = (S ** 2) / max(1e-9, (S ** 2).sum())
    return proj, ev[:2]


def _plot_centroid_curve(proj: np.ndarray, stage_ids: List[int], title: str, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    xs, ys = proj[:, 0], proj[:, 1]
    plt.plot(xs, ys, marker="o", linewidth=1.5, markersize=6, alpha=0.9)
    for i, (x, y) in enumerate(zip(xs, ys)):
        plt.text(x, y, f"{stage_ids[i]}", fontsize=9, ha="center", va="bottom")
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='datalim')
    plt.xlabel("PC1 (centroids)")
    plt.ylabel("PC2 (centroids)")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def run_centroid_pca(cfg: ExperimentConfig, layer_regex: str = r"^g4_b2$", save_dir: Path | None = None, batch_size: int = 512, pool_hw: int = 4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = save_dir or (cfg.work_dir / "centroid_pca_plots")

    # Load final model (do NOT auto-train if missing; explicit by request)
    final_ckpt = cfg.stage_ckpt(cfg.stage_count - 1)
    if not final_ckpt.exists():
        raise FileNotFoundError(f"Missing final checkpoint: {final_ckpt}. Run 'train' first.")
    state = torch.load(final_ckpt, map_location="cpu")
    model = CifarResNet26(num_classes=cfg.num_classes).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"Loaded final checkpoint: {final_ckpt}")

    stage_to_classes = _infer_stage_class_sets_from_pickle(cfg)
    test_ds = _load_test_dataset(cfg)
    test_indices_per_stage = _make_test_indices_per_stage(test_ds, stage_to_classes)

    layer_pat = re.compile(layer_regex)
    centroids_by_layer: Dict[str, List[np.ndarray]] = {}
    kept_stages: List[int] = []

    ordered_stages = sorted(test_indices_per_stage.keys())
    for s in ordered_stages:
        idxs = test_indices_per_stage[s]
        if len(idxs) == 0:
            print(f"[warn] Stage {s} has 0 test samples — skipping.")
            continue
        print(f"Stage {s}: {len(idxs)} test samples")
        cents = _compute_centroids_for_stage(model, test_ds, idxs, batch_size=batch_size, pool_hw=pool_hw)
        kept_stages.append(s)
        for name, vec in cents.items():
            if layer_pat.search(name):
                centroids_by_layer.setdefault(name, []).append(vec)

    for name, cent_list in centroids_by_layer.items():
        if len(cent_list) < 2:
            print(f"[warn] Layer {name}: need >= 2 centroids for PCA, got {len(cent_list)}. Skipping.")
            continue
        C = np.stack(cent_list, axis=0)
        proj, ev2 = _pca_project_centroids(C)
        title = f"{name} — centroid PCA (PC1 {ev2[0]*100:.1f}%, PC2 {ev2[1]*100:.1f}%)"
        out = save_dir / f"centroid_pca_{name}.png"
        _plot_centroid_curve(proj, kept_stages, title, out)
        print(f"Saved: {out}")

# %%
# ──────────────────────────────────────────────────────────────────────────────
# 5) CLI
# ──────────────────────────────────────────────────────────────────────────────
# %% Notebook-friendly main runner


# Try a few likely module names; edit the first line if needed.


def main(
    *,
    # --- training defaults ---
    dataset: str = "imagenet32",            # "imagenet32" or "cifar"
    stage_count: int = 10,                   # 10 stages split by class
    epochs_per_stage: int = 20,
    split_strategy: str = "classes_as_entities",  # or "even_per_class"
    split_seed: int = 42,
    batch_size: int = 3072,
    lr: float = 3e-3,
    weight_decay: float = 1e-5,
    data_root: str | Path = "./data",
    work_dir: str | Path = "./vision_experiment_outputs",
    skip_if_ckpt_exists: bool = True,        # idempotent: skip finished stages

    # --- viz defaults ---
    layer_regex: str = r"^g4_b2$",           # choose layers to plot
    pool_hw: int = 4,                         # adaptive pool H=W before flatten
    viz_batch_size: int = 512,

    # --- pipeline toggles ---
    do_train: bool = True,                    # run idempotent training
    do_viz: bool = True,                      # make centroid-PCA plots
):
    """
    Run the staged training (idempotent) and centroid-PCA visualization with
    reasonable defaults. Call `main()` from a notebook cell, or override args:
        main(dataset="cifar", stage_count=10, epochs_per_stage=3)
    """
    data_root = Path(data_root)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig(
        dataset=dataset,
        data_root=data_root,
        work_dir=work_dir,
        stage_count=stage_count,
        epochs_per_stage=epochs_per_stage,
        split_strategy=split_strategy,
        split_seed=split_seed,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        skip_if_ckpt_exists=skip_if_ckpt_exists,
    )

    if do_train:
        print("=== Staged training (idempotent) ===")
        run_train(cfg)

    if do_viz:
        print("\n=== Centroid PCA (on centroids, not activations) ===")
        try:
            run_centroid_pca(
                cfg,
                layer_regex=layer_regex,
                save_dir=work_dir / "centroid_pca_plots",
                batch_size=viz_batch_size,
                pool_hw=pool_hw,
            )
        except FileNotFoundError as e:
            print(f"[warn] {e}")
            print("Hint: set do_train=True so checkpoints are created.")

    return cfg


# %% Example call (just run this cell)

if __name__ == "__main__":
    cfg = main()
# %%
