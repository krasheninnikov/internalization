# %%
from __future__ import annotations
"""CIFAR‑100 training script (ResNet‑26)"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────
import math
import argparse
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

from einops import rearrange
from torchvision import datasets, transforms



# ──────────────────────────────────────────────────────────────────────────────
# 1.  Model definition
# ──────────────────────────────────────────────────────────────────────────────

class ChannelLayerNorm(nn.Module):
    """Layer‑norm applied over the *channel* dimension of a 4‑D tensor."""

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, H, W)
        # Rearrange so that channels become the last dimension, apply layer‑norm,
        # then restore the original (B, C, H, W) layout.
        x = rearrange(x, "b c h w -> b h w c")
        x = self.layer_norm(x)
        return rearrange(x, "b h w c -> b c h w")


class ResidualBlock(nn.Module):
    """A basic residual block with pre‑norm and GELU activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.norm1 = ChannelLayerNorm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.activation = nn.GELU()
        self.norm2 = ChannelLayerNorm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        # The skip connection needs to down‑sample / change channel‑count when
        # stride > 1 *or* the number of channels changes.
        self.skip_connection: nn.Module
        if stride == 1 and in_channels == out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, H, W)
        residual = self.skip_connection(x)
        x = self.conv1(self.norm1(x))
        x = self.activation(x)
        x = self.conv2(self.norm2(x))
        return x + residual


class CifarResNet26(nn.Module):
    """Slightly modified ResNet‑26 variant for CIFAR‑100."""

    def __init__(self, num_classes: int = 100):
        super().__init__()
        widths: List[int] = [48, 96, 192, 384]

        # Stem
        self.stem = nn.Conv2d(3, widths[0], kernel_size=3, stride=1, padding=1, bias=False)

        # Residual groups; each _make call returns a Sequential of residual blocks.
        self.residual_group1 = self._make_group(widths[0], widths[0], num_blocks=3, stride=1)
        self.residual_group2 = self._make_group(widths[0], widths[1], num_blocks=3, stride=2)
        self.residual_group3 = self._make_group(widths[1], widths[2], num_blocks=3, stride=2)
        self.residual_group4 = self._make_group(widths[2], widths[3], num_blocks=3, stride=2)

        # Head = layer‑norm + global average pool + linear classifier
        self.head = nn.Sequential(ChannelLayerNorm(widths[3]), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(widths[3], num_classes, bias=False)

    @staticmethod
    def _make_group(
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Create a residual stage starting with a stride‑2 block if requested."""
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
# 2.  Data
# ──────────────────────────────────────────────────────────────────────────────
# NOTE: These are the *dataset‑wide* means and standard deviations of CIFAR‑100.
# They are hard‑coded here for reproducibility and because torchvision does not
# expose them directly.
# CIFAR100_MEAN: Tuple[float, float, float] = (0.5071, 0.4865, 0.4409)
# CIFAR100_STD: Tuple[float, float, float] = (0.2673, 0.2564, 0.2761)

# def loaders(
#     batch_size: int,
#     data_root: Path,
#     use_augmentation: bool,
# ) -> Tuple[DataLoader, DataLoader]:
#     """Return (train_loader, test_loader) for CIFAR‑100."""
#     train_transforms: List[transforms.Compose | transforms.Transform] = []
#     if use_augmentation:
#         train_transforms += [
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandAugment(num_ops=2, magnitude=9),
#         ]
#     train_transforms += [transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)]
#     if use_augmentation:
#         train_transforms.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)))
#     tf_train = transforms.Compose(train_transforms)

#     # Test transforms are deterministic.
#     tf_test = transforms.Compose(
#         [transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)]
#     )

#     # Instantiate datasets / dataloaders.
#     train_set = datasets.CIFAR100(data_root, train=True, download=True, transform=tf_train)
#     test_set = datasets.CIFAR100(data_root, train=False, download=True, transform=tf_test)

#     return (
#         DataLoader(train_set, batch_size, shuffle=True, num_workers=4, pin_memory=True),
#         DataLoader(test_set, batch_size, shuffle=False, num_workers=4, pin_memory=True),
#     )

# ─────────────────── 1. generic transform builder ───────────────────
def make_transforms(mean: Tuple[float, float, float], std: Tuple[float, float, float],
                    *, size: int = 32, aug: bool = True):
    """Return (train_tf, test_tf) pipelines with optional RandAug + erasing."""
    train_ops: List = []
    if aug:
        train_ops += [transforms.RandomCrop(size, padding=4, padding_mode="reflect"),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandAugment(num_ops=2, magnitude=9)]
    train_ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if aug:
        train_ops.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), value="random"))
    train_tf = transforms.Compose(train_ops)
    test_tf  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return train_tf, test_tf

# ─────────────────── 2. CIFAR-100 loaders ────────────────────────
def get_cifar_loaders(batch_size: int, data_root: Path, use_augmentation: bool = True):
    CIFAR100_MEAN, CIFAR100_STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)
    tf_train, tf_test = make_transforms(CIFAR100_MEAN, CIFAR100_STD, aug=use_augmentation)
    train_ds = datasets.CIFAR100(data_root, train=True,  download=True, transform=tf_train)
    test_ds  = datasets.CIFAR100(data_root, train=False, download=True, transform=tf_test)
    return (DataLoader(train_ds, batch_size, shuffle=True,  num_workers=4, pin_memory=True),
            DataLoader(test_ds,  batch_size, shuffle=False, num_workers=4, pin_memory=True))

# ─────────────────── 3. HF ImageNet-32 loaders ───────────────────
class HFImageNet32(Dataset):
    def __init__(self, split: str, transform, *, cache_dir: Path | None = None):
        self.ds   = load_dataset("benjamin-paine/imagenet-1k-32x32", split=split,
                                 cache_dir=str(cache_dir) if cache_dir else None, streaming=False)
        self.tfm  = transform
    def __len__(self):                      
        return len(self.ds)
    def __getitem__(self, idx):
        ex = self.ds[idx]
        return self.tfm(ex["image"]), ex["label"]          # label already 0-999

def get_imagenet32_loaders(batch_size: int, data_root: Path, use_augmentation: bool = True):
    IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    tf_train, tf_val = make_transforms(IMAGENET_MEAN, IMAGENET_STD, aug=use_augmentation)
    train_ds = HFImageNet32("train",       tf_train, cache_dir=data_root)
    val_ds   = HFImageNet32("validation",  tf_val,   cache_dir=data_root)
    return (DataLoader(train_ds, batch_size, shuffle=True,  num_workers=4, pin_memory=True),
            DataLoader(val_ds,   batch_size, shuffle=False, num_workers=4, pin_memory=True))

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Train / evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    top1 = (logits.argmax(dim=1) == targets).float().mean().item()

    top5 = (
        (logits.topk(5, dim=1).indices == targets.unsqueeze(1))
        .any(dim=1)                 # → shape (B,) of 0/1
        .float().mean().item()
    )
    return top1, top5

def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    is_training: bool,
) -> Tuple[float, float, float]:
    """Run a single epoch (train or eval) and return (loss, acc)."""
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

# ──────────────────────────────────────────────────────────────────────────────
# 4.  Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 100 if args.dataset == "cifar" else 1000
    get_loaders_fn = get_cifar_loaders if args.dataset == "cifar" else get_imagenet32_loaders

    train_loader, test_loader = get_loaders_fn(
        args.batch_size,
        args.data_root,
        use_augmentation=not args.disable_aug,  # Enabled by default.
    )

    model = CifarResNet26(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # ------------------------------------------------------------------
    # Learning‑rate scheduler: linear warm‑up → cosine decay.
    #   • For the first `warmup_epochs` we increase LR linearly.
    #   • Afterwards we follow a half‑period cosine schedule down to zero.
    # ------------------------------------------------------------------
    warmup_epochs = max(1, int(args.warmup_frac * args.epochs))

    def lr_schedule(epoch: int) -> float:
        if epoch < warmup_epochs:
            # Warm‑up phase (epoch counts from 0): scale linearly from 0 → 1.
            return (epoch + 1) / warmup_epochs
        # Cosine decay for the remaining epochs.
        progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

    best_val_accuracy = 0.0

    for epoch in range(args.epochs):
        train_loss, train_acc_top1, train_acc_top5 = run_epoch(model, train_loader, criterion, optimizer,device, is_training=True)
        val_loss, val_acc_top1, val_acc_top5 = run_epoch(model, test_loader, criterion, optimizer, device, is_training=False)
        scheduler.step()

        print(
            f"Epoch {epoch + 1:3d}/{args.epochs}: "
            f"train {train_acc_top1 * 100:6.2f}% | test {val_acc_top1 * 100:6.2f}%"
            f"\t(top5: {train_acc_top5 * 100:6.2f}% | {val_acc_top5 * 100:6.2f}%)"
        )

        if val_acc_top1 > best_val_accuracy:
            best_val_accuracy = val_acc_top1
            Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": model.state_dict(), "acc": best_val_accuracy, "epoch": epoch + 1},
                Path(args.ckpt_dir) / "best.pt",
            )

    print(f"Done.  Best test accuracy: {best_val_accuracy * 100:.2f}%")


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Command‑line interface
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=30, help="Total number of training epochs")  # 150 epochs gives ~73% val acc on cifar with bs=512
    parser.add_argument("--batch-size", type=int, default=3072, help="Global batch size")
    parser.add_argument("--lr", type=float, default=3e-3, help="Initial learning rate")
    parser.add_argument("--wd", type=float, default=1e-3, help="Weight decay")
    parser.add_argument("--warmup-frac", type=float,default=0.05, help="Fraction of epochs used for linear LR warm‑up")
    parser.add_argument("--data-root", type=Path, default=Path("./data"), help="Dataset root directory")
    parser.add_argument("--ckpt-dir", type=Path, default=Path("./checkpoints"), help="Checkpoint directory")
    parser.add_argument("--disable-aug", dest="disable_aug", action="store_true", help="Disable data augmentation (enabled by default)")
    parser.add_argument("--dataset", type=str, default="imagenet32", help="Dataset to use (cifar or imagenet32)")
    cli_args, _ = parser.parse_known_args()
    # main(cli_args)


# %%


# %%


# %%
# cifar_stage_experiment.py
"""Notebook‑friendly end‑to‑end pipeline for probing whether a CNN encodes
training‑stage information on CIFAR‑100.

Typical interactive workflow
----------------------------
```python
cfg = ExperimentConfig(stage_count=2,
                       split_strategy="even_per_class",
                       epochs_per_stage=5)  # quick smoke‑test
run_train(cfg)                                          # Sequential fine‑tune
acts, cls_lbls, stg_lbls = run_dump_activations(cfg)    # Stage‑labelled feats
run_probe_cv(cfg, acts, cls_lbls, stg_lbls)             # k‑fold probes & plots
```
The file assumes your **ResNet‑26 training script** (`CifarResNet26`,
`loaders`, `run_epoch`, and `accuracy`) has already been executed in an earlier
notebook cell – so those symbols live in `__main__`.
"""
from __future__ import annotations

# =============================================================================
# Imports & type hints
# =============================================================================
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import sys, math, pickle
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# =============================================================================
# A. Configuration
# =============================================================================
@dataclass
class ExperimentConfig:
    # ----- stage set‑up -------------------------------------------------------
    stage_count: int                = 2        # sequential training stages
    epochs_per_stage: int           = 10       # fine‑tuning epochs *per* stage
    split_strategy: str             = "even_per_class"  # or "classes_as_entities"
    split_seed: int                 = 42       # RNG for stage split

    # ----- optimisation -------------------------------------------------------
    batch_size: int                 = 512
    lr: float                       = 3e-3
    weight_decay: float             = 1e-3

    # ----- probe & dumping ----------------------------------------------------
    class_fold_count: int           = 5        # k in k‑fold CV over *classes*
    cv_seed: int                    = 0
    activation_dtype: torch.dtype   = torch.float16
    subsample_every_n: int | None   = None    # dump 1 / N images if set

    # ----- paths --------------------------------------------------------------
    data_root: Path                 = Path("./data")
    work_dir: Path                  = Path("./experiment_outputs")
    
    # --------------------------------------------------------------------------
    dataset: str = "cifar"  # either "cifar" or "imagenet32"

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def stage_ckpt(self, stage_id: int) -> Path:
        return self.work_dir / f"ckpt_{self.dataset}_stage{stage_id}.pt"

    def activation_dir(self) -> Path:
        return self.work_dir / "activations"

# =============================================================================
# B. Dataset splitting helpers
# =============================================================================

def _indices_by_class(ds) -> Dict[int, np.ndarray]:
    labels = np.array(ds.targets)
    return {c: np.where(labels == c)[0] for c in range(100)}


def _split_even_per_class(ds, stage_count: int, seed: int):
    rng = np.random.default_rng(seed)
    per_stage: Dict[int, List[int]] = {s: [] for s in range(stage_count)}
    for c, idxs in _indices_by_class(ds).items():
        rng.shuffle(idxs)
        chunks = np.array_split(idxs, stage_count)
        for s, chunk in enumerate(chunks):
            per_stage[s].extend(chunk.tolist())
    return per_stage


def _split_by_class(ds, stage_count: int, seed: int):
    rng = np.random.default_rng(seed)
    class_perm = rng.permutation(100)
    classes_per_stage = np.array_split(class_perm, stage_count)
    per_stage = {s: [] for s in range(stage_count)}
    labels = np.array(ds.targets)
    for s, cls_ids in enumerate(classes_per_stage):
        per_stage[s] = np.where(np.isin(labels, cls_ids))[0].tolist()
    return per_stage


def make_stage_split(ds, cfg: ExperimentConfig):
    if cfg.split_strategy == "even_per_class":
        return _split_even_per_class(ds, cfg.stage_count, cfg.split_seed)
    if cfg.split_strategy == "classes_as_entities":
        return _split_by_class(ds, cfg.stage_count, cfg.split_seed)
    raise ValueError("unknown split_strategy")


def make_cv_class_folds(k: int, seed: int = 0) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    return list(np.array_split(rng.permutation(100), k))

# =============================================================================
# Utility to pull symbols from earlier notebook cells
# =============================================================================

# def _sym(name: str):
#     try:
#         return getattr(sys.modules["__main__"], name)
#     except AttributeError as e:
#         raise RuntimeError(f"Symbol '{name}' must be defined earlier in the notebook.") from e

# CifarResNet26 = _sym("CifarResNet26")
# get_loaders        = _sym("get_cifar_loaders")
# run_epoch      = _sym("run_epoch")

# =============================================================================
# C. Multi‑stage training
# =============================================================================

def _train_one_stage(model, loaders_dict, cfg: ExperimentConfig, device, resume: Path | None, save_path: Path):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimiser = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    warm = max(1, int(0.05 * cfg.epochs_per_stage))
    lr_sched = optim.lr_scheduler.LambdaLR(
        optimiser,
        lr_lambda=lambda ep: (ep+1)/warm if ep < warm else 0.5*(1+math.cos(math.pi*((ep-warm)/(cfg.epochs_per_stage-warm))))
    )

    if resume and resume.exists():
        state = torch.load(resume, map_location="cpu")
        model.load_state_dict(state["model"])
        print("Resumed from", resume)

    best_val = 0.0
    for ep in range(cfg.epochs_per_stage):
        _ = run_epoch(model, loaders_dict["train"], criterion, optimiser, device, True)
        _, val_acc_top1, val_acc_top5 = run_epoch(model, loaders_dict["val"], criterion, optimiser, device, False)
        lr_sched.step()
        best_val = max(best_val, val_acc_top1)
        print(f"  ep {ep+1}/{cfg.epochs_per_stage}  val_acc={val_acc_top1:.3f}")
        if val_acc_top1 >= best_val:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "acc": best_val}, save_path)
    print("Stage done – best val_acc", best_val, "\n")


def run_train(cfg: ExperimentConfig):
    num_classes = 100 if cfg.dataset == "cifar" else 1000
    get_loaders_fn = get_cifar_loaders if cfg.dataset == "cifar" else get_imagenet32_loaders
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.work_dir.mkdir(parents=True, exist_ok=True)

    raw_train = datasets.CIFAR100(cfg.data_root, train=True, download=True, transform=None)
    stage_indices = make_stage_split(raw_train, cfg)
    pickle.dump(stage_indices, open(cfg.work_dir / "stage_indices.pkl", "wb"))

    model = CifarResNet26(num_classes=num_classes).to(device)
    for stage_id in range(cfg.stage_count):
        print(f"=== Stage {stage_id}/{cfg.stage_count-1} ===")
        base_train_loader, val_loader = get_loaders_fn(cfg.batch_size, cfg.data_root, use_augmentation=True)
        subset = Subset(base_train_loader.dataset, stage_indices[stage_id])
        train_loader = DataLoader(subset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        loaders_dict = {"train": train_loader, "val": val_loader}
        resume = cfg.stage_ckpt(stage_id-1) if stage_id > 0 else None
        _train_one_stage(model, loaders_dict, cfg, device, resume, cfg.stage_ckpt(stage_id))

    print("Training complete. Final checkpoint:", cfg.stage_ckpt(cfg.stage_count-1))

# =============================================================================
#  D.  Activation dumping  – 4×4 pooled activations saved as NumPy arrays
# =============================================================================
import torch.nn.functional as F
from collections import defaultdict
import numpy as np

# --------------------------------------------------------------------------- #
# 1)  Hook registration helper
# --------------------------------------------------------------------------- #
def _register_hooks(model: nn.Module,
                    dtype: torch.dtype,
                    pool_hw: int = 4):
    """
    Returns
    -------
    acts     : defaultdict(str -> List[Tensor])
               Each list accumulates *flattened* (B, C*P*P) tensors on CPU.
               (P == pool_hw).
    handles  : List[RemovableHandle]  – for later removal.
    """
    acts, handles = defaultdict(list), []

    pool = nn.AdaptiveAvgPool2d(pool_hw)          # shared module

    def make_hook(name):
        def _hook(_, __, out):
            # out : (B, C, H, W) → (B, C, P, P)
            flat = pool(out).flatten(start_dim=1) # (B, C*P*P)
            acts[name].append(flat.to("cpu", dtype))
        return _hook

    # register after every residual block
    for g in range(1, 5):                         # residual_group1 … 4
        group = getattr(model, f"residual_group{g}")
        for b, block in enumerate(group):
            h = block.register_forward_hook(
                make_hook(f"g{g}_b{b}")
            )
            handles.append(h)

    return acts, handles

# --------------------------------------------------------------------------- #
# 2)  Main dumping routine  (callable from notebook)
# --------------------------------------------------------------------------- #
def run_dump_activations(cfg: ExperimentConfig):
    """Forward all stage‑labelled images once and save pooled activations."""
    # ---------- load final checkpoint ----------
    state = torch.load(cfg.stage_ckpt(cfg.stage_count - 1), map_location="cpu")
    model = CifarResNet26(num_classes=100 if cfg.dataset == "cifar" else 1000)
    model.load_state_dict(state["model"])
    model.eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # ---------- dataset ----------
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409),
                             (0.2673, 0.2564, 0.2761)),
    ])
    full_train = datasets.CIFAR100(cfg.data_root, train=True,
                                   download=True, transform=tf)
    stage_indices = pickle.load(open(cfg.work_dir / "stage_indices.pkl", "rb"))

    # ---------- hooks ----------
    acts_dict, hook_handles = _register_hooks(model, cfg.activation_dtype)

    class_labels, stage_labels = [], []

    # ---------- forward pass ----------
    with torch.no_grad():
        for stage_id, idxs in stage_indices.items():
            if cfg.subsample_every_n:
                idxs = idxs[::cfg.subsample_every_n]

            loader = DataLoader(
                Subset(full_train, idxs),
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

            device = next(model.parameters()).device
            for imgs, cls in loader:
                imgs = imgs.to(device, non_blocking=True)
                model(imgs)                          # hooks collect feats
                class_labels.extend(cls.numpy())
                stage_labels.extend([stage_id] * imgs.size(0))

    # ---------- remove hooks ----------
    for h in hook_handles:
        h.remove()

    # ---------- concatenate & save ----------
    cfg.activation_dir().mkdir(parents=True, exist_ok=True)
    for layer, chunks in acts_dict.items():
        acts_layer = torch.cat(chunks)        # (N, C*16)  torch tensor
        acts_np    = acts_layer.cpu().numpy() # convert *once*
        np.save(cfg.activation_dir() / f"{layer}.npy", acts_np)
        acts_dict[layer] = acts_np            # keep ndarray in dict

    class_labels = np.asarray(class_labels, dtype=np.int16)
    stage_labels = np.asarray(stage_labels, dtype=np.int8)
    return acts_dict, class_labels, stage_labels


# =============================================================================
# E. Probe training + k‑fold CV (stand‑alone snippet)
# =============================================================================
"""Utilities to train linear probes that predict the *training stage* from
layer activations, with k‑fold cross‑validation over **classes**.

Public API
~~~~~~~~~~
run_probe_cv(cfg,
             acts_by_layer: Dict[str, np.ndarray],
             class_labels:  np.ndarray,   # shape (N,)
             stage_labels:  np.ndarray)   # shape (N,)
    • Fits a multinomial logistic‑regression probe *per layer*.
    • Uses k‑fold CV where each fold holds out an exclusive set of classes.
    • Prints per‑layer mean ± std accuracy and confusion matrices.
    • Returns a dict {layer_name: {"accs": [...], "models": [...]} } for
      downstream analysis / plotting.

plot_layer_projections(layer_name, model, acts, stage_labels, bins=100)
    • 1‑D histogram overlay of probe dot‑products coloured by stage id.
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _make_class_folds(k: int, seed: int = 0) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    return list(np.array_split(rng.permutation(100), k))


def _fit_single_probe(X_train: np.ndarray, y_train: np.ndarray,
                      max_iter: int = 2000) -> LogisticRegression:
    """Thin wrapper so hyper‑params live in one place."""
    clf = LogisticRegression(max_iter=max_iter,
                             multi_class="multinomial",
                             solver="lbfgs",
                             C=0.1,
                             n_jobs=4)
    clf = LogisticRegression(max_iter=5000, solver="saga", penalty="l2", n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf


def _evaluate_probe(clf: LogisticRegression, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    preds = clf.predict(X)
    acc   = (preds == y).mean()
    cm    = confusion_matrix(y, preds, labels=np.unique(y))
    return acc, cm

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def run_probe_cv(cfg,
                 acts_by_layer: Dict[str, np.ndarray],
                 class_labels:   np.ndarray,
                 stage_labels:   np.ndarray):
    """Train probes layer‑wise and print k‑fold accuracies."""

    # TODO this makes all folds correlated (across layers) -- should each layer have its own folds?
    folds = _make_class_folds(cfg.class_fold_count, cfg.cv_seed)
    results = {}

    for layer_name, acts in acts_by_layer.items():
        print(f"\n=== Probing layer: {layer_name} with {acts.shape[1:]} features per sample (total = {np.prod(acts.shape[1:])}) ===")
        acts = acts.astype(np.float32)   # logistic‑reg expects float32/64
        fold_accs: List[float] = []
        fold_models: List[LogisticRegression] = []

        for k, eval_classes in enumerate(folds):
            mask_eval  = np.isin(class_labels, eval_classes)
            X_train, y_train = acts[~mask_eval], stage_labels[~mask_eval]
            X_eval,  y_eval  = acts[mask_eval],  stage_labels[mask_eval]

            clf = _fit_single_probe(X_train, y_train)
            acc, cm = _evaluate_probe(clf, X_eval, y_eval)
            fold_accs.append(acc)
            fold_models.append(clf)
            print(f"  fold {k+1}/{len(folds)}  acc={acc:.3f}")
            # Optional: print confusion matrix per fold if desired
            # print(cm)

        mean_acc = np.mean(fold_accs); std_acc = np.std(fold_accs)
        print(f"  → mean accuracy {mean_acc:.3f} ± {std_acc:.3f}")
        results[layer_name] = {"accs": fold_accs, "models": fold_models}
    return results

# -----------------------------------------------------------------------------
# Visualisation helper
# -----------------------------------------------------------------------------

def plot_layer_projections(layer_name: str,
                           clf: LogisticRegression,
                           acts: np.ndarray,
                           stage_labels: np.ndarray,
                           bins: int = 100):
    """Overlay 1‑D histograms of probe dot‑products, colour‑coded by stage."""
    if acts.ndim > 2:
        acts_flat = acts.reshape(acts.shape[0], -1)
    else:
        acts_flat = acts

    projections = acts_flat @ clf.coef_.T  # shape (N, stage_count)
    # For multinomial LR the correct class is argmax; here we treat the *row*
    # corresponding to our learned weight for each stage id.
    # One convenient visual: use the column of *true* stage id so each example
    # uses the log‑odds of its ground‑truth class.
    proj_scalar = projections[np.arange(len(stage_labels)), stage_labels]

    unique_stages = np.unique(stage_labels)
    for s in unique_stages:
        plt.hist(proj_scalar[stage_labels == s], bins=bins, density=True,
                 alpha=0.6, label=f"stage {s}")
    plt.title(f"Probe projection – {layer_name}")
    plt.xlabel("logit along true‑stage weight vector")
    plt.ylabel("density")
    plt.legend()
    plt.show()

cfg = ExperimentConfig(
    stage_count=2,                 # change to 3,4,… as you wish
    split_strategy="classes_as_entities",  # or "even_per_class"
    epochs_per_stage=2,
    dataset="imagenet32",
)

# 1.  Sequential multi‑stage training
run_train(cfg)

# 2.  Dump activations after every residual block
acts_by_layer, class_labels, stage_labels = run_dump_activations(cfg)
#   • acts_by_layer: Dict[str, np.ndarray]  (shape = [N_images, C*H*W])
#   • class_labels : np.ndarray            (CIFAR‑100 class id for each row)
#   • stage_labels : np.ndarray            (stage id 0…S‑1 for each row)




# %%
print(acts_by_layer['g4_b2'].shape)
acts_by_layer_filtered = {k: v for k, v in acts_by_layer.items() if ('b2' in k)}  # only take block2 from each group
acts_by_layer_filtered.keys()

# %%
# 3.  k‑fold (class‑split) probes + plots
probe_stats = run_probe_cv(cfg,
                           acts_by_layer_filtered,
                           class_labels,
                           stage_labels)

# %%



