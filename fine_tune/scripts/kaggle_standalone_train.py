"""Self-contained end-to-end TransUNet flood segmentation trainer for Kaggle / Colab GPU.

Contains the complete TransUNet architecture, Global Batch Soft Dice Loss,
3-channel SAR preprocessor, and 6-metric benchmark evaluator with anti-overfitting regularization.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import torchvision.models as tv_models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ==============================================================================
# 1. SAR GEOTIFF READING & 3-CHANNEL PREPROCESSING
# ==============================================================================

def read_geotiff(path: Path | str) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    if RASTERIO_AVAILABLE:
        try:
            with rasterio.open(path) as src:
                arr = src.read()
                return arr.astype(np.float32)
        except Exception:
            pass

    if PIL_AVAILABLE:
        try:
            img = Image.open(path)
            arr = np.array(img, dtype=np.float32)
            if arr.ndim == 2:
                return arr
            elif arr.ndim == 3:
                return np.transpose(arr, (2, 0, 1))
            return arr
        except Exception:
            pass

    raise RuntimeError(f"Could not read image: {path}")


def normalize_sar(arr: np.ndarray, vv_min: float = -25.0, vv_max: float = 0.0, vh_min: float = -32.0, vh_max: float = -5.0) -> np.ndarray:
    """Convert linear SAR power to decibels (dB) and normalize to [0, 1]."""
    normed = np.zeros_like(arr, dtype=np.float32)
    for c in range(arr.shape[0]):
        ch = arr[c]
        valid = ch[~np.isnan(ch)]
        if valid.size == 0:
            normed[c] = 0.0
            continue
        c_min = float(np.min(valid))
        if c_min >= 0.0:
            ch_clipped = np.clip(ch, 1e-5, None)
            ch_db = 10.0 * np.log10(ch_clipped)
        else:
            ch_db = ch

        if c == 0:
            b_min, b_max = vv_min, vv_max
        elif c == 1:
            b_min, b_max = vh_min, vh_max
        else:
            p2 = float(np.nanpercentile(ch_db, 2))
            p98 = float(np.nanpercentile(ch_db, 98))
            b_min, b_max = p2, max(p98, p2 + 1e-4)

        normed[c] = np.clip((ch_db - b_min) / (b_max - b_min + 1e-6), 0.0, 1.0)
    return np.nan_to_num(normed, nan=0.0, posinf=1.0, neginf=0.0)


# ==============================================================================
# 2. DATASET WITH 3-CHANNEL SAR SYNTHESIS & JOINT AUGMENTATIONS
# ==============================================================================

class Sen1Floods11Dataset(Dataset):
    def __init__(self, data_root: str | Path, split: str = "train", image_size: int = 224, augment: bool = True):
        self.data_root = Path(data_root)
        self.split = split.lower()
        self.image_size = image_size
        self.augment = augment
        self.samples: List[Tuple[Path, Path]] = []
        self._discover_pairs()

    def _discover_pairs(self):
        # 1. Search for split CSVs
        csv_candidates = [
            self.data_root / "splits" / "flood_handlabeled" / f"flood_{self.split}_data.csv",
            self.data_root / f"flood_{self.split}_data.csv",
            self.data_root / f"{self.split}.csv",
        ]
        for c_path in csv_candidates:
            if c_path.is_file():
                with open(c_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if not row:
                            continue
                        img_rel, msk_rel = row[0].strip(), row[1].strip() if len(row) > 1 else row[0].replace("_S1Hand", "_LabelHand")
                        img_p = self.data_root / img_rel
                        msk_p = self.data_root / msk_rel
                        if not img_p.is_file():
                            img_p = self.data_root / "S1Hand" / Path(img_rel).name
                            msk_p = self.data_root / "LabelHand" / Path(msk_rel).name
                        if img_p.is_file() and msk_p.is_file():
                            self.samples.append((img_p, msk_p))
                if len(self.samples) > 0:
                    print(f"[{self.split.upper()} Dataset] Loaded {len(self.samples)} sample pairs from {c_path.name}")
                    return

        # 2. Recursive file discovery fallback
        all_tifs = list(self.data_root.rglob("*.tif*")) + list(self.data_root.rglob("*.png"))
        masks = [p for p in all_tifs if any(k in p.name.lower() for k in ("label", "mask", "gt"))]
        images = [p for p in all_tifs if not any(k in p.name.lower() for k in ("label", "mask", "gt"))]

        mask_map = {m.stem.lower().replace("_labelhand", "").replace("label", ""): m for m in masks}
        matched = []
        for img in images:
            key = img.stem.lower().replace("_s1hand", "").replace("image", "").replace("s1", "")
            if key in mask_map:
                matched.append((img, mask_map[key]))

        # Hash split fallback
        for img_p, msk_p in matched:
            h = abs(hash(img_p.name)) % 100
            if self.split == "train" and h < 70:
                self.samples.append((img_p, msk_p))
            elif self.split in ("val", "valid", "validation") and 70 <= h < 85:
                self.samples.append((img_p, msk_p))
            elif self.split == "test" and h >= 85:
                self.samples.append((img_p, msk_p))

        print(f"[{self.split.upper()} Dataset] Loaded {len(self.samples)} sample pairs (Total available: {len(matched)})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_p, msk_p = self.samples[idx]
        raw_img = read_geotiff(img_p)
        if raw_img.ndim == 2:
            raw_img = raw_img[np.newaxis, ...]
        raw_msk = read_geotiff(msk_p)
        if raw_msk.ndim == 3:
            raw_msk = raw_msk[0]

        # 1. Normalize SAR
        norm_img = normalize_sar(raw_img)

        # 2. Synthesize 3rd channel [VV, VH, VV - VH]
        if norm_img.shape[0] == 2:
            diff = np.clip(norm_img[0] - norm_img[1] + 0.5, 0.0, 1.0)
            norm_img = np.stack([norm_img[0], norm_img[1], diff], axis=0)
        elif norm_img.shape[0] > 3:
            norm_img = norm_img[:3]
        elif norm_img.shape[0] < 3:
            pad = ((0, 3 - norm_img.shape[0]), (0, 0), (0, 0))
            norm_img = np.pad(norm_img, pad, mode="edge")

        # 3. Standardize mask {0: non-flood, 1: flood, -1/255: ignore}
        raw_int = raw_msk.astype(np.int64)
        mask = np.copy(raw_int)
        u_vals = set(np.unique(raw_int))
        if u_vals.issubset({0, 255}) and 255 in u_vals:
            mask[mask == 255] = 1
        elif 255 in u_vals and 1 in u_vals:
            mask[mask == 255] = -1

        # 4. Joint Augmentations (Train only)
        if self.augment:
            if random.random() < 0.5:
                norm_img = np.flip(norm_img, axis=-1).copy()
                mask = np.flip(mask, axis=-1).copy()
            if random.random() < 0.5:
                norm_img = np.flip(norm_img, axis=-2).copy()
                mask = np.flip(mask, axis=-2).copy()
            if random.random() < 0.5:
                k = random.choice([1, 2, 3])
                norm_img = np.rot90(norm_img, k=k, axes=(-2, -1)).copy()
                mask = np.rot90(mask, k=k, axes=(-2, -1)).copy()
            if random.random() < 0.3:
                scale = random.uniform(0.92, 1.08)
                noise = np.random.normal(0.0, 0.02, size=norm_img.shape).astype(np.float32)
                norm_img = np.clip(norm_img * scale + noise, 0.0, 1.0)

        # 5. Resize to target size if needed
        c, h, w = norm_img.shape
        if h != self.image_size or w != self.image_size:
            it = torch.from_numpy(norm_img).unsqueeze(0).float()
            mt = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
            it = F.interpolate(it, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            mt = F.interpolate(mt, size=(self.image_size, self.image_size), mode="nearest")
            norm_img = it.squeeze(0).numpy()
            mask = mt.squeeze(0).squeeze(0).long().numpy()

        return torch.from_numpy(norm_img).float(), torch.from_numpy(mask).long()


# ==============================================================================
# 3. MODEL ARCHITECTURE: TRANSUNET (RESNET34 + TRANSFORMER + RESIDUAL SKIP UNET)
# ==============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.15):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(p=dropout)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return self.act(out + res)


class TransUNet(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int = 256, depth: int = 4, num_heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.embed_dim = embed_dim

        # 1. ResNet34 Stem & Stages
        if TORCHVISION_AVAILABLE:
            try:
                resnet = tv_models.resnet34(weights=tv_models.ResNet34_Weights.DEFAULT)
            except Exception:
                resnet = tv_models.resnet34(weights=None)
        else:
            resnet = tv_models.resnet34(weights=None) if TORCHVISION_AVAILABLE else None

        if resnet is not None:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if hasattr(resnet.conv1, "weight"):
                with torch.no_grad():
                    w = resnet.conv1.weight.data
                    if w.shape[1] == 3:
                        self.conv1.weight.data = w.clone()
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool
            self.layer1 = resnet.layer1  # (64, 56, 56)
            self.layer2 = resnet.layer2  # (128, 28, 28)
            self.layer3 = resnet.layer3  # (256, 14, 14)
        else:
            # Native fallback
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = nn.Sequential(ResidualBlock(64, dropout=0.0), ResidualBlock(64, dropout=0.0), ResidualBlock(64, dropout=0.0))
            self.layer2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.GELU(), ResidualBlock(128, dropout=0.0))
            self.layer3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.GELU(), ResidualBlock(256, dropout=0.0))

        # Skip projections
        self.skip_s8 = nn.Conv2d(128, 128, kernel_size=1)
        self.skip_s4 = nn.Conv2d(64, 64, kernel_size=1)
        self.skip_s2 = nn.Conv2d(64, 64, kernel_size=1)

        # 2. Transformer Attention Bottleneck
        self.trans_proj = nn.Conv2d(256, embed_dim, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, dropout=0.10, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.feat_proj = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(embed_dim), nn.GELU())

        # 3. Progressive Multi-Scale Skip Decoder with Spatial Dropout (p=0.15)
        # Stage 0: 14 -> 28 + s8 (128)
        self.up0 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block0 = nn.Sequential(nn.Conv2d(embed_dim + 128, 128, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(128), nn.GELU(), ResidualBlock(128, dropout=dropout))

        # Stage 1: 28 -> 56 + s4 (64)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block1 = nn.Sequential(nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU(), ResidualBlock(64, dropout=dropout))

        # Stage 2: 56 -> 112 + s2 (64)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block2 = nn.Sequential(nn.Conv2d(64 + 64, 32, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(32), nn.GELU(), ResidualBlock(32, dropout=dropout))

        # Stage 3: 112 -> 224
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block3 = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(16), nn.GELU(), ResidualBlock(16, dropout=dropout))

        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # Stem & ResNet
        s2 = self.relu(self.bn1(self.conv1(x)))
        s2_sk = self.skip_s2(s2)
        s4 = self.layer1(self.maxpool(s2))
        s4_sk = self.skip_s4(s4)
        s8 = self.layer2(s4)
        s8_sk = self.skip_s8(s8)
        s16 = self.layer3(s8)

        # Transformer Bottleneck
        s16_p = self.trans_proj(s16)
        ph, pw = s16_p.shape[-2], s16_p.shape[-1]
        tokens = s16_p.flatten(2).transpose(1, 2)
        n_p = tokens.shape[1]
        tokens = tokens + self.pos_embed[:, :n_p, :]
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        bot = self.feat_proj(tokens.transpose(1, 2).view(b, self.embed_dim, ph, pw))

        # Decoder U-Net
        d0 = self.block0(torch.cat([self.up0(bot), s8_sk], dim=1))
        d1 = self.block1(torch.cat([self.up1(d0), s4_sk], dim=1))
        d2 = self.block2(torch.cat([self.up2(d1), s2_sk], dim=1))
        d3 = self.block3(self.up3(d2))
        logits = self.head(d3)
        return logits


# ==============================================================================
# 4. BALANCED COMPOUND LOSS: WEIGHTED BCE (pos_weight=5.5) + GLOBAL DICE + TVERSKY
# ==============================================================================

class BalancedCombinedLoss(nn.Module):
    def __init__(self, pos_weight: float = 5.5, bce_weight: float = 0.4, dice_weight: float = 0.4, tversky_weight: float = 0.2, smooth: float = 1.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 4 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        valid_mask = (targets != -1) & (targets != 255)
        if not valid_mask.any():
            return logits.sum() * 0.0

        v_logits = logits[valid_mask]
        v_targets = (targets[valid_mask] == 1).float()

        # 1. Masked BCE with pos_weight (balances 85:15 class imbalance)
        weight = torch.tensor([self.pos_weight], device=logits.device)
        bce_loss = F.binary_cross_entropy_with_logits(v_logits, v_targets, pos_weight=weight)

        # 2. Global Batch Soft Dice (sums across all valid pixels in batch)
        probs = torch.sigmoid(v_logits)
        intersection = (probs * v_targets).sum()
        union = probs.sum() + v_targets.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice

        # 3. Tversky Recall Loss (alpha=0.3 FP, beta=0.7 FN)
        tp = (probs * v_targets).sum()
        fp = (probs * (1.0 - v_targets)).sum()
        fn = ((1.0 - probs) * v_targets).sum()
        tversky = (tp + self.smooth) / (tp + 0.3 * fp + 0.7 * fn + self.smooth)
        tversky_loss = 1.0 - tversky

        return (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
            + self.tversky_weight * tversky_loss
        )


# ==============================================================================
# 5. METRICS TRACKER
# ==============================================================================

class BenchmarkMetricsTracker:
    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        if logits.ndim == 4 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        probs = torch.sigmoid(logits)
        preds = (probs >= self.threshold).long()

        valid = (targets != -1) & (targets != 255)
        gt_binary = (targets == 1).long()

        p_v = preds[valid]
        t_v = gt_binary[valid]

        self.tp += int(((p_v == 1) & (t_v == 1)).sum().item())
        self.fp += int(((p_v == 1) & (t_v == 0)).sum().item())
        self.fn += int(((p_v == 0) & (t_v == 1)).sum().item())
        self.tn += int(((p_v == 0) & (t_v == 0)).sum().item())

    def compute(self) -> Dict[str, float]:
        eps = 1e-7
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        iou = tp / (tp + fp + fn + eps)
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        specificity = tn / (tn + fp + eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        return {
            "accuracy": round(float(accuracy), 4),
            "specificity": round(float(specificity), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "dice": round(float(dice), 4),
            "iou": round(float(iou), 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }


# ==============================================================================
# 6. COMPLETE END-TO-END TRAINING LOOP
# ==============================================================================

def train_transunet_flood(
    data_root: str = "/kaggle/input/datasets",
    epochs: int = 25,
    batch_size: int = 8,
    lr: float = 0.0003,
    output_dir: str = "fine_tune/results/terramind",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Hardware] Compute Device: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("fine_tune/checkpoints/terramind")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. DataLoaders
    train_ds = Sen1Floods11Dataset(data_root=data_root, split="train", augment=True)
    val_ds = Sen1Floods11Dataset(data_root=data_root, split="valid", augment=False)
    test_ds = Sen1Floods11Dataset(data_root=data_root, split="test", augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 2. Model, Loss, Optimizer
    model = TransUNet(in_channels=3, embed_dim=256, depth=4, num_heads=4, dropout=0.15).to(device)
    loss_fn = BalancedCombinedLoss(pos_weight=5.5, bce_weight=0.4, dice_weight=0.4, tversky_weight=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    tracker = BenchmarkMetricsTracker(threshold=0.35)
    best_dice = -1.0
    best_epoch = 0

    print("\n" + "=" * 70)
    print("STARTING TRANSUNET FINE-TUNING (BALANCED 6-METRIC BALANCED LOSS)")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {lr}")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        # TRAIN
        model.train()
        tracker.reset()
        t_loss = 0.0
        n_batches = len(train_loader)

        t0 = time.time()
        for step, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(images)
                loss = loss_fn(logits, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            t_loss += loss.item()

        scheduler.step()
        train_loss = t_loss / max(n_batches, 1)

        # VALIDATION
        model.eval()
        tracker.reset()
        v_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(images)
                    loss = loss_fn(logits, masks)
                v_loss += loss.item()
                tracker.update(logits, masks)

        val_loss = v_loss / max(len(val_loader), 1)
        val_m = tracker.compute()
        elapsed = time.time() - t0

        # Save Best
        curr_dice = val_m["dice"]
        is_best = (curr_dice > best_dice) if curr_dice > 0.0 else (val_loss < 0.8)
        star = " * [BEST]" if is_best else ""
        if is_best:
            best_dice = curr_dice
            best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict(), "metrics": val_m, "epoch": epoch}, ckpt_dir / "best.pt")

        print(
            f"Epoch {epoch:02d}/{epochs:02d} ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_m['dice']*100:.2f}% | Val IoU: {val_m['iou']*100:.2f}% | "
            f"Precision: {val_m['precision']*100:.2f}% | Recall: {val_m['recall']*100:.2f}%{star}"
        )

    # FINAL BENCHMARK TEST
    print("\n" + "=" * 70)
    print("RUNNING FINAL TEST BENCHMARK EVALUATION")
    print("=" * 70)
    best_ckpt = ckpt_dir / "best.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best checkpoint from Epoch {ckpt.get('epoch', best_epoch)}")

    model.eval()
    tracker.reset()
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            tracker.update(logits, masks)

    final_m = tracker.compute()

    print("=" * 65)
    print("FINAL 6-METRIC BALANCED BENCHMARK SUMMARY (SEN1FLOODS11)")
    print("=" * 65)
    print(f"  1. Overall Pixel Accuracy : {final_m['accuracy'] * 100:.2f}%")
    print(f"  2. Specificity (TNR)      : {final_m['specificity'] * 100:.2f}%")
    print(f"  3. Flood Precision        : {final_m['precision'] * 100:.2f}%")
    print(f"  4. Flood Recall (TPR)     : {final_m['recall'] * 100:.2f}%")
    print(f"  5. Flood Dice / F1-Score  : {final_m['dice'] * 100:.2f}%")
    print(f"  6. Flood IoU (Jaccard)    : {final_m['iou'] * 100:.2f}%")
    print("-" * 65)
    print(f"  Confusion: TP={final_m['tp']:,} | FP={final_m['fp']:,} | FN={final_m['fn']:,} | TN={final_m['tn']:,}")
    print("=" * 65)

    with open(out_path / "metrics_test.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": final_m, "best_epoch": best_epoch}, f, indent=2)

    return final_m


if __name__ == "__main__":
    train_transunet_flood()
