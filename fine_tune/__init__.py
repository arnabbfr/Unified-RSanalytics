"""Geospatial Foundation Model Fine-Tuning Framework (fine_tune).

A modular PyTorch framework for fine-tuning remote sensing foundation models:
- TerraMind-1.0-base (IBM / ESA)
- Prithvi-EO-2.0-600M-TL (IBM / NASA)
- SatMAE++ (BiliSakura)
- GFM Composition Pretraining (05kashyap)

Targeted at SAR flood segmentation (Sen1Floods11) with low-VRAM optimizations
for local RTX GPUs (e.g. 6 GB RTX 4050) and cloud accelerators (Kaggle / Google Colab).
"""

__version__ = "1.0.0"
