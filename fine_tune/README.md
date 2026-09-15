# Geospatial Foundation Model Fine-Tuning Framework (`fine_tune/`)

A modular, research-grade framework for fine-tuning remote sensing foundation models for **Sentinel-1 SAR Flood Inundation Segmentation** (Sen1Floods11) with low-VRAM optimization (6 GB NVIDIA RTX 4050 Laptop GPU / Kaggle / Colab GPU).

---

## 1. Overview & Supported Models

| Foundation Model | Organization | Modalities Supported | Pretrained Weights Reference | SAR Flood Compatibility |
| :--- | :--- | :--- | :--- | :--- |
| **TerraMind-1.0-base** | IBM / ESA | Multi-modal (Text, SAR VV/VH, MSI, DEM) | [`ibm-esa-geospatial/TerraMind-1.0-base`](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base) | **Native / Direct Support** (Primary model) |
| **Prithvi-EO-2.0-600M-TL** | IBM / NASA | Multi-Temporal Optical (B02, B03, B04, B8A, B11, B12) | [`ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL`](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL) | Multispectral Optical (Modality Guarded) |
| **SatMAE++** | BiliSakura | Grouped Multi-Spectral (RGB, RedEdge, NIR, SWIR) | [`BiliSakura/SATMAE-PP-transformers`](https://huggingface.co/BiliSakura/SATMAE-PP-transformers) | Grouped Optical (Modality Guarded) |
| **GFM Composition** | 05kashyap | SAR + Optical Multi-Sensor Fusion | [`05kashyap/GFM_Composition_Pretraining`](https://github.com/05kashyap/GFM_Composition_Pretraining) | Dual-Sensor / SAR Composition |

> [!IMPORTANT]
> **Modality Integrity**: Prithvi and SatMAE++ were pretrained on multi-spectral optical data. In accordance with rigorous remote sensing standards, our adapters **never silently spoof SAR channels into optical bands**. If an incompatible modality is supplied, the framework raises a descriptive `ModalityMismatchError`.

---

## 2. Directory Structure

```
fine_tune/
├── README.md                           # Comprehensive documentation & execution guide
├── requirements.txt                    # Minimal required packages
├── configs/                            # Experiment YAML configuration files
│   ├── terramind.yaml                  # TerraMind Sentinel-1 SAR flood segmentation
│   ├── prithvi.yaml                    # Prithvi EO 2.0 multispectral configuration
│   ├── satmaepp.yaml                   # SatMAE++ grouped multispectral configuration
│   └── gfm.yaml                        # GFM SAR+Optical composition configuration
├── datasets/                           # Data loading, parsing, and augmentation
│   ├── __init__.py
│   ├── sen1floods_dataset.py           # Sen1Floods11 GeoTIFF reader & SAR normalizer
│   ├── transforms.py                   # Joint geometric/radiometric data augmentations
│   └── datamodule.py                   # PyTorch DataLoader factory & split manager
├── models/                             # Foundation model adapters
│   ├── __init__.py
│   ├── base_model.py                   # FoundationModelBase interface & modality validator
│   ├── terramind.py                    # TerraMind adapter (native SAR support)
│   ├── prithvi.py                      # Prithvi adapter (optical / temporal support)
│   ├── satmaepp.py                     # SatMAE++ adapter (grouped multispectral support)
│   └── gfm.py                          # GFM adapter (dual-sensor fusion support)
├── decoders/                           # Downstream segmentation heads
│   ├── __init__.py
│   ├── segmentation_decoder.py         # Adaptive multi-scale convolutional decoder
│   └── unet_decoder.py                 # Progressive U-Net decoder with residual refinement
├── training/                           # Training loops, losses, and CLI trainers
│   ├── __init__.py
│   ├── trainer.py                      # Reusable PyTorch Trainer (AMP, Grad Accum, Early Stopping)
│   ├── losses.py                       # Masked BCE + Soft Dice combined loss
│   ├── train_terramind.py              # CLI trainer for TerraMind
│   ├── train_prithvi.py                # CLI trainer for Prithvi
│   ├── train_satmaepp.py               # CLI trainer for SatMAE++
│   └── train_gfm.py                    # CLI trainer for GFM
├── evaluation/                         # Evaluation and metrics suite
│   ├── __init__.py
│   ├── metrics.py                      # Flood IoU, Dice/F1, Precision, Recall, Specificity, Accuracy
│   ├── evaluate.py                     # Checkpoint evaluator on validation/test splits
│   └── visualize_predictions.py        # 4-panel visual comparison card generator
├── utils/                              # Hardware, config, logging, and checkpointing
│   ├── __init__.py
│   ├── config.py                       # YAML parser with ${ENV_VAR} interpolation
│   ├── checkpoint.py                   # Checkpoint manager & experiment.json logger
│   ├── logging.py                      # Console, CSV, and TensorBoard loggers
│   ├── seed.py                         # Deterministic multi-framework seed setter
│   └── device.py                       # Hardware auto-detection & low-VRAM monitoring
├── inference/                          # Production raster inference
│   ├── __init__.py
│   └── predict.py                      # GeoTIFF raster flood segmentation predictor
├── scripts/                            # Diagnostic and orchestration tools
│   ├── prepare_dataset.py              # Synthetic demo dataset generator & Sen1Floods11 helper
│   ├── test_model_loading.py           # Pre-flight model diagnostic tool
│   └── run_experiment.py               # Unified experiment runner
├── notebooks/                          # Analysis & exploration notebooks
│   ├── 01_dataset_analysis.ipynb       # Statistical backscatter distribution & class balance
│   ├── 02_model_comparison.ipynb       # Benchmark performance tables & comparison plots
│   └── 03_prediction_visualization.ipynb # Side-by-side qualitative prediction visualizer
├── checkpoints/                        # Saved model weights (best.pt, last.pt)
└── results/                            # Metrics JSON, CSV logs, and visual cards
```

---

## 3. Low VRAM (6 GB RTX 4050 / Laptop / Colab) Optimizations

To run large foundation models smoothly on GPUs with 6 GB VRAM:
1. **Automatic Mixed Precision (AMP)**: Enabled via PyTorch `torch.amp.autocast('cuda')` and `GradScaler`.
2. **Gradient Accumulation**: Simulated batch size $N = \text{batch\_size} \times \text{gradient\_accumulation\_steps}$ (e.g. $1 \times 4 = 4$).
3. **Micro-Batch Size**: Default `batch_size: 1` fits tokenized transformer features comfortably in VRAM.
4. **Frozen / Partial Fine-Tuning**: Freeze backbone to only train decoder weights (`strategy: frozen` or `strategy: partial`).

---

## 4. Quick Start & Execution Commands

### Step 1: Pre-flight Model Diagnostic Test
Verify model loading, parameter counts, and forward passes without training:
```bash
# Test TerraMind
python fine_tune/scripts/test_model_loading.py --model terramind

# Test all foundation models & modality guards
python fine_tune/scripts/test_model_loading.py --model all
```

### Step 2: Prepare Sample Dataset
Generate a synthetic Sen1Floods11 sample dataset for immediate testing:
```bash
python fine_tune/scripts/prepare_dataset.py --create-synthetic --output-dir fine_tune/data/sample_sen1floods11
```

### Step 3: Run a Tiny Sanity Training Run (1-2 Epochs)
```bash
python fine_tune/training/train_terramind.py --config fine_tune/configs/terramind.yaml --epochs 2 --batch-size 1
```

### Step 4: Run Full Fine-Tuning Experiment
```bash
# Using dedicated script
python fine_tune/training/train_terramind.py --config fine_tune/configs/terramind.yaml --epochs 10 --strategy frozen

# Or using unified experiment runner (trains, evaluates, and visualizes in one go)
python fine_tune/scripts/run_experiment.py --model terramind --epochs 10 --strategy frozen
```

### Step 5: Evaluate Checkpoint
```bash
python fine_tune/evaluation/evaluate.py --config fine_tune/configs/terramind.yaml --checkpoint fine_tune/checkpoints/terramind/best.pt --split valid
```

### Step 6: Generate Prediction Visualization Cards
```bash
python fine_tune/evaluation/visualize_predictions.py --config fine_tune/configs/terramind.yaml --checkpoint fine_tune/checkpoints/terramind/best.pt --num-samples 4
```

### Step 7: Run Raster Inference
```bash
python fine_tune/inference/predict.py --config fine_tune/configs/terramind.yaml --checkpoint fine_tune/checkpoints/terramind/best.pt --input fine_tune/data/sample_sen1floods11/S1Hand/India_Assam_0000_S1Hand.tif
```

---

## 5. Kaggle / Colab Cloud GPU Setup Guide

### Running on Kaggle / Colab GPU:
1. **Clone the repository (`fine_tune` branch)**:
   ```bash
   !git clone -b fine_tune https://github.com/<your-username>/Unified-RSanalytics.git
   %cd Unified-RSanalytics
   ```
2. **Install dependencies**:
   ```bash
   !pip install -r fine_tune/requirements.txt
   ```
3. **Attach Sen1Floods11 Dataset**:
   Add the Sen1Floods11 dataset to your Kaggle notebook input (e.g. `/kaggle/input/sen1floods11`).
4. **Set environment variable**:
   ```bash
   import os
   os.environ["DATASET_ROOT"] = "/kaggle/input/sen1floods11"
   ```
5. **Set Hugging Face token (if accessing gated weights)**:
   ```bash
   os.environ["HF_TOKEN"] = "hf_your_token_here"
   ```
6. **Launch fine-tuning**:
   ```bash
   !python fine_tune/scripts/run_experiment.py --model terramind --epochs 15 --strategy partial
   ```
7. **Download checkpoints & results**:
   Saved in `fine_tune/checkpoints/terramind/best.pt` and `fine_tune/results/terramind/`.

---

## 6. Fine-Tuning Strategies

Configure in YAML (`training.strategy`) or via CLI (`--strategy`):
- `frozen`: Backbone weights are completely frozen; only the segmentation decoder is updated. Ideal for low VRAM and fast convergence.
- `partial`: Unfreezes the last $N$ transformer blocks + decoder (`unfreeze_last_n_blocks: 2`).
- `full`: Unfreezes all backbone parameters and decoder. Recommended for high-end GPUs (>16 GB VRAM).
