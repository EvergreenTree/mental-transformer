# Mental Representation-Guided Supervision Baseline

A compact PyTorch baseline for aligning frozen image-model representations with
human fMRI-derived representational geometry. It implements the minimal core of
mental representation-guided supervision: precomputed image features, an image
projection head, an fMRI encoder, Sinkhorn matching, and RDM alignment.

This is not a full reproduction of the paper. It intentionally omits full
Gromov-Wasserstein graph matching, Gumbel-Softmax local graph sampling,
WordNet/THINGS/COCO evaluations, and model-scale sweeps.

## Install

```bash
python -m pip install -r requirements.txt
```

The code selects CUDA, Apple MPS, or CPU automatically. Pass `--device mps`,
`--device cuda`, or `--device cpu` to override.

## Quick Smoke Test

Run the synthetic paired-data path without downloading fMRI or images:

```bash
python -m mrgs.train \
  --config configs/minimal_clip.yaml \
  --synthetic \
  --subject S1 \
  --epochs 3 \
  --batch-size 64
```

## Real Data Path

The minimal real-data path uses the public Deep Image Reconstruction dataset and
offline frozen image features.

Sources:

- DIR / OpenNeuro: <https://openneuro.org/datasets/ds001506/versions/1.3.1>
- Preprocessed fMRI / figshare: <https://figshare.com/articles/dataset/Deep_Image_Reconstruction/7033577>
- ImageNet: <https://image-net.org/download>
- Single synset archives: `https://image-net.org/data/winter21_whole/[synsetid].tar`

Download available figshare files and write a local manifest:

```bash
python scripts/download_dir_metadata.py --download --out data/raw/DIR
```

Preprocess staged or downloaded DIR files:

```bash
python scripts/prepare_dir.py --raw data/raw/DIR --out data/processed --roi HVC
```

Expected processed files:

```text
data/processed/
  S1_train.pt
  S1_test.pt
  S2_train.pt
  S2_test.pt
  S3_train.pt
  S3_test.pt
```

Each file contains:

```python
{
    "image_paths": list[str],
    "class_ids": torch.LongTensor,
    "class_names": list[str],
    "fmri": torch.FloatTensor,
    "subject": str,
    "split": str,
}
```

If real image paths are not available, `prepare_dir.py` stores stable stimulus
IDs in `image_paths`; training can still run with aligned cached feature files.

## Image Features

The image encoder is frozen and used only for offline extraction. The default
backend is OpenCLIP `ViT-B-32` with `laion2b_s34b_b79k`; on MPS the default
feature-extraction batch size is `8`.

```bash
python scripts/extract_image_features.py \
  --processed data/processed/S1_train.pt \
  --output features/S1_train_features.pt \
  --backend open_clip \
  --device mps
```

Use MobileNetV3-small for a lighter fallback:

```bash
python scripts/extract_image_features.py \
  --processed data/processed/S1_train.pt \
  --output features/S1_train_features.pt \
  --backend mobilenet_v3_small
```

Feature files are saved on CPU:

```python
{
    "image_paths": list[str],
    "features": torch.FloatTensor,
    "backend": str,
    "model_name": str,
}
```

## Train And Evaluate

Train one subject with precomputed features:

```bash
python -m mrgs.train \
  --config configs/minimal_clip.yaml \
  --subject S1 \
  --processed-dir data/processed \
  --feature-dir features \
  --device mps
```

Evaluate a checkpoint:

```bash
python scripts/eval_subject.py \
  --checkpoint outputs/S1/last.pt \
  --processed data/processed/S1_test.pt \
  --features features/S1_test_features.pt \
  --device mps
```

Metrics include brain-to-image retrieval top-1, top-5, mean rank, median rank,
and RDM Spearman correlation.

## S1 Convenience Run

For a narrow end-to-end real-data run:

```bash
python scripts/run_minimal_real_s1.py --device mps
```

This checks/downloads figshare files, preprocesses S1, extracts or validates S1
features, runs a short training job, and prints data dimensions plus metrics.

## S1 Comparison

Run paper-aligned loss ablations on the same S1 train/test split and frozen
OpenCLIP features:

```bash
python scripts/run_comparison_s1.py \
  --processed-dir data/processed \
  --feature-dir features \
  --device mps \
  --epochs 10 \
  --batch-size 64
```

Methods:

- `contrastive_only`: contrastive loss only.
- `contrastive_ot`: contrastive plus Sinkhorn OT.
- `mrgs`: contrastive plus Sinkhorn OT plus RDM alignment.

Plot the comparison:

```bash
python scripts/plot_comparison.py --root outputs/comparison_s1
```

Outputs include `summary.csv`, retrieval bar charts, an RDM bar chart, and
training curves under `outputs/comparison_s1/`.

## Development

```bash
python -m pytest
```
