# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Preprocess BUSI dataset (set CURATED=True in script for curated version)
python -m src.dataset.Curated_BUSI_preprocessing

# Train (CV must be ≥ 2 in config.yaml)
python -m src.training_multitask      # segmentation + classification
python -m src.training_segmentation   # segmentation only
python -m src.training_classification # classification only

# Production variants (merge train+val, no separate val set)
python -m src.training_multitask_prod
python -m src.training_segmentation_prod
python -m src.training_classification_prod

# Federated (FedPer) training — see "Federated training" section below
python -m src.dataset.federated_partition   # build the master partition CSV (run first)
python -m src.training_federated            # run the federated simulation
```

All hyperparameters are controlled by `src/config.yaml`. The run creates a timestamped output directory under `runs/`.

## Architecture Overview

The project is a multi-task learning framework for simultaneous breast tumor **segmentation** and **classification** in 2D ultrasound images (128×128, grayscale).

### Data pipeline

- Raw BUSI dataset (`data/Dataset_BUSI_with_GT/`) is preprocessed by `src/dataset/Curated_BUSI_preprocessing.py`: images are resized to 128×128, multiple masks are merged, and a `mapping.csv` index is generated.
- The curated dataset (`data/Curated_BUSI_128/`) contains 450 images (222 benign, 164 malignant, 64 normal) after duplicate removal via SSIM.
- `BUSI_dataloader.py` reads `mapping.csv`, performs stratified K-fold splitting, then applies deterministic oversampling on the training fold to balance classes before constructing `DataLoader`s.

### Model zoo

Models live under `src/models/` in three sub-packages:

| Sub-package | Architectures |
|---|---|
| `segmentation/` | BTSUNet, nnUNet, ResidualUNet (custom); UNet, AttentionUNet, UnetPlusPlus, SwinUNETR, SegResNet (MONAI) |
| `classification/` | BTSUNetClassifier, UNetPlusPlusClassifier, nnUNetClassifier |
| `multitask/` | **MTnnUNet** (default), MTUNetPlusPlus, Multi_BTSUNet, Multi_FSB_BTS_UNet, AdityanNetwork |

`experiment_init.py` dispatches architecture names to constructors and wires up the optimizer, loss criteria, and LR scheduler from config values.

### MTnnUNet (default multi-task model)

5-level encoder-decoder (U-Net style, fixed widths `[32, 64, 128, 256, 320]`) with:
- **Deep supervision**: 4 segmentation outputs at different scales; losses are inversely weighted (`1/(n+1)`) if `loss.inversely_weighted=True`.
- **Classification head**: fuses `encoder5`, `upsample(bottleneck)`, and `process_decoder5` features via concatenation → `ConvInNormLeReLU` → `AdaptiveAvgPool2d` → two FC layers.
- `forward()` returns `([cls_logits], [seg4, seg3, seg2, seg1])`.

### Loss and training loop

Multi-task loss: `total = α * seg_loss + (1-α) * cls_loss`  
- `α` is `training.alpha` in config (default 0.85, so segmentation dominates).
- Segmentation criterion: DICE (MONAI), with sigmoid applied inside the criterion — **do not apply sigmoid before passing to the loss**.
- Classification criterion: Focal loss for multiclass, BCE for binary.
- Early stopping via `training.max_patience`; best checkpoint saved per fold.

### Output structure

```
runs/{timestamp}_{arch}_{width}_alpha_{α}_batch_{B}_{classes}/
  config.yaml          # copy of the config used
  model.txt            # printed model architecture
  fold_{n}/
    model_*_fold_{n}   # best checkpoint (torch.save dict)
    metrics.csv
    loss_evolution.png
    segs/              # predicted segmentation PNGs
```

### Key config parameters

| Key | Effect |
|---|---|
| `model.architecture` | Which model class to instantiate |
| `training.CV` | Number of folds (must be ≥ 2) |
| `training.alpha` | Weight on seg loss (0=cls only, 1=seg only) |
| `data.input_img` | Path to preprocessed dataset folder containing `mapping.csv` |
| `data.classes` | Which classes to include; determines binary vs. multiclass mode |
| `data.oversampling` | Enables deterministic oversampling in train folds |
| `loss.inversely_weighted` | Weight deep supervision outputs by `1/(n+1)` |

## Federated training (FedPer)

A federated variant trains the multi-task model across clients using **Flower** (`flwr[simulation]`)
simulation + **FedAvg**, following the **FedPer** scheme: the encoder is shared/federated, while each
client keeps a **personalized head** that is never aggregated. Each client owns **one task** (seg *or*
cls); the same image may be used by clients of different tasks, but train/test stay disjoint per fold.

### Pipeline

1. `src/dataset/federated_partition.py` — splits the curated dataset at the image level with
   `StratifiedKFold`, then partitions each fold's train pool across the clients of each task using a
   **Dirichlet(α)** distribution (high α ≈ IID). Writes a single **master CSV**
   (`data/federated/federated_mapping.csv`) with columns `fold, client_id, task, split`. The `normal`
   class is dropped from segmentation clients; a validation slice is carved per client for early stopping.
2. `src/dataset/federated_dataloader.py` — filters the master CSV into the existing `BUSI` dataset
   (per-task oversampling), no change to the dataset itself.
3. `src/federated/model_split.py` — splits MTnnUNet into the **shared** block (`encoder1..5` +
   `bottleneck`, federated) vs the **personalized** block (decoders, upsamples, outputs, classifier — local).
4. `src/federated/local_trainer.py` — per-task local train/eval (seg → Dice only, cls → Focal only).
5. `src/federated/client.py` — Flower `NumPyClient`; persists the personalized state + optimizer and a
   `best.pt` snapshot (lowest own val loss) **to disk per client** (simulation clients are ephemeral).
   Resolves its device per worker (falls back to CPU if the worker has no GPU).
6. `src/federated/server.py` — `FedAvg` subclass aggregating **only the encoder**, weighted by
   `num_examples × task_weight[task]`; logs mean val loss and best round.
7. `src/training_federated.py` — per-fold orchestrator: runs `flwr` simulation, then evaluates each
   client's `best.pt` on its own held-out test split. Saves `federated_test_results.csv` (per
   client × fold) and `federated_summary.csv` (per task) under `runs/{ts}_FEDPER_*/`.

The final artifact is **one shared encoder + N personalized heads** (not a single global model).
Server-side early stopping is logged but not enforced; each client keeps its own best snapshot.

### Key federated config (`federated:` section)

| Key | Effect |
|---|---|
| `n_clients_seg` / `n_clients_cls` | Number of clients per task |
| `rounds` / `local_epochs` | Federated rounds × local epochs per round |
| `dirichlet_alpha` | Train partition skew (high = IID, low = non-IID) |
| `task_weights.{seg,cls}` | Extra per-task weight when aggregating the shared encoder |
| `oversampling.{seg,cls}` | Per-task override of `data.oversampling` |
| `device` | `auto` / `cpu` / `cuda` (resolved per worker) |
| `client_resources.num_gpus` | `>0` lets a client use the GPU; `1.0` = 1 client/GPU (sequential) |
| `ray_num_cpus` | Caps Ray concurrency to bound memory use |

### Notes / gotchas

- `flwr` requires `protobuf>=4`, which is incompatible with the old pinned `tensorboard==2.10.0`;
  tensorboard was removed (it was a dead import). **numpy must stay `<2`** (pandas/monai ABI).
- Running clients in parallel can exhaust system RAM (and GPU memory); keep `ray_num_cpus` /
  `client_resources` conservative. The defaults run one client at a time.
