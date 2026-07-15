# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Preprocess a dataset -> data/<dataset>/<variant>/. Each script refuses to run unless
# `data.dataset` in config.yaml names the dataset it handles (it would otherwise overwrite
# another dataset's variant), so set that first.
python -m src.dataset.Curated_BUSI_preprocessing   # data.dataset: Curated_BUSI
python -m src.dataset.ISIC_2018_preprocessing      # data.dataset: ISIC_2018

# Train (CV must be ≥ 2 in config.yaml)
python -m src.training_multitask      # segmentation + classification
python -m src.training_segmentation   # segmentation only
python -m src.training_classification # classification only

# Production variants (merge train+val, no separate val set)
python -m src.training_multitask_prod
python -m src.training_segmentation_prod
python -m src.training_classification_prod

# Federated (FedPer) training — see "Federated training" section below
python -m src.dataset.federated_partition   # build data/federated_multi/federated_mapping.csv
python -m src.dataset.federated_partition --legacy  # only to reproduce frozen single-dataset BUSI
python -m src.training_federated            # run the federated simulation
python -m scripts.smoke_federated --setup both  # paired 2-round CPU end-to-end validation

# Comparison experiment (Federated vs Local-only vs Centralized) — see section below
python -m src.training_federated            # federated  (federated.standalone: False)
python -m src.training_federated            # local-only (set federated.standalone: True first)
python -m src.training_centralized          # centralized MTL (from the same master CSV)
python -m src.experiments.analyze --results <3 *_test_results.csv> --preds <3 *_cls_predictions.csv> --out runs/comparison
```

All hyperparameters are controlled by `src/config.yaml`. The run creates a timestamped output directory under `runs/`.

## Architecture Overview

The project is a multi-task learning framework for simultaneous breast tumor **segmentation** and **classification** in 2D ultrasound images (128×128, grayscale).

### Dataset layout (multi-dataset)

Each dataset owns one folder under `data/` and follows the SAME internal convention, so no code
hardcodes a dataset path — switching datasets is a one-line change to `data.dataset` in config.yaml:

```
data/<dataset>/
  raw/           # original download, extracted (only the preprocessing script reads it)
  archives/      # original .zip archives
  <variant>/     # preprocessed images/, masks/, mapping.csv   <- what training reads
  federated/     # federated_mapping.csv, the frozen master partition
```

Currently present: `data/Curated_BUSI/` and `data/ISIC_2018/`, both with a `processed_128` variant.
The variant's resolution comes from `data.image_size`; the preprocessing refuses to write a
different size into an existing variant.

**All dataset paths are derived in `src/dataset/paths.py` from `data.root`/`data.dataset`/`data.variant`.**
Never rebuild these paths by hand — call `paths.processed_dir()`, `paths.mapping_file()`,
`paths.partition_file()`. Deriving rather than storing them is what guarantees every setup
(federated / standalone / centralized) reads the same partition, which the comparison relies on.

### Data pipeline

Each dataset has its OWN preprocessing script (the raw layouts have nothing in common); whatever is
genuinely shared lives in `src/dataset/preprocessing_utils.py` (mask statistics, the mapping metadata
columns, resize helpers, the guards). All of them emit the same contract: `<variant>/{images,masks,mapping.csv}`.

- **Curated BUSI** — `src/dataset/Curated_BUSI_preprocessing.py`. Raw at `data/Curated_BUSI/raw/`; images resized to `data.image_size`, multiple masks merged, `mapping.csv` generated. `CURATED` filters through `data/Curated_BUSI/curation_list.csv`, leaving 450 images (222 benign, 164 malignant, 64 normal) after SSIM duplicate removal. It resizes with `INTER_NEAREST` — **do not "fix" this to INTER_AREA**: it would change the curated images and invalidate the frozen federated partition and every result derived from it.
- **ISIC 2018** — `src/dataset/ISIC_2018_preprocessing.py`. Ingests BOTH challenge tasks into one mapping: Task 1 (3,694 images with masks) and Task 3/HAM10000 (11,720 images with labels). Images written RGB with `INTER_AREA`, masks nearest-neighbour + re-binarized. See `data/ISIC_2018/PAPER_NOTES.md` for the measured facts.
- `BUSI_dataloader.py` reads `mapping.csv`, performs stratified K-fold splitting, then applies deterministic oversampling on the training fold to balance classes before constructing `DataLoader`s.

#### `mapping.csv` schema

Base columns (every dataset): `img_path, mask_path, class, id, dim1, dim2, tumor_pixels, y_max,
y_min, x_max, x_min, y_size, x_size`.

ISIC 2018 adds three, because its two tasks are **disjoint image sets** (verified: zero ID overlap —
no image has both a mask and a label, so BUSI's assumption that every row has both does not hold):

| Column | Meaning |
|---|---|
| `task` | `seg` (has `mask_path`, `class` empty) or `cls` (has `class`, `mask_path` empty) |
| `official_split` | `train`/`val`/`test` — which official release the row came from |
| `lesion_id` | groups images of the same physical lesion. **Split on this, not on `id`** — 44.9% of Task 3 images share a lesion. Only published for Task 3 *training*; empty elsewhere. |

Mask-derived columns are `NaN` (not `0`) on rows without a mask: `0` legitimately means "empty mask"
for BUSI's `normal` class, so `NaN` is the only honest "not applicable".

**Consequence for ISIC:** it suits FedPer (each client owns one task, so disjoint pools are natural)
but **cannot feed `training_centralized.py`**, which needs image+mask+label on the same row.

Paths inside `mapping.csv` / `federated_mapping.csv` are relative to the repo root, so **moving a
dataset folder invalidates both CSVs** — regenerate them (deterministic given the same config, so
the partition is reproduced exactly) rather than editing them by hand.

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
| `data.root` / `data.dataset` / `data.variant` | Select the dataset folder + preprocessed variant (see "Dataset layout") |
| `data.classes` | Which classes to include; determines binary vs. multiclass mode |
| `data.seg_exclude_classes` | Classes dropped from the segmentation task (empty masks); `[normal]` for BUSI |
| `data.oversampling` | Enables deterministic oversampling in train folds |
| `loss.inversely_weighted` | Weight deep supervision outputs by `1/(n+1)` |

## Federated training (FedPer)

A federated variant trains the multi-task model across clients using **Flower** (`flwr[simulation]`)
simulation + **FedAvg**, following the **FedPer** scheme: the encoder is shared/federated, while each
client keeps a **personalized head** that is never aggregated. Each client owns **one task** (seg *or*
cls); the same image may be used by clients of different tasks, but train/test stay disjoint per fold.

### Pipeline

The default federated configuration is multi-dataset: Curated BUSI (1-channel ultrasound) and
ISIC 2018 (3-channel dermoscopy). `encoder1` is a personalized modality stem; the shared trunk is
`encoder2..5 + bottleneck`. Set `share_stem: True` only for compatible single-dataset experiments.

1. `src/dataset/federated_partition.py` — builds a separate multi-dataset master. BUSI uses the
   historical image-level stratified folds. ISIC segmentation uses KFold; ISIC classification uses
   grouped stratified folds, group-preserving client allocation, and group-preserving train/val, so
   a `lesion_id` never crosses a client or split. The `--legacy` CLI retains exact BUSI reproduction.
   The original single-dataset design splits the curated dataset at the image level with
   `StratifiedKFold`, then partitions each fold's train pool across the clients of each task using a
   **Dirichlet(α)** distribution (high α ≈ IID). Writes a single **master CSV**
   (`data/<dataset>/federated/federated_mapping.csv`) with columns `fold, client_id, task, split`. The
   classes in `data.seg_exclude_classes` are dropped from segmentation clients; a validation slice is
   carved per client for early stopping.
2. `src/dataset/federated_dataloader.py` — filters the master CSV into the lazy, channel-aware
   `BUSI` compatibility dataset, applies per-task oversampling, and resolves fold/local class weights.
3. `src/federated/model_split.py` — splits MTnnUNet into the shared trunk (`encoder2..5` +
   `bottleneck`) vs personalized stem/decoders/outputs/classifier. Legacy `share_stem=True` also
   federates `encoder1`.
4. `src/federated/local_trainer.py` — per-task local train/eval (seg → Dice only, cls → Focal only).
5. `src/federated/client.py` — Flower `NumPyClient`; persists the latest personalized state +
   optimizer **to disk per client** (simulation clients are ephemeral). Stable worker seeds pair
   model initialization, shuffling, and transforms between federated and local-only runs. A
   `best.pt` local post-fit snapshot remains diagnostic and is not the final federated artifact.
   Resolves its device per worker (falls back to CPU if the worker has no GPU).
6. `src/federated/server.py` — `hierarchical` mode normalizes
   `num_examples × task_weight[task]` within each dataset and then applies `dataset_weights`;
   `flat` mode is the sample-weighted ablation. Logs effective dataset/task participation.
7. `src/training_federated.py` — per-fold orchestrator: validates shared shapes before Flower, runs
   the simulation, persists `global_shared.pt`, and evaluates every federated client with the same
   final global trunk plus its latest personalized state. Local-only evaluates each latest full
   local model at the identical round budget. Saves `{setup}_test_results.csv` and
   `{setup}_cls_predictions.csv` under the timestamped run directory.

The final artifact is **one shared trunk + N personalized stems/heads** (not a single global model).
Server-side early stopping is diagnostic only; final comparison uses the same configured last-round
budget in both arms.

### Key federated config (`federated:` section)

| Key | Effect |
|---|---|
| `datasets` / `n_clients.<dataset>` | Active datasets and clients per dataset/task |
| `share_stem` | Federate the input stem; must be false for mixed 1ch/3ch runs |
| `aggregation.mode` | `hierarchical` (main) or `flat` (FedAvg-style ablation) |
| `aggregation.dataset_weights` | Dataset-level mixture weights; 1:1 gives exact 50/50 participation |
| `datasets.<name>.class_weighting` | `none`, `balanced_fold` (main), or `balanced_local` |
| `rounds` / `local_epochs` | Federated rounds × local epochs per round |
| `dirichlet_alpha` | Train partition skew (high = IID, low = non-IID) |
| `aggregation.task_weights.{seg,cls}` | Per-task weight inside each dataset aggregate |
| `oversampling.{seg,cls}` | Per-task override of `data.oversampling` |
| `device` | `auto` / `cpu` / `cuda` (resolved per worker) |
| `client_resources.num_gpus` | `>0` lets a client use the GPU; `1.0` = 1 client/GPU (sequential) |
| `ray_num_cpus` | Caps Ray concurrency to bound memory use |

### Notes / gotchas

- `flwr` requires `protobuf>=4`, which is incompatible with the old pinned `tensorboard==2.10.0`;
  tensorboard was removed (it was a dead import). **numpy must stay `<2`** (pandas/monai ABI).
- Running clients in parallel can exhaust system RAM (and GPU memory); keep `ray_num_cpus` /
  `client_resources` conservative. The defaults run one client at a time.

## Comparison experiment

The original BUSI-only study compares three setups. The multi-dataset study intentionally compares
only Federated vs Local-only on the same master; centralized MTL is out of scope because no ISIC row
contains both mask and diagnosis. `unified_eval.py` remains the shared scorer.

| Setup | Role | How to run |
|---|---|---|
| **Local-only** per client | controlled baseline (floor) | `training_federated.py` with `federated.standalone: True` (clients ignore the federated encoder) |
| **Federated (FedPer)** | the proposed method | `training_federated.py` with `federated.standalone: False` |
| **Centralized MTL** | BUSI-only upper bound | Not valid for the multi-dataset ISIC experiment |

Each setup writes its own `{setup}_test_results.csv` (+ `{setup}_cls_predictions.csv`) under its run dir.

**Fairness controls (baked in):** federated and local-only resolve the same configured master and
error if it is missing (generate it once with `federated_partition`, then freeze it). They receive
the same initial shared trunk and deterministic per-client/fold/round seeds; therefore local stems,
heads, batch shuffles, and geometric transforms are paired across arms.
Prediction-refining is OFF
(`unified_eval` never applies it) and `normal` is absent from BUSI seg test slices. In the legacy
BUSI-only experiment, the centralized model is trained/evaluated on the same frozen folds; this
centralized path is not used for the multi-dataset study.

**Metrics:** seg → Dice, IoU, Sensitivity, Specificity, Precision; cls → Accuracy, macro-F1,
balanced accuracy, dynamic per-class precision/recall, OvR-macro AUC. All outputs and analysis are
isolated by `dataset`; 3-class BUSI probabilities are never mixed with 7-class ISIC probabilities.

**Analysis:** `src/experiments/analyze.py` accepts two or three result/prediction sets and writes
dataset-isolated summaries, per-client deltas, pooled AUC, exploratory paired Wilcoxon diagnostics,
and `report.html`. Client × CV-fold pairs are not independent inferential replicates, so their
p-values must not support confirmatory significance claims; use independent seeds or paired OOF
sample-level inference for that purpose.
