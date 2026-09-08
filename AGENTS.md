# AGENTS.md

Project brief for coding agents working in this repository. This is the **canonical** file:
Codex reads it natively, and `CLAUDE.md` pulls it in with an `@AGENTS.md` import. Edit this file,
not the copy.

<!-- GUARD-RAILS:START -->
## Non-negotiables

Violating any of these silently invalidates frozen artifacts or already-published results.

- **`Curated_BUSI_preprocessing.py` resizes with `INTER_NEAREST`. Do NOT "fix" it to `INTER_AREA`** —
  it would change the curated images and invalidate the frozen federated partition and every
  result derived from it.
- **Never build dataset paths by hand.** Derive them from `src/dataset/paths.py`
  (`processed_dir()`, `mapping_file()`, `partition_file()`).
- **Never hand-edit `mapping.csv` or `federated_mapping.csv`.** Regenerate them — they are
  deterministic given the same config. Paths inside them are relative to the repo root, so moving
  a dataset folder invalidates both.
- **Never regenerate the federated partition without explicit intent.** Every experiment arm is
  comparable only because all of them read the same frozen file.
- **Do not apply `sigmoid` before the DICE criterion** — the criterion applies it internally.
- **`numpy` must stay `<2`** (pandas/monai ABI).
- **Do not call a p-value from this study "statistically significant".** They come from paired
  Wilcoxon over 8 non-independent client × fold pairs on a single seed, and the pipeline tags them
  `exploratory_only_non_independent_client_fold_pairs`.
<!-- GUARD-RAILS:END -->

## Skills

Task-specific playbooks have a **single canonical copy** under `.agents/skills/`. Claude Code
discovers them through `.claude/skills/` and Codex through `.codex/skills/`; both discovery trees
contain relative symlinks to the canonical directories, maintained by `scripts/sync_agent_assets.py`.
Never edit a discovery link: edit `.agents/skills/<name>/` and load the skill matching the task:

| Skill | Use it when |
|---|---|
| `dataset-preprocessing` | Running or debugging preprocessing for an already-supported dataset |
| `dataset-onboarding` | Adding a brand-new dataset and writing its `*_preprocessing.py` |
| `federated-study` | Running `study_runner`, resuming arms, or interpreting the analysis tables |
| `error-investigator` | A traceback, crash, failing test, or unexplained numeric result |
| `grill-me` | Stress-testing a plan or design before building it |
| `python-plotting` | Creating, reviewing, or standardizing Python charts and scientific figures |

After adding, renaming, or removing a skill, run `python -m scripts.sync_agent_assets`. Content
edits inside an existing canonical skill are visible immediately through both links; run
`python -m scripts.sync_agent_assets --check` to validate the complete discovery structure.

## Commands

```bash
# Preprocess a dataset -> data/<dataset>/<variant>/. Each script refuses to run unless
# `data.dataset` in config.yaml names the dataset it handles (it would otherwise overwrite
# another dataset's variant), so set that first.
python -m src.dataset.Curated_BUSI_preprocessing   # data.dataset: Curated_BUSI
python -m src.dataset.ISIC_2018_preprocessing      # data.dataset: ISIC_2018

# Train (CV=1 selects deterministic holdout; CV>=2 selects cross-validation)
python -m src.training_multitask      # segmentation + classification
python -m src.training_segmentation   # segmentation only
python -m src.training_classification # classification only

# Federated (FedPer) training — see "Federated training" section below
python -m src.dataset.federated_partition   # build data/<dataset>/federated/federated_mapping.csv
python -m src.dataset.federated_partition --legacy  # only to reproduce frozen single-dataset BUSI
python -m src.training_federated            # run the federated simulation
python -m scripts.smoke_federated --setup both  # paired 2-round CPU end-to-end validation

# Comparison experiment (Federated vs Local-only vs Centralized) — see section below
python -m src.training_federated            # federated  (federated.standalone: False)
python -m src.training_federated            # local-only (set federated.standalone: True first)
python -m src.training_centralized          # centralized MTL (from the same master CSV)
python -m src.experiments.analyze --results <3 *_test_results.csv> --preds <3 *_cls_predictions.csv> --out runs/comparison

# Multi-arm study driver (see the `federated-study` skill)
python -m src.experiments.study_runner --dry-run
python -m src.experiments.study_runner --smoke --seed-profile operational
python -m scripts.benchmark_training_runtime  # paired FP32-sequential vs BF16-dual GPU gate
python -m src.experiments.study_runner --seed-profile operational

# Tests, and the agent-asset consistency check
python -m unittest discover -v
python -m scripts.sync_agent_assets --check
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

In the federated path `training.alpha` is not used. A `single_task` client optimizes Dice **or** the
classification criterion, never a weighted sum. A `multi_task` client optimizes
`L = Σ_t λ_t · L_t` with `λ_t = aggregation.task_weights[t] / Σ task_weights`, evaluated only over
the samples each task actually supervises (per-sample `has_mask` / `has_label` flags). A task with
no supervised sample in a batch is **omitted, never contributed as zero** — an all-zero mask is a
legitimate target for BUSI's `normal` class, so a zero term and an absent term differ.

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
| `training.CV` | `1` = deterministic holdout; `>=2` = number of cross-validation folds |
| `training.holdout_test_size` | Test fraction when `training.CV=1` (default `0.30`; ignored otherwise) |
| `training.alpha` | Weight on seg loss (0=cls only, 1=seg only) |
| `data.root` / `data.dataset` / `data.variant` | Select the dataset folder + preprocessed variant (see "Dataset layout") |
| `data.classes` | Which classes to include; determines binary vs. multiclass mode |
| `data.seg_exclude_classes` | Classes dropped from the segmentation task (empty masks); `[normal]` for BUSI |
| `data.oversampling` | Enables deterministic oversampling in train folds |
| `loss.inversely_weighted` | Weight deep supervision outputs by `1/(n+1)` |

## Federated training (FedPer)

A federated variant trains the multi-task model across clients using **Flower** (`flwr[simulation]`)
simulation + **FedAvg**, following the **FedPer** scheme: the encoder is shared/federated, while each
client keeps a **personalized head** that is never aggregated.

`datasets.<name>.client_topology` decides how tasks map to clients, and the default is
`single_task` — every existing result was produced under it:

- **`single_task`** (default) — each client owns **one task** (seg *or* cls). Each task's pool is
  partitioned independently, so the same image ends up in a seg client **and** in a *different* cls
  client. Train/test stay disjoint per fold, but one silo's data is duplicated across two silos.
- **`multi_task`** — one client owns **several tasks over the same images**, so an image belongs to
  exactly one silo. Requires `fold_strategy: stratified` and the same `n_clients` count for every
  task. ISIC cannot use it: its seg and cls pools are disjoint image sets.

### Pipeline

The default federated configuration is multi-dataset: Curated BUSI (1-channel ultrasound) and
ISIC 2018 (3-channel dermoscopy). `encoder1` is a personalized modality stem; the shared trunk is
`encoder2..5 + bottleneck`. Set `share_stem: True` only for compatible single-dataset experiments.

1. `src/dataset/federated_partition.py` — builds a separate multi-dataset master. BUSI uses the
   historical image-level stratified folds, or a stratified 70/30 holdout when `CV=1`. ISIC
   segmentation uses a shuffled outer split; ISIC classification uses a grouped stratified outer
   split, group-preserving client allocation, and group-preserving train/val, so
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
6. `src/federated/server.py` — client weight is `client_factor × Σ_t task_weight[t] · mass_t / n`,
   where `mass_t` is how many of the client's samples supervise task `t`. A single-task client
   reports its whole slice under its own task, so the sum collapses to `task_weight[task]` and
   reproduces the historical weights exactly. `hierarchical` mode normalizes within each dataset and
   then applies `dataset_weights`; `flat` mode is the sample-weighted ablation. Every round also
   records the **pairwise cosine between client trunk deltas** (per block and overall) into
   `aggregation_history.json` — no artifact persists a client's post-fit trunk, so that matrix
   cannot be recovered after the run.
7. `src/training_federated.py` — per-fold orchestrator: validates shared shapes before Flower, runs
   the simulation, persists `global_shared.pt`, and evaluates every federated client with the same
   final global trunk plus its latest personalized state. Local-only evaluates each latest full
   local model at the identical round budget. Saves `{setup}_test_results.csv` and
   `{setup}_cls_predictions.csv` under the timestamped run directory.

The final artifact is **one shared trunk + N personalized stems/heads** (not a single global model).
Server-side early stopping is diagnostic only; final comparison uses the same configured last-round
budget in both arms.

### Local training budget

`federated.local_training.mode` decides how much each client trains per round, and it is the single
most consequential hyperparameter in the study:

- `steps` — every client takes `steps_per_round` optimizer steps regardless of its size. With
  clients spanning 115 to ~3,000 effective samples this hands the small dataset ~26× more epochs
  than the large one over a 50-round run.
- `epochs` — every client makes `local_epochs` passes over its own data, so effort tracks size.

When comparing arms, only compare pairs that share a budget mode; otherwise the comparison measures
the budget, not the federation.

### Key federated config (`federated:` section)

| Key | Effect |
|---|---|
| `datasets` / `n_clients.<dataset>` | Active datasets and clients per dataset/task |
| `datasets.<name>.client_topology` | `single_task` (default) or `multi_task` (one client, several tasks, same images) |
| `share_stem` | Federate the input stem; must be false for mixed 1ch/3ch runs |
| `local_training.mode` | `steps` (fixed `steps_per_round`) or `epochs` (`local_epochs` passes) |
| `aggregation.mode` | `hierarchical` (main) or `flat` (FedAvg-style ablation) |
| `aggregation.client_weighting` | `uniform` or `num_examples`, inside the hierarchical mode |
| `aggregation.dataset_weights` | Dataset-level mixture weights; 1:1 gives exact 50/50 participation |
| `datasets.<name>.class_weighting` | `none`, `balanced_fold` (main), or `balanced_local` |
| `rounds` / `local_epochs` | Federated rounds × local epochs per round |
| `dirichlet_alpha` | Train partition skew (high = IID, low = non-IID) |
| `aggregation.task_weights.{seg,cls}` | Per-task weight inside each dataset aggregate |
| `oversampling.{seg,cls}` | Per-task override of `data.oversampling` |
| `device` | `auto` / `cpu` / `cuda` (resolved per worker) |
| `runtime.federated.client_resources.num_gpus` | `0.5` = 2 clients/GPU; `1.0` = sequential |
| `runtime.federated.ray_num_cpus` | Caps Ray concurrency to bound memory use |
| `training.precision` | `bf16` for new CUDA runs; missing field preserves legacy FP32 |
| `runtime.inference_batch_size` | Batched final evaluation without changing per-sample rows |

### Notes / gotchas

- `flwr` requires `protobuf>=4`, which is incompatible with the old pinned `tensorboard==2.10.0`;
  tensorboard was removed (it was a dead import). **numpy must stay `<2`** (pandas/monai ABI).
- Running clients in parallel can exhaust system RAM (and GPU memory); keep `ray_num_cpus` /
  `client_resources` conservative. The defaults run one client at a time.
- Oversampling inflates a client's effective training set: BUSI classification clients go from
  120/148 raw rows to 371/453 effective. Reason about epochs from the effective count, which is
  what `metadata.yaml` records as `effective_train_examples`.
- **Oversampling is refused on a `multi_task` client.** Replicating rows to balance classes would
  also duplicate the segmentation targets of the boosted classes. Rebalance with
  `datasets.<name>.class_weighting` instead.
- A `multi_task` client emits **one result row per task**, so `analyze.py` (which keys on
  `dataset/fold/client_id/task`) works unchanged. Those two rows share images and a model, so they
  are even less independent than the single-task pairs — the `exploratory_only_*` tag still applies.
- Test slices differ between the two topologies, so a topology comparison is an aggregate
  dataset/task comparison, **not** a per-client paired delta.

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

**Reading classification numbers:** ISIC's NV class is 66.9% of the labelled pool, so always report
balanced accuracy and macro-F1 next to accuracy — 0.669 is what a constant "NV" predictor scores,
and 0.143 is a uniform guess over 7 classes. High AUC with floor-level accuracy is not a bug; it is
the signature of a class-weighted loss under an insufficient training budget.

**Analysis:** `src/experiments/analyze.py` accepts two or three result/prediction sets and writes
dataset-isolated summaries, per-client deltas, pooled AUC, exploratory paired Wilcoxon diagnostics,
and `report.html`. Client × CV-fold pairs are not independent inferential replicates, so their
p-values must not support confirmatory significance claims; use independent seeds or paired OOF
sample-level inference for that purpose. With `CV=1`, Wilcoxon is disabled and comparisons are
tagged `descriptive_only_single_holdout`.

### Client topology arms

`studies/example_multi_dataset.yaml` (the default manifest) runs the `multi_task` BUSI topology in
its base arms (`primary`, `local_only`, `ablation_flat`) and carries two arms
(`single_task_primary`, `single_task_local`) that declare `partition_variant: single_task` to run
the historical one-task-per-client topology. An arm may override partition-defining fields **only**
when it declares a variant; the base arms keep the original guard and share one partition per
seed, so adding a topology never invalidates completed runs. A variant's master is nested one
directory deeper under the same `partition_template`.

## Multi-arm studies

`src/experiments/study_runner.py` drives several arms over one frozen partition per
`(seed, partition_variant)` from a manifest in `studies/`. See the `federated-study` skill for the workflow, the artifact layout under
`runs/studies/<study_id>/`, and the rules for reading the analysis tables.

A manifest pins its scientific protocol (`CV`, precision, topology, client counts, budget,
aggregation) in `config_overrides` instead of inheriting it from `src/config.yaml`, which is a
working default that drifts between experiments. Precision and `cuda_benchmark` are scientific hash
inputs; `runtime` and NVML telemetry are operational and excluded. See
`docs/TRAINING_ACCELERATION.md`.
