# Research notes — Federated multi-task breast-cancer ultrasound (FedPer)

Working notes for a future paper. Covers (1) objectives, (2) the system built, (3) the
experimental protocol and fairness controls, and (4) the results of the two comparison runs
(`runs/comparison_2seg_2cls`, `runs/comparison_4seg_4cls`) with interpretation.

---

## 1. Objectives

1. Extend the centralized multi-task model (Aumente-Maestro et al.) — simultaneous tumor
   **segmentation** + **classification** on breast ultrasound (Curated BUSI, 450 imgs:
   222 benign / 164 malignant / 64 normal, 128×128) — to a **federated** setting.
2. Use a **task-decoupled FedPer** scheme: every client owns a **single task** (seg *or* cls);
   only the **encoder** is federated (FedAvg), while each client keeps a **personalized head/decoder**
   that is never aggregated. The final artifact is **one shared encoder + N personalized heads**
   (a generalist feature extractor + task/site-specific heads), motivated by a future expansion to
   **multiple, heterogeneous datasets per client**.
3. Quantify, under strictly controlled conditions, whether federation helps each participant by
   comparing three setups: **Local-only (floor)**, **Federated (FedPer, proposed)**, and
   **Centralized MTL (upper bound)**.

---

## 2. System built (methods section material)

Backbone: **MTnnUNet** — 5-level U-Net encoder–decoder (widths `[32,64,128,256,320]`), deep
supervision (4 seg outputs), classification head fusing `encoder5`, `upsample(bottleneck)` and
`decoder5` features. Uses **InstanceNorm** (no BatchNorm running stats → no stat-sync issue in FL).

| Component | File | Role |
|---|---|---|
| Federated partition | `src/dataset/federated_partition.py` | image-level `StratifiedKFold` → per-task **Dirichlet(α)** partition → single **master CSV** (`fold, client_id, task, split`). `normal` dropped from seg clients; per-client val carved for early stopping. Deterministic/frozen. |
| Model split | `src/federated/model_split.py` | shared = `encoder1..5` + `bottleneck` (federated); personalized = decoders, upsamples, outputs, classifier (local). |
| Local trainer | `src/federated/local_trainer.py` | per-task local train/eval (seg → Dice, cls → Focal). |
| Federated client | `src/federated/client.py` | Flower `NumPyClient`; persists personalized state + optimizer + `best.pt` to disk per client; resolves device per worker; **`standalone` toggle** (ignore the global encoder → local-only baseline through the same code path). |
| Federated server | `src/federated/server.py` | `FedAvg` subclass aggregating **only the encoder**, weighted by `num_examples × task_weight[task]`; logs mean val loss / best round. |
| Federated orchestrator | `src/training_federated.py` | per-fold Flower simulation + per-client test via `unified_eval`. |
| Centralized baseline | `src/training_centralized.py` | one MTL model trained from the **master CSV's** fold (unique images, `split != test`) — same folds, leak-free — evaluated on the same per-client test slices. |
| Unified evaluation | `src/federated/unified_eval.py` | single shared evaluator used by **all** setups (comparable numbers); no prediction-refining; dumps cls per-image probabilities for pooled AUC. |
| Analysis | `src/experiments/analyze.py` | summary, paired Wilcoxon, per-client deltas, pooled AUC, and a self-contained **HTML report** (boxplots per metric×setup, per-client delta bars, AUC bars). |

Framework: **Flower** (`flwr[simulation]==1.30`) + **FedAvg**, single-machine simulation.
Key config (`federated:` in `config.yaml`): `n_clients_seg/cls`, `rounds`, `local_epochs`,
`dirichlet_alpha` (high ≈ IID), `task_weights{seg,cls}`, `oversampling{seg,cls}`.

---

## 3. Experimental protocol & fairness controls

- **Frozen, shared partition.** All three setups read the SAME master CSV; runners **error** if it
  is missing rather than regenerating (generate once with `federated_partition`). The fold split is
  at the **image level** with a fixed seed, so the test *images* per fold are identical regardless of
  how many clients the pool is split into.
- **Same evaluator, no refino.** `unified_eval` is the single scorer for every setup; prediction-
  refining (overlap seg↔cls) is **OFF** (it cannot run in FedPer, where seg and cls live on different
  clients) — it is left as a separate ablation.
- **`normal` excluded from segmentation evaluation** in all setups (empty-mask images otherwise give
  a free Dice = 1 and inflate centralized).
- **Local-only = federated pipeline with aggregation disabled** (`standalone: True`): identical
  partition, model, and training budget (`rounds × local_epochs` local epochs), differing only in
  whether the encoder is shared. This is the controlled "does federation help?" baseline.
- **Variance source:** CV folds × clients (no extra seeds). Paired **Wilcoxon signed-rank** compares
  federated vs local-only across (fold, client) pairs (n = 8 for 2+2, n = 16 for 4+4).
- **Metrics:** seg → Dice, IoU, Sensitivity, Specificity, Precision; cls → Accuracy, macro-F1,
  balanced accuracy, per-class precision/recall, OvR-macro **AUC** (per-slice, NaN when a class is
  absent, and **pooled** across clients per fold).

> ⚠️ **Centralized run reuse (important).** The SAME centralized run is used in **both** comparisons
> (its rows are byte-identical, n = 16 = 4 folds × 4 client-slices). It was scored on the 4+4 client
> slices. Because the fold split is image-level and deterministic, the centralized **aggregate covers
> the same test images** as the 2+2 federated/local runs, so it is a valid **aggregate upper-bound
> reference** in both. It is NOT a per-client paired comparator (Wilcoxon is federated-vs-local only).

---

## 4. Results

### 4.1 Centralized MTL — upper bound (n = 16; identical in both experiments)

| Task | Dice | IoU | Sens | Spec | Prec |
|---|---|---|---|---|---|
| seg | **0.712 ± 0.051** | 0.607 ± 0.050 | 0.774 ± 0.070 | 0.976 ± 0.007 | 0.766 ± 0.051 |

| Task | Acc | macro-F1 | bal-acc | AUC | pooled-AUC |
|---|---|---|---|---|---|
| cls | **0.742 ± 0.076** | 0.719 ± 0.095 | 0.720 ± 0.097 | 0.882 ± 0.052 | 0.881 |

### 4.2 Experiment A — 2 seg + 2 cls clients (fed/local n = 8)

**Classification**

| Metric | Federated | Local-only | Δ (fed−loc) | Wilcoxon p |
|---|---|---|---|---|
| Accuracy | 0.667 ± 0.071 | 0.567 ± 0.058 | **+0.100** | 0.055 |
| macro-F1 | 0.621 ± 0.070 | 0.499 ± 0.082 | **+0.122** | 0.078 |
| balanced-acc | 0.613 ± 0.063 | 0.507 ± 0.086 | **+0.106** | 0.055 |
| AUC | 0.805 ± 0.031 | 0.700 ± 0.032 | **+0.104** | **0.0078** ✓ |
| pooled-AUC | 0.780 | 0.695 | +0.085 | — |

**Segmentation**

| Metric | Federated | Local-only | Δ (fed−loc) | Wilcoxon p |
|---|---|---|---|---|
| Dice | 0.615 ± 0.029 | 0.622 ± 0.045 | −0.007 | 0.64 |
| IoU | 0.490 ± 0.025 | 0.498 ± 0.040 | −0.008 | 0.64 |
| Sens / Spec / Prec | ~0.79 / 0.962 / 0.631 | ~0.80 / 0.961 / 0.626 | ≈ 0 | > 0.6 |

Also significant: cls precision (malignant, class 1) +0.194, p = 0.039 ✓.

### 4.3 Experiment B — 4 seg + 4 cls clients (fed/local n = 16)

**Classification**

| Metric | Federated | Local-only | Δ (fed−loc) | Wilcoxon p |
|---|---|---|---|---|
| Accuracy | 0.605 ± 0.100 | 0.547 ± 0.102 | +0.058 | 0.090 |
| macro-F1 | 0.502 ± 0.131 | 0.433 ± 0.145 | +0.068 | 0.25 |
| balanced-acc | 0.512 ± 0.108 | 0.467 ± 0.116 | +0.045 | 0.26 |
| AUC | 0.729 ± 0.090 | 0.669 ± 0.122 | **+0.060** | **0.039** ✓ |
| pooled-AUC | 0.714 | 0.670 | +0.044 | — |

**Segmentation**

| Metric | Federated | Local-only | Δ (fed−loc) | Wilcoxon p |
|---|---|---|---|---|
| Dice | 0.530 ± 0.045 | 0.521 ± 0.042 | +0.010 | 0.27 |
| IoU | 0.401 ± 0.041 | 0.395 ± 0.038 | +0.007 | 0.46 |

Also significant: cls recall (benign, class 0) +0.166, p = 0.0056 ✓.

---

## 5. Key findings (for the paper)

1. **Federation helps classification, robustly for AUC.** Federated > Local-only on every cls metric
   in both experiments, and **AUC is significant in both** (p = 0.008 at 2+2, p = 0.039 at 4+4).
   Interpretation: the cls head is small and the encoder is the bottleneck, so sharing the encoder
   exposes each cls client to more diverse anatomy than its own shard.
2. **Federation is neutral for segmentation.** No seg metric is significant in either experiment
   (all p > 0.27); deltas are within ±0.01. Interpretation: the **personalized decoder dominates**
   segmentation quality, so encoder sharing neither helps nor hurts — consistent with the FedPer
   premise that decoders are best kept local.
3. **Centralized is a clear upper bound** on every metric (cls Acc 0.742, seg Dice 0.712), and the
   gap **widens with more clients** as each shard shrinks.
4. **More clients → less data each → lower absolute performance** for both decentralized setups
   (e.g., federated cls Acc 0.667 → 0.605; seg Dice 0.615 → 0.530 from 2+2 to 4+4), and a **smaller
   federation margin for classification** (Acc Δ +0.100 → +0.058). Notably, the seg delta flips from
   slightly negative to slightly positive — federation becomes marginally useful for seg only when
   shards are very small.
5. **The minority `normal` class is the hardest** (low, high-variance class-2 precision/recall across
   all setups), and **federation improves its detection** (e.g., recall_class_2 0.31 → 0.47 at 2+2).

**One-line story:** *A shared encoder with personalized heads (FedPer) recovers most of the
classification gap to a centralized model — significantly improving AUC over training in isolation —
while leaving segmentation essentially unchanged, because segmentation quality is governed by the
personalized decoder.*

---

## 6. Caveats / things to verify before publication

- **Hyperparameters of these runs** (rounds, local_epochs, `task_weights`, `dirichlet_alpha`,
  oversampling) are stored in each run dir's `config.yaml`. Current config uses `task_weights
  seg:4 / cls:1` (encoder aggregation weighted 4× toward seg) and Dirichlet α = 100 (≈ IID) — both
  should be reported and ideally ablated.
- The **centralized run is shared across both experiments** (see §3 box). Re-running a centralized
  model whose test slices exactly match the 2+2 partition would remove any residual doubt, though the
  aggregate is already over identical images.
- Statistical power is modest (n = 8 / 16 pairs); trends consistent across both experiments lend
  credibility, but **non-AUC cls gains are suggestive, not significant** (0.05 < p < 0.10 several
  times) — consider more folds/seeds or more clients for tighter intervals.
- `federated.patience` is currently **inert** (no server-side early termination); each client keeps
  its own `best.pt`. Worth stating explicitly in the methods.

## 7. Possible next experiments

- **Non-IID** partitions (low Dirichlet α) — the regime where federation is expected to help most.
- **Heterogeneous datasets per client** (e.g., BUSI vs BUS-UCLM) + **leave-one-client-out** encoder
  generality probe (hook already planned).
- **`task_weights` ablation** (seg/cls balance of the shared encoder) and **rounds × local_epochs**.
- **Prediction-refining as a post-hoc cross-client step** (pair a seg client with a cls client on the
  same image) — recovers the refino that single-client FedPer cannot apply.
