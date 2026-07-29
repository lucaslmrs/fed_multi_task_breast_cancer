# ISIC 2018 — how the reference paper used it, and what our local copy actually contains

**Purpose of this file.** Notes for a future agent/reader on how ISIC 2018 is used for multi-task
(classification + segmentation) learning, extracted from the reference paper in
`papers/ISIC_2018/`. It also records what our local copy of the dataset actually contains, because
**several of the paper's claims do not survive contact with the real data**. Read the
"Conflicts" section before designing anything on top of this dataset.

Every claim below is tagged:

- **[PAPER]** — asserted by the paper. Reported as the paper reports it, not as fact.
- **[VERIFIED]** — measured directly on `data/ISIC_2018/raw/` (2026-07-14). Reproducible.
- **[CONFLICT]** — the paper and the data disagree. These are the load-bearing ones.
- **[INFERRED]** — my reading, not stated by the paper. Flagged so it is not mistaken for a source claim.

---

## 0. Bottom line (read this first)

1. **The paper's ISIC multi-task setup is not reproducible as literally described.** It conflates
   two *disjoint* subsets of ISIC 2018: Task 1 (2,594 images, has masks, no labels) and Task 3 /
   HAM10000 (10,015 images, has labels, no masks). **[VERIFIED]** the ID intersection is **exactly
   zero**. No image in the official release carries both a segmentation mask and a diagnosis label,
   so a single image cannot feed both heads.
2. **The paper's per-class table is mislabeled.** The counts are genuinely HAM10000's, but the class
   *names* for three of the seven classes are attached to the wrong counts (details in §5.3). Do not
   copy that table.
3. **The paper's preprocessing section is ~6 sentences long** and omits nearly everything needed to
   reproduce it: no normalization, no resize/interpolation method, no mask-resize handling, no split
   procedure, no seed. Treat §3 below as the *complete* extent of what is documented.
4. **A naive split of Task 3 leaks.** **[VERIFIED]** 10,015 images map to only 7,470 lesions; 44.9%
   of images share a lesion with another image. The paper's implied random 80/20 split puts
   near-duplicates of the same lesion on both sides. This is the same failure mode our Curated BUSI
   work removes via SSIM.
5. **[INFERRED] The disjointness that breaks the paper is *not* a blocker for our federated design.**
   See §7 — our FedPer setup gives each client exactly one task, so a disjoint seg-pool and cls-pool
   is natural. It *is* a blocker for our centralized MTL baseline.

---

## 1. Paper identification

| Field | Value |
|---|---|
| Title | Multi-Task Deep Learning for Simultaneous Classification and Segmentation of Cancer Pathologies in Diverse Medical Imaging Modalities |
| Authors | Maryem Rhanoui; Khaoula Alaoui Belghiti; Mounia Mikram |
| Venue | *Onco* 2025, 5, 34 (MDPI) |
| DOI | 10.3390/onco5030034 |
| License | CC BY 4.0 (open access) |
| Dates | Received 2025-05-04; accepted 2025-07-06; published 2025-07-11 |
| Local file | `papers/ISIC_2018/Multi-Task Deep Learning for Simultaneous Classification and Segmentation of Cancer Pathologies in Diverse Medical Imaging M.pdf` (18 pages) |

**Scope caveat.** ISIC 2018 is only *one of four* datasets in this paper (the others: TCGA-LGG brain,
PANDA prostate, SIIM-ACR pneumothorax). ISIC-specific content is therefore thin — most of the
methodology is shared across all four. Relevant sections: §5.1.1 (data), §5.2 (preprocessing +
environment), §5.3 (ablations), §6 (results), Tables 1–3.

---

## 2. What the paper says about the ISIC 2018 data

**[PAPER]** §5.1.1: ISIC 2018 is a large-scale dermoscopic image dataset from the International Skin
Imaging Collaboration. It is described as containing **2,594 RGB images** at a resolution of
**2166 × 3188**. Seven lesion classes are listed: actinic keratoses, basal cell carcinoma, benign
keratosis, dermatofibroma, melanocytic nevi, vascular skin lesions, melanoma.

**[PAPER]** Table 1 ("Publicly Available Medical Image Datasets used in our tests"):

| Dataset | Organ | Number of Images | Input Size | Image Type | Classes |
|---|---|---|---|---|---|
| ISIC 2018 | Skin Lesion | **10,015** | (224, 224) | Dermoscopic image | 7 |
| TCGA-LGG | Brain Tumor | 3,929 | (256, 256) | MRI | 2 |
| PANDA | Prostate Cancer | 10,616 | (128, 128) | Digital Histopathology | 6 |
| SIIM-ACR | Pneumothorax | 12,047 | (224, 224) | X-Ray | 2 |

Note the paper's own §5.1.1 (2,594) and Table 1 (10,015) **already disagree with each other**. See §5.1.

**[PAPER]** Table 2, skin-lesion rows (class distribution, train/test/total) — **reproduced only so
it can be corrected in §5.3; do not use as-is**:

| # | Class as named by the paper | Train | Test | Total |
|---|---|---|---|---|
| 0 | Actinic keratoses | 262 | 65 | 327 |
| 1 | Basal cell carcinoma | 412 | 102 | 514 |
| 2 | Benign keratosis | 879 | 220 | 1,099 |
| 3 | Dermatofibroma | 92 | 23 | 115 |
| 4 | Melanocytic nevi | 891 | 222 | 1,113 |
| 5 | Vascular skin lesi[ons] | 5,341 | 1,341 | 6,705 |
| 6 | Melanoma | 114 | 28 | 142 |

Totals sum to 10,015, matching Table 1 and confirming the underlying source is HAM10000 (Task 3).

**[INFERRED]** The split ratio is never stated in prose, but every row is ~80/20 (262/65, 412/102,
879/220 …), and it is **stratified by class**. There is no separate validation set in the table,
despite early stopping being enabled (§3) — what early stopping monitors is **not documented**.

---

## 3. Preprocessing and training setup — the complete documented extent

This is genuinely all the paper provides (§5.2). It is thin; the gaps in §6 are not oversights in
these notes.

**[PAPER] Preprocessing:**
- **Resize** to 224 × 224 (from Table 1; the resize *procedure* and interpolation are not stated,
  and mask resizing is never mentioned).
- **Data augmentation**, applied "during training" to expand the training set: **rotation**,
  **flipping**, **shearing**. No probabilities, ranges, or library given. (Figure 7 shows examples.)
- **Not mentioned at all:** normalization/standardization, ImageNet mean-std scaling (despite
  ImageNet-pretrained backbones), color constancy / shade-of-gray (standard for dermoscopy), hair
  removal (standard for dermoscopy), mask binarization threshold, handling of the 7→binary mask
  question, deduplication, lesion grouping, random seed.

**[PAPER] Training / environment:**

| Setting | Value |
|---|---|
| Hardware | Google Colab, NVIDIA T4 GPU |
| Epochs | 50 |
| Batch size | 32 |
| Early stopping | Enabled (monitored quantity + patience not stated) |
| Optimizer | Adam |
| Learning rate | 0.001 (described as Adam's default) |
| LR schedule | None mentioned |

**[PAPER] Class imbalance:** the classification loss uses scikit-learn's `'balanced'` class-weighting
strategy — weights from inverse class frequency. This matters for ISIC: HAM10000 is severely skewed
(NV alone is 67% of the data).

---

## 4. Model, loss and reported results

### 4.1 Architecture

**[PAPER]** §4.1. A UNet whose encoder is replaced by an **ImageNet-pretrained backbone** — two
variants tested: **VGG16** and **MobileNetV2**. Four blocks: feature extraction, bottleneck,
classification task, segmentation task. Hard parameter sharing. The encoder is shared; the
classification head is fully-connected layers hanging off the encoder path; the decoder (with skip
connections to the encoder) produces the mask.

### 4.2 Loss

**[PAPER]** §4.2:

```
L_total = λ_seg · L_seg + λ_cls · L_cls        with  λ_seg = 5,  λ_cls = 1
```

- `L_seg` = **binary cross-entropy** (pixel is lesion vs. background). Stated to affect the
  **entire network** (encoder + decoder).
- `L_cls` = **weighted categorical cross-entropy** (7 classes), with sklearn `'balanced'` weights.
  Stated to affect **only the encoder section**.

> **[INFERRED] Non-obvious design choice worth noting:** restricting the classification gradient to
> the encoder is a real architectural decision, not a throwaway detail — it makes the decoder a pure
> segmentation specialist. Our `MTnnUNet` does **not** do this: its classification head consumes
> `encoder5`, `upsample(bottleneck)` *and* `process_decoder5`, so our cls gradient reaches decoder
> features. Worth remembering before comparing our alpha to their λ.

### 4.3 Task-weight ablation

**[PAPER]** §5.3.2. Ratios seg:cls of [1:1], [2:1], [5:1], [10:1], [1:2], [1:5], [1:10] were run for
50 epochs on a **reduced subset** of data. Best: **[5:1]**, giving segmentation 0.9366 / classification
0.8021. Beyond [5:1] (e.g. [10:1]) segmentation gained negligibly while classification degraded;
raising the classification weight destabilized segmentation.

> **[INFERRED] Cross-reference to our config:** our `training.alpha = 0.85` means
> `0.85·seg + 0.15·cls`, i.e. a **5.67:1** ratio — remarkably close to their empirically-selected
> 5:1, arrived at independently. Mild external corroboration of our alpha.

### 4.4 Reported results (skin lesion only)

**[PAPER]** Table 3:

| Setup | Skin-lesion classification | Skin-lesion segmentation |
|---|---|---|
| Mono-task VGG16 | 0.74 | — |
| Mono-task MobileNetV2 | 0.83 | — |
| Mono-task UNet | — | 0.88 |
| Mono-task Mask-RCNN | — | 0.94 |
| **Multi-task, VGG16 encoder** | 0.83 | **0.95** |
| **Multi-task, MobileNetV2 encoder** | **0.86** | **0.95** |

Headline claim: multi-task lifts skin classification from 0.74 → 0.86 and segmentation to 0.95 Dice.

**[CONFLICT]** §6 also reports a 5-fold cross-validation with a paired t-test giving Dice
**0.995 ± 0.001** vs. a 0.829 ± 0.034 baseline. This is irreconcilable with the 0.95 Dice in Table 3,
and the text never says which dataset(s) the 5-fold applies to. Treat the 0.995 figure as
uninterpretable.

---

## 5. Conflicts between the paper and the real data

This section is why this file exists.

### 5.1 [CONFLICT] 2,594 vs. 10,015 — two different, disjoint subsets

**[VERIFIED]** exact inventory of `data/ISIC_2018/raw/`:

| Folder | Contents | Count |
|---|---|---|
| `ISIC2018_Task1-2_Training_Input` | dermoscopic images | **2,594** |
| `ISIC2018_Task1_Training_GroundTruth` | segmentation masks | **2,594** |
| `ISIC2018_Task1-2_Validation_Input` / `ISIC2018_Task1_Validation_GroundTruth` | images / masks | 100 / 100 |
| `ISIC2018_Task1-2_Test_Input` / `ISIC2018_Task1_Test_GroundTruth` | images / masks | 1,000 / 1,000 |
| `ISIC2018_Task2_Training_GroundTruth_v3` | attribute masks (5 per image → 2,594 × 5) | 12,970 |
| `ISIC2018_Task2_Validation_GroundTruth` | 100 × 5 | 500 |
| `ISIC2018_Task2_Test_GroundTruth` | 1,000 × 5 | 5,000 |
| `ISIC2018_Task3_Training_Input` | dermoscopic images (HAM10000) | **10,015** |
| `ISIC2018_Task3_Training_GroundTruth/…csv` | one-hot labels, 7 classes | **10,015 rows** |
| `ISIC2018_Task3_Validation_Input` / GT csv | images / labels | 193 / 193 |
| `ISIC2018_Task3_Test_Input` / GT csv | images / labels | 1,512 / 1,512 |
| `ISIC2018_Task3_Training_LesionGroupings.csv` | image → lesion_id map | 10,015 rows |

So the paper's "2,594" is **Task 1** (segmentation) and its Table-1 "10,015" is **Task 3** (HAM10000
classification). They are different releases serving different challenge tasks.

**[VERIFIED] The two sets are completely disjoint.** ID format is identical (`ISIC_XXXXXXX`), so the
comparison is valid:

```
Task 1 (masks)  ID range: ISIC_0000000 .. ISIC_0016072   (2,594 images)
Task 3 (labels) ID range: ISIC_0024306 .. ISIC_0034320   (10,015 images)
Intersection: 0        Task1 not in Task3: 2,594        Task3 not in Task1: 10,015
```

**Consequence:** there is **no image with both a mask and a diagnosis label**. A conventional
multi-task model — one image → (mask, label) — cannot be trained on ISIC 2018 out of the box. The
paper does not acknowledge this, and never explains how it obtained both supervisions.

**[VERIFIED] Scope note on those ranges.** The two blocks above are the *training* splits — the
slice the paper used. Ingest val/test too (3,694 seg / 11,720 cls, which is what
`ISIC_2018_preprocessing.py` does) and the picture is no longer two blocks but three: seg occupies
**0..24191 AND 36066..36347**, with cls sitting in the gap between them at **24306..36064**. Still
zero intersection, still clean gaps (24191→24306 and 36064→36066) — but seg's naive min..max
(0..36347) now *spans* cls's range, so quoting min..max invites the wrong conclusion. State the
disjointness as a fact about the sets. `ISIC_2018_preprocessing.py` asserts it (id uniqueness over
all 15,414 rows) rather than relying on the intervals.

**[INFERRED]** Plausible explanations, none confirmed by the text: (a) they trained the two heads on
different subsets and reported per-task numbers separately; (b) they used pseudo-labels or
pseudo-masks; (c) segmentation was evaluated on Task 1 while Table 2's class table describes only the
classification stream. Any reproduction must **choose and state** a strategy — it cannot be copied.

### 5.2 [CONFLICT] Resolution claim is wrong for both subsets

**[PAPER]** "each with a resolution of 2166 × 3188 pixels".

**[VERIFIED]** (sampled 400 files per folder):

- `Task1-2_Training_Input`: **highly variable** — 76 distinct sizes in 400 files. Most common:
  767×576, 2048×1536, 1504×1129, 3072×2304.
- `Task3_Training_Input`: **uniformly 600×450** (HAM10000 is pre-standardized) — 1 distinct size in
  400 files.
- `Task1_Training_GroundTruth` masks: variable, matching their source images; `uint8`, values
  strictly `{0, 255}`.

2166×3188 matches neither. Any resize plan must handle Task 1's heterogeneity (aspect ratios vary
widely — squashing to 224×224 distorts) while Task 3 needs none.

### 5.3 [CONFLICT] Table 2's class names are attached to the wrong counts

**[VERIFIED]** true HAM10000 / Task 3 distribution, computed from the official GT csv:

| Code | Class | True count | Paper's Table 2 says this count is… | Verdict |
|---|---|---|---|---|
| NV | Melanocytic nevi | **6,705** | "Vascular skin lesions" | ❌ wrong |
| MEL | Melanoma | **1,113** | "Melanocytic nevi" | ❌ wrong |
| BKL | Benign keratosis | **1,099** | Benign keratosis | ✅ |
| BCC | Basal cell carcinoma | **514** | Basal cell carcinoma | ✅ |
| AKIEC | Actinic keratoses | **327** | Actinic keratoses | ✅ |
| VASC | Vascular lesions | **142** | "Melanoma" | ❌ wrong |
| DF | Dermatofibroma | **115** | Dermatofibroma | ✅ |
| | **Total** | **10,015** | 10,015 | ✅ |

The counts are exactly HAM10000's, so the data is right — but classes 4, 5 and 6 have their names
rotated onto the wrong rows. Read literally, the paper claims vascular lesions are the majority class
(67%) of a dermoscopy dataset, which is false; NV is. **Use the CSV column order
(`MEL, NV, BCC, AKIEC, BKL, DF, VASC`), never the paper's Table 2.**

### 5.4 [CONFLICT] "slightly exceed 1200 images"

The abstract describes the datasets as "relatively small" and "slightly exceed 1200 images", which
contradicts its own Table 1 (10,015 / 3,929 / 10,616 / 12,047 — every one an order of magnitude
larger). Unexplained.

### 5.5 [VERIFIED] Lesion-level leakage — the paper's split is unsafe

Not a paper/data conflict but an omission with real consequences.

```
Task 3: 10,015 images  →  only 7,470 unique lesions
  1,956 lesions (26.2%) have >1 image;  max 6 images per lesion
  4,501 images (44.9%) belong to a multi-image lesion
```

`ISIC2018_Task3_Training_LesionGroupings.csv` (columns: `image`, `lesion_id`,
`diagnosis_confirm_type`) exists precisely to prevent this. The paper never mentions it, and its
implied random stratified 80/20 split therefore places multiple images **of the same physical lesion**
in both train and test — inflating test accuracy.

> **[INFERRED] Direct relevance to this repo:** this is structurally the same problem our Curated
> BUSI addresses (330 duplicates removed via SSIM). Any ISIC work here should **split on `lesion_id`,
> not on `image`** — otherwise we would reproduce, in a new dataset, exactly the bias our own
> contribution exists to eliminate.

---

## 6. What the paper does not answer

Ranked by how much they block reproduction:

1. **How were mask and label supervision combined at all**, given the zero intersection? (§5.1) — blocker.
2. **Which images were used for the segmentation results** in Table 3? Task 1 train? Task 1 test?
3. **What did early stopping monitor**, with no validation set defined in Table 2?
4. Were input images normalized (ImageNet mean/std, [0,1], nothing)?
5. How were masks resized to 224×224, and re-binarized after interpolation?
6. Augmentation ranges/probabilities; were mask and image transformed jointly? (they must be, for seg)
7. Random seed / number of runs. Table 3 reports single point estimates with no variance.
8. Reconciliation of Table 3 (0.95 Dice) with §6's 5-fold claim (0.995 Dice).
9. Was the 7-class problem ever reduced to binary anywhere?

---

## 7. Implications for this repository

Our project is federated multi-task on breast ultrasound (`CLAUDE.md`). Mapping this paper onto it:

**The disjointness is survivable for us — and this is the key insight.** Our FedPer design gives each
client **exactly one task** (`seg` *or* `cls`, see `src/dataset/federated_partition.py`). A dataset
where the mask-bearing images and the label-bearing images are different images is therefore a
**natural fit**: `seg` clients take Task 1, `cls` clients take Task 3, and the "same image used by
clients of different tasks" property simply does not arise. **[INFERRED]** ISIC 2018 may in fact be a
*better* motivating example for FedPer than for the centralized MTL the paper attempts.

**But it breaks our centralized baseline.** `src/training_centralized.py` trains one MTL model on
`split != test` and needs `image`, `mask` *and* `label` per row. With ISIC there is no such row. The
centralized "upper bound" of our comparison would need redefinition (e.g. a two-headed model fed
alternating single-task batches) — this is a **design decision, not a port**.

**Concrete blockers already known in our code** (all confirmed in the previous reorg work):

| Blocker | Where | Note |
|---|---|---|
| Grayscale forced | `src/dataset/BUSI_dataset.py` — `cv2.imread(path, 0)` | ISIC is RGB; `model.sequences: 1` |
| Hardcoded label map | `src/dataset/BUSI_dataset.py` | `benign/malignant/normal` → 0/1/2 |
| Mask convention | `BUSI_dataset.py` does `mask[mask == 255] = 1` | **[VERIFIED]** ISIC masks are also `{0,255}` uint8 → compatible |
| No preprocessing script | — | ISIC has none; `data/ISIC_2018/` is `raw/` + `archives/` only |
| `seg_exclude_classes` | `src/config.yaml` | `[normal]` is BUSI-specific; ISIC needs `[]` |

**Layout note.** Per `CLAUDE.md`, a preprocessed ISIC variant must land at
`data/ISIC_2018/<variant>/` with `images/`, `masks/`, `mapping.csv`, and paths resolve through
`src/dataset/paths.py` (`data.dataset: ISIC_2018`). Do not hardcode.

**Status (2026-07-14):** the preprocessing now exists — `src/dataset/ISIC_2018_preprocessing.py`
(run with `data.dataset: ISIC_2018`). It ingests both tasks into one `mapping.csv` tagged with
`task`/`official_split`/`lesion_id`, sidestepping §5.1 by never pretending a row has both
supervisions. `data/ISIC_2018/processed_128/` holds 15,414 rows (3,694 seg + 11,720 cls). The
training-side blockers in the table above are still open.

**Reusable from the paper regardless:** the 5:1 seg:cls weighting (corroborates our `alpha=0.85`),
balanced inverse-frequency class weights (relevant — NV is 67% of Task 3), and rotation/flip
augmentation (we already do flips + rotation; we do not do shear).

---

## 8. Reproduction checklist, if ISIC is pursued

1. **Decide the multi-task strategy first** (§5.1). Nothing else matters until this is settled.
2. **Split on `lesion_id`** from `ISIC2018_Task3_Training_LesionGroupings.csv`, never on `image` (§5.5).
3. Use the **GT csv column order** for classes, not the paper's Table 2 (§5.3).
4. Handle **Task 1's variable aspect ratios** deliberately (pad vs. squash); Task 3 needs no resize care (§5.2).
5. **Re-binarize masks** after any interpolation — resize with nearest-neighbour, as
   `Curated_BUSI_preprocessing.py` already does.
6. Note the **official val/test splits exist** (Task 1: 100/1,000; Task 3: 193/1,512) — the paper
   ignores them and re-splits training 80/20. Using the official ones would be more comparable to
   other literature.

---

## 9. Provenance of the [VERIFIED] numbers

Measured on 2026-07-14 against `data/ISIC_2018/raw/` at repo `base-implementation`. Text extracted
from the PDF with `pypdf` (18 pages, ~55k chars). Reproduce with a script that: counts files per
task folder; compares `Task1-2_Training_Input` stems against the `image` column of the Task 3 GT
csv; derives class counts via `idxmax(axis=1)` on the one-hot GT columns; and value-counts
`lesion_id` in the groupings csv. Resolution figures come from `cv2.imread(...).shape` over the
first 400 files of each folder — a sample, not a census, so treat "76 distinct sizes" as a lower
bound on Task 1's heterogeneity.

**Attribution.** Paper content summarized from Rhanoui, Alaoui Belghiti & Mikram (2025),
*Onco* 5(3):34, CC BY 4.0. Reported claims are the authors'; the [CONFLICT]/[INFERRED] readings are
this analysis, not theirs.
