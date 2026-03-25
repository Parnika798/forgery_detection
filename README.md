# Receipt Forgery Detection System

A production-grade, multi-signal AI pipeline that detects tampered receipts by combining a CNN image classifier, a pixel-level segmentation model, rule-based physical artifact detection, and a semantic OCR reasoning engine — all served through a FastAPI backend.

---

## The Problem This Solves

Receipt fraud is a real and growing problem in expense reimbursement, insurance claims, and corporate auditing. A fraudster typically:

- Inflates a line-item price (e.g., ₹120 → ₹1,200)
- Replaces the grand total with a sticker or reprinted label
- Modifies only 1–2 digits so the change is visually subtle
- Photographs the altered receipt to make it look like an original scan

Standard image classifiers fail on this problem because **the forgery region is tiny** (often <5% of pixels), the rest of the receipt is authentic, and high-quality edits leave no visible compression artifacts. A system that only asks "does this look forged?" will miss cases where the numbers are wrong but the image looks clean.

This system solves the problem by combining visual signals with **semantic reasoning** — it checks whether the numbers on the receipt actually add up.

---

## System Architecture

```
                    ┌──────────────────────────────┐
                    │      Input: Receipt Image      │
                    └──────────────┬───────────────-┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         │                         │                         │
         ▼                         ▼                         ▼
┌──────────────────┐   ┌───────────────────┐   ┌────────────────────┐
│ EfficientNet-B3  │   │  U-Net Segmenter  │   │  Tesseract OCR     │
│ Image Classifier │   │ Tamper Locator    │   │  + Feature Extract │
│                  │   │                   │   │                    │
│ clf_prob ∈ [0,1] │   │ seg_mask (H×W)    │   │ 13 OCR features    │
│ TTA (h-flip avg) │   │ tamper_area %     │   │ IsolationForest    │
└────────┬─────────┘   └────────┬──────────┘   └─────────┬──────────┘
         │                      │                         │
         │                      ▼                         ▼
         │           ┌──────────────────┐    ┌───────────────────────┐
         │           │ Physical Artifact│    │ Semantic / Logical     │
         │           │ Detection        │    │ Validation             │
         │           │                  │    │                        │
         │           │ ELA, bright blob │    │ extract_structured_   │
         │           │ edge borders     │    │ fields()               │
         │           │ brightness jump  │    │ validate_totals()      │
         │           │ texture variance │    │ validate_critical_     │
         │           └────────┬─────────┘    │ fields()               │
         │                    │              └──────────┬─────────────┘
         └────────────────────┼──────────────────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │  Cross-Signal Reasoner   │
                 │                         │
                 │ seg_mask ∩ TOTAL bbox   │
                 │ blob ∩ TOTAL bbox       │
                 │ ELA + low OCR conf      │
                 │ logic mismatch + visual │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │  Layered Decision Engine │
                 │                         │
                 │ L1: Definitive rules    │
                 │ L2: Multi-signal corr.  │
                 │ L3: Ensemble fallback   │
                 │ L4: Clean veto          │
                 └────────────┬────────────┘
                              │
                              ▼
         ┌──────────────────────────────────────┐
         │  Verdict: clean / suspicious / forged │
         │  + Confidence % + Human-readable flags│
         │  + Tamper heatmap overlay (base64)    │
         │  + Structured field parse (items, tax)│
         └──────────────────────────────────────┘
```

---

## Dataset

### Source
**SROIE 2019** — a publicly available scanned receipt dataset from Kaggle (`urbikn/sroie-datasetv2`).

```
973 images
  ├── train/img/      ← receipt scans (.jpg)
  ├── train/box/      ← OCR bounding boxes (8 coords + text per line)
  ├── train/entities/ ← structured fields: TOTAL, DATE, COMPANY
  └── test/  (same structure)
```

Every image has a corresponding `.txt` file with word-level OCR bounding boxes in the format `x1,y1,x2,y1,x2,y2,x1,y2,text`. This is the key input to the forgery generator.

### Forgery Generation

Since no large-scale ground-truth forged receipt dataset exists, forgeries were generated programmatically. The forgery engine reads the OCR box coordinates and patches specific fields directly into the image.

**Implemented forgery types:**

| Type | Target | Method |
|---|---|---|
| `price_replace` | Line-item amounts | Inflate 5×–50×, or modify 1–2 digits (`partial_replace`) |
| `total_replace` | Grand total field | Find box after TOTAL keyword, inflate |
| `date_replace` | Date fields | Swap year with a different year |
| `smudge` | Any financial field | Motion blur + noise — makes value unreadable |
| `copy_move` | Any financial field | Paste patch from elsewhere in the same image |

**Realism improvements in the forgery generator:**
- Local background patch sampling instead of flat-colour fill (prevents visible rectangle)
- Randomised ink colour (slightly off-black, simulates faded/uneven printer ink)
- ±3px position jitter when rendering replacement text
- Gaussian noise injection matching surrounding texture
- JPEG re-compression artifact simulation on the whole image (60–88% quality)
- Soft mask edges via Gaussian blur (simulates annotation imprecision)
- Brightness/contrast variation (simulates uneven scan lighting)

**Final dataset composition (confirmed from notebook output):**

```
Total images : 1,903
  Real       : 973   (label = 0)
  Forged     : 930   (label = 1)   ← 43 skipped (no matching OCR boxes)
  With masks : 930   ← binary pixel-level tamper mask per forged image

Class balance : 1.05:1 (real:forged)
Dataset size  : ~1,040 MB on Google Drive

Forgery type breakdown (from actual run):
  total_replace   : 330  (35.5%)
  price_replace   : 310  (33.3%)
  date_replace    : 290  (31.2%)
```

**Splits (stratified by label):**

```
Train : 1,426 images  (697 forged / 729 real)
Val   :   286 images
Test  :   191 images
```

---

## Training

All training ran on **Google Colab T4 GPU**. Two notebooks cover the full pipeline:

| Notebook | Role |
|---|---|
| `SROIE_Forgery_Training_Colab.ipynb` | v1 baseline — data download, forgery generation, initial training |
| `Improved_Detection_v2.ipynb` | v2 — improved classifier, U-Net, anomaly detector, evaluation |

---

### Model 1: EfficientNet-B3 Image Classifier

**Task:** Binary classification — real (0) vs forged (1)

**Architecture:**
```
EfficientNet-B3 encoder (ImageNet pretrained, global average pool)
  └─ BatchNorm1d(1536)
  └─ Dropout(0.4)
  └─ Linear(1536 → 512)
  └─ ReLU
  └─ BatchNorm1d(512)
  └─ Dropout(0.3)
  └─ Linear(512 → 2)

Total parameters: 11,488,298
```

**Key training decisions and why each was made:**

| Setting | v1 | v2 (this run) | Why it matters |
|---|---|---|---|
| Input resolution | 224 px | **320 px** | Sub-pixel tamper regions need higher resolution to survive resizing |
| Learning rate strategy | Single LR 2e-4 | **Differential LR** enc=2e-5, head=2e-4 | With a single LR, the encoder barely updates; differential LR lets the pretrained backbone fine-tune at its own pace |
| Scheduler | CosineAnnealingLR | **OneCycleLR** (per-batch steps) | Faster warmup + better final convergence |
| Max epochs | 25 | **60** (early stop patience=15) | More training time with safety net |
| Forged class weight | 1.5× | **2.61×** (computed from class counts) | Higher penalty for missing forgeries — recall matters more than precision in fraud detection |
| Label smoothing | None | **0.05** | Prevents overconfident wrong predictions |
| TTA at inference | No | **Horizontal flip average** | Simple, cheap, consistent small improvement |

**Augmentation pipeline (forgery-aware):**
```python
A.ImageCompression(quality_lower=55, quality_upper=95, p=0.5)  # JPEG re-save artifacts
A.CoarseDropout(max_holes=4, max_height=24, max_width=80, fill_value=240, p=0.35)  # copy-paste blocks
A.GridDistortion(num_steps=3, distort_limit=0.05, p=0.2)   # scanner warp
A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=4, p=0.4)
A.GaussNoise, A.GaussianBlur, A.CLAHE, A.RandomBrightnessContrast, A.HueSaturationValue
```

The JPEG compression and CoarseDropout augmentations are specifically chosen to make the model robust to the artifacts produced by the forgery generator.

---

### Model 2: U-Net Segmenter

**Task:** Pixel-level tamper localisation — produce a spatial heatmap of *where* the receipt was modified

**Architecture:** U-Net with EfficientNet-B3 encoder (ImageNet pretrained)

| Setting | v1 | v2 |
|---|---|---|
| Encoder | MobileNet-V2 | **EfficientNet-B3** |
| Epochs | 15 | **40** |
| Loss | Dice + BCE | **Focal + Dice + BCE** |

**Combined loss function:**
```python
def seg_loss(pred, target):
    return (0.40 * focal_loss(pred, target) +   # handles sparse masks
            0.40 * dice_loss(pred, target)  +   # enforces shape quality
            0.20 * bce_loss(pred, target))      # pixel-level gradient
```

Focal loss is critical here because tampered regions are sparse — typically <7% of pixels — and without it, the model learns to predict "untampered" everywhere and still achieves low BCE loss.

**Score conversion at inference:**
```python
tamper_area = (seg_raw > 0.35).mean()                   # pixel fraction above threshold
seg_score   = 0.0 if tamper_area < 0.05 else            # noise floor: ignore <5%
              min((tamper_area - 0.05) * 10, 1.0)
```

The 5% noise floor was added because the U-Net produces low-level activations on even clean receipts; ignoring sub-5% area prevents false positives.

---

### Model 3: OCR Anomaly Detector

**Task:** Statistical and semantic detection of unusual financial patterns from Tesseract OCR output

**Feature set — 13 features (up from 7 in v1):**

| Feature | Type | Fraud signal |
|---|---|---|
| `max_amount` | Numeric | Inflated amounts exceed real-world thresholds |
| `n_amounts` | Count | Zero = OCR failure or heavy tampering |
| `n_dates` | Count | Missing dates on a receipt |
| `valid_date_fmt` | Binary | Forged dates often have incorrect format |
| `has_currency` | Binary | Edited receipts often lose $ / ₹ symbol |
| `line_count` | Count | Sparser text = heavier image manipulation |
| `total_to_max_ratio` | Ratio | total > largest item is mathematically suspicious |
| `amount_std` | Numeric | High variance = inflated line items (new) |
| `unique_amount_ratio` | Ratio | Repeated amounts = copy-paste artefact (new) |
| `has_total_keyword` | Binary | Missing TOTAL keyword = keyword erasure (new) |
| `avg_word_conf` | Numeric | Low Tesseract confidence = font inconsistency (new) |
| `text_density` | Numeric | Chars per line — forged text is sparser (new) |
| `round_number_ratio` | Ratio | Suspiciously round numbers (new) |

**Model:** IsolationForest trained **exclusively on real receipts** (258 real samples from the 500-sample OCR extraction run)

```python
IsolationForest(n_estimators=300, contamination=0.15, max_samples='auto', random_state=42)
```

Training on real-only samples means the forest learns the normality manifold. Forged receipts are then statistical outliers relative to it. If forged receipts were included in training, the model would incorporate their patterns into "normal."

**Three scoring layers:**
1. Rule-based semantic checks (amount threshold, confidence, keyword presence)
2. Format/consistency checks (date format, currency symbol, total/max ratio)
3. Isolation Forest score (normalised decision function)

---

## Inference Pipeline (API)

At inference, five modules run in parallel and their results are fused through a reasoning engine. The execution order matters: `_text()` runs **before** `anomaly_detector.score()` so logical validation is computed once and cached.

### Physical Tampering Detection

Five OpenCV checks on the raw image:

| Check | Method | Score |
|---|---|---|
| Bright blob / sticker | Morphological opening on thresholded image; aspect ratio + position heuristic | +0.35 to +0.65 |
| Rectangular border lines | Hough line transform; ≥3H + 3V = border of pasted region | +0.25 |
| Brightness discontinuity | Per-quadrant mean brightness; jump >28 levels | +0.20 |
| ELA (Error Level Analysis) | JPEG re-encode at Q75, absdiff; residual >5% = paste | +0.35 |
| Texture variance | Local variance map; unnaturally smooth 0.5–8% patch | +0.25 to +0.40 |

---

### Semantic / Logical Validation

This is the core innovation added in v4. It detects receipts where the numbers don't add up — even when the image looks visually clean.

**`extract_structured_fields(text)`** — classifies each OCR line into a semantic bucket:

```
Priority order (first match wins per line):
  total    ← "Grand Total", "Net Payable", "Amount Due"
  subtotal ← "Sub Total", "Subtotal", "Taxable Amount"
  tax      ← "SGST", "CGST", "GST", "VAT", "Tax"
  discount ← "Discount", "Offer", "Coupon"
  payment  ← "Cash", "Card", "Paid", "Tendered", "Advance"
  skip     ← "Qty", "Rate", "HSN", "GSTIN", "Date", "Invoice No."
  orphan   ← pure-number line (no alphabetic tokens)
  item     ← everything else with an amount
```

**`_join_split_lines(text)`** — thermal printers often emit keyword and amount on separate lines. This pre-processor merges them before field extraction:
```
"TOTAL."         →  merged →  "TOTAL.  5,800.00"
"5,800.00"
```

**`validate_totals(structured)`** — tries five arithmetic checks:

| Attempt | Formula | Covers |
|---|---|---|
| A | items + tax − discount ≈ total | Full breakdown |
| B | subtotal + tax − discount ≈ total | Subtotal-based |
| C | items + tax ≈ total | No discount info |
| D | sum(items) ≈ total | Tax-inclusive pricing |
| E | subtotal ≈ total | No items parsed |

All attempts failing → `score = 0.55` → triggers definitive **forged** verdict.
Tolerance: `max(total × 2%, ₹2)` handles rounding drift.

**`validate_critical_fields(text, structured)`** — catches:
- Multiple total-family fields that disagree with each other
- Subtotal exceeding total (mathematically impossible)
- Single tax line exceeding total (mathematically impossible)
- Digit prepend/append between subtotal and total (e.g., "2000" → "52000")

---

### Cross-Signal Reasoning

Links spatial evidence with semantic evidence. A tamper heatmap overlapping a TOTAL field is not two independent signals — it is one piece of evidence that means something specific.

| Link | What it checks | Score |
|---|---|---|
| A | Seg mask overlaps TOTAL/PAYABLE/SUBTOTAL bounding box | +0.55 to +0.85 |
| B | Physical blob/sticker overlaps TOTAL keyword bbox | +0.75 |
| C | ELA region + low OCR confidence on critical keyword | +0.45 |
| D | TOTAL bbox word ≠ parsed total value | +0.35 |
| E | Logical mismatch + any spatial signal | +0.40 to +0.65 |

---

### Decision Engine

Five layers, checked in order, first match wins:

```
Layer 1 — Definitive rules:
  cross_score ≥ 0.55          → "forged"
  text_score  ≥ 0.25          → "forged"   (logical mismatch)
  phys_score  ≥ 0.45          → "forged"   (strong visual artifact)

Layer 2 — Multi-signal corroboration:
  strong signals (seg≥0.35, phys≥0.45, cross≥0.45, clf≥0.90) ≥ 2 → "forged"
  moderate signals (seg≥0.20, phys≥0.30, anom≥0.30,
                    text≥0.25, cross≥0.30) ≥ 3 → "forged"

Layer 3 — Ensemble fallback:
  ensemble_score ≥ optimal_threshold → "forged"
  ensemble = 0.25·clf + 0.20·seg + 0.15·anom + 0.20·phys
           + 0.15·text + 0.25·cross

Layer 4 — Clean veto:
  all signals below noise floor → "clean"

Layer 5 — Suspicious:
  any 1 moderate signal or ensemble ≥ threshold × 0.5 → "suspicious"
```

The optimal threshold (0.43 in this run) is found by sweeping `[0.25, 0.75]` on the validation set and selecting the value maximising F1 on the forged class.

---

## Results

All numbers below come directly from notebook cell outputs — no estimates.

### Dataset Statistics

| Metric | Value |
|---|---|
| Total images | 1,903 |
| Real receipts | 973 |
| Forged receipts | 930 |
| Skipped (no OCR box match) | 43 |
| Dataset size | ~1,040 MB |
| Training platform | Google Colab T4 GPU |
| Tesseract version | 4.1.1 |

### Classifier Training

| Metric | Value |
|---|---|
| Best checkpoint | Epoch 22 |
| Best validation AUC | 0.6632 |
| Model parameters | 11,488,298 |
| Forged class weight (computed) | 2.61× |

### U-Net Segmentation

| Metric | Value |
|---|---|
| Mean tamper area — forged | **6.91%** of pixels |
| Mean tamper area — real | 1.94% of pixels |

The 3.5× difference in tamper area between real and forged confirms the U-Net is learning to localise tampered regions, not just flagging the whole image.

### Anomaly Detector

| Metric | Value |
|---|---|
| Training samples (real only) | 258 |
| Total OCR samples extracted | 500 |
| Mean anomaly score — forged | 0.178 |
| Mean anomaly score — real | 0.170 |

Note: The anomaly detector scores are very close between real and forged in this run (0.178 vs 0.170), which reflects the fundamental challenge: the SROIE dataset uses receipt images from Malaysia with currency symbols (RM/MYR) that the feature extractor partially misses. The anomaly detector contributes to the ensemble but is not the strongest individual signal.

### Ensemble Evaluation (Test Set, n=191)

**AUC comparison:**

| Model | AUC |
|---|---|
| Classifier alone (with TTA) | 0.6715 |
| **Ensemble (clf + U-Net + anomaly)** | **0.8085** |

The ensemble delivers a **+13.7 percentage point AUC improvement** over the classifier alone. This is the primary justification for the multi-signal architecture — no single model is sufficient.

**Classification report — Classifier only (threshold=0.50):**

```
              precision    recall  f1-score   support

        Real       0.59      0.30      0.39        98
      Forged       0.51      0.78      0.62        93

    accuracy                           0.53       191
   macro avg       0.55      0.54      0.51       191
```

**Classification report — Ensemble (threshold=0.43):**

```
              precision    recall  f1-score   support

        Real       0.80      0.71      0.75        98
      Forged       0.73      0.81      0.77        93

    accuracy                           0.76       191
   macro avg       0.76      0.76      0.76       191
```

**Summary of gains from ensemble:**

| Metric | Classifier only | Ensemble | Gain |
|---|---|---|---|
| AUC | 0.6715 | 0.8085 | +13.7 pp |
| Accuracy | 53% | 76% | +23 pp |
| Real precision | 0.59 | 0.80 | +21 pp |
| Forged recall | 0.78 | 0.81 | +3 pp |
| Forged F1 | 0.62 | 0.77 | +15 pp |
| Optimal threshold | 0.50 | 0.43 | Tuned on val F1 |

The classifier alone barely beats random (AUC 0.67, accuracy 53%). The ensemble brings accuracy to 76% and AUC to 0.81. The additional OCR semantic reasoning layer in the API (v4.1) catches logically inconsistent receipts that neither the classifier nor U-Net detects, particularly for high-quality digital edits.

---

## API Reference

### `POST /scan`

Accepts a receipt image, runs all five analysis modules, returns a full verdict.

**Request:** `multipart/form-data` with field `file` (JPG/PNG/WEBP)

**Response:**
```json
{
  "verdict":     "forged",
  "is_forged":   true,
  "confidence":  87.3,

  "clf_prob":    67.1,
  "seg_score":   0.0,
  "tamper_area": 0.0,
  "anom_score":  34.2,
  "phys_score":  85.0,
  "text_score":  55.0,
  "cross_score": 45.0,
  "n_signals":   3,

  "flags": [
    "TOTAL MISMATCH — all 2 check(s) failed: sum_items(2000.00)≠total(5800.00)",
    "Sticker/label detected over lower receipt area",
    "Rectangular border lines (42H/268V) — patch edges",
    "Brightness discontinuity (78 levels)"
  ],

  "ocr": {
    "n_amounts":  5,
    "n_dates":    1,
    "total":      5800.0,
    "avg_conf":   74.2,
    "raw_text":   "DEF Electronics\nHeadphones  1,200.00\n...",
    "amounts":    [350.0, 450.0, 1200.0, 5000.0, 5800.0],
    "dates":      ["12/03/2024"],
    "ocr_error":  null
  },

  "structured": {
    "items":          [1200.0, 450.0, 350.0],
    "taxes":          [],
    "discounts":      [],
    "payments":       [5000.0],
    "subtotal":       null,
    "total":          5800.0,
    "inferred_total": false
  },

  "overlay_b64": "<base64-encoded JPEG tamper heatmap>"
}
```

The `structured` field exposes the semantic parse — which amounts were classified as items, payments, taxes, etc. The `flags` list is human-readable and tells reviewers exactly what the system found.

---

### `POST /scan-debug`

**Diagnostic endpoint — no ML models required.** Runs OCR and logical validation only. Use this whenever a verdict looks wrong.

```bash
curl -X POST http://localhost:8000/scan-debug \
     -F "file=@receipt.jpg" | python -m json.tool
```

Returns the full line-by-line classification of every OCR line (`raw_fields`), the score from each individual validator, and a `diagnosis` string explaining why OCR succeeded or failed. This was added specifically to debug cases where `ocr_error` was silently returning `{}` and all scores were 0.

---

### `GET /health`

```json
{ "status": "ok", "models_loaded": true }
```

---

## Model Artifacts

Four files required for inference:

| File | Description |
|---|---|
| `classifier_final.pth` | EfficientNet-B3 weights + training metadata (best_auc, test_auc, ensemble_auc, threshold) |
| `unet_final.pth` | U-Net segmenter weights |
| `anomaly_model.pkl` | Fitted IsolationForest + StandardScaler |
| `config.json` | img_size=320, backbone, seg_encoder, optimal_threshold=0.43 |

---

## Versioning

| Version | What changed |
|---|---|
| **v1 (training)** | Baseline: EfficientNet-B3 (224px, single LR) + MobileNetV2 U-Net (15ep) + 7-feature anomaly detector |
| **v2 (training)** | 320px input, differential LR (enc=2e-5, head=2e-4), OneCycleLR, forged weight 2.61×, label smoothing, EfficientNet-B3 U-Net (40ep), Focal+Dice+BCE loss, 13-feature anomaly detector, TTA, threshold sweep |
| **v3 (API)** | Cross-signal reasoning module; layered decision engine |
| **v4 (API)** | Semantic logical validation: `extract_structured_fields`, `validate_totals`, `validate_critical_fields`, logical anomaly score; result caching between modules |
| **v4.1 (API)** | `_join_split_lines` for thermal printer split lines; `_infer_total_from_items` fallback; multi-PSM OCR retry (PSM 6→11→4→3); `ocr_error` surfaced instead of swallowed; `/scan-debug` endpoint; `_KW_PAYMENT` bucket to stop "cash paid" lines from contaminating item sums |

---

## What the Numbers Mean for a Reviewer

The classifier alone (AUC 0.67, accuracy 53%) is not usable in production. It performs barely above random on this dataset, which reflects how hard the problem is — the forgery generator creates realistic edits, and the receipt images vary widely in quality, format, and language.

The ensemble (AUC 0.81, accuracy 76%) is significantly better, but the most important addition is the **semantic reasoning layer** in v4/v4.1 that operates entirely outside the trained models. When a receipt shows items summing to ₹2,000 but a total of ₹5,800, no amount of visual training data will catch that — it requires reading the text, parsing the structure, and doing arithmetic. That layer has no AUC metric because it fires in a deterministic, rule-based way on real receipts, but it closes the gap that the trained models leave open.

---

## Resume / LinkedIn

**Technical framing:**
> Built a multi-signal receipt forgery detection pipeline combining EfficientNet-B3 image classification (AUC 0.67 standalone → 0.81 ensemble), U-Net pixel-level tamper segmentation localising edits to <7% of image area, and a custom OCR semantic reasoning engine that validates arithmetic consistency of financial fields — achieving a 23-point accuracy improvement over the classifier alone and serving explainable verdicts via FastAPI.

**Impact framing:**
> Designed an end-to-end AI fraud detection system for receipt tampering that integrates computer vision, OCR-based logical validation, and physical artifact detection into a layered decision engine — the semantic layer catches visually clean digital edits by detecting when stated totals don't match line-item sums, addressing the core failure mode of image-only classifiers.

**Concise (CV bullet / LinkedIn headline):**
> Receipt forgery detection: EfficientNet-B3 + U-Net segmentation + OCR arithmetic validation → ensemble AUC 0.81 (+14pp over classifier alone), FastAPI backend, explainable flag-level output.
