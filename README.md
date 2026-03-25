# Receipt Forgery Detection System

A multi-signal AI pipeline that detects tampered receipts by combining a CNN image classifier, a pixel-level segmentation model, rule-based physical artifact detection, and a OCR-driven semantic validation layer — all served through a FastAPI backend.

---

## The Problem This Solves

Receipt fraud is a real and growing problem in expense reimbursement, insurance claims, and corporate auditing. A fraudster typically:

- Inflates a line-item price (e.g., ₹120 → ₹1,200)
- Replaces the grand total with a sticker or reprinted label
- Modifies only 1–2 digits so the change is visually subtle
- Photographs the altered receipt to make it look like an original scan

Standard image classifiers fail on this problem because **the forgery region is tiny** (often <5% of pixels), the rest of the receipt is authentic, and high-quality edits leave no visible compression artifacts. A system that only asks "does this look forged?" will miss cases where the numbers are wrong but the image looks clean.

This system addresses these challenges by combining visual signals with semantic validation. Instead of relying only on appearance, it uses OCR to extract and approximate the financial structure of the receipt, and then verifies whether values such as line items, taxes, and totals are logically consistent.

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
## Dataset

### Source  
**SROIE 2019** — a publicly available scanned receipt dataset from Kaggle (`urbikn/sroie-datasetv2`).

- 973 receipt images with OCR bounding boxes and structured entity annotations (e.g., TOTAL, DATE, COMPANY)  
- Each image has a corresponding `.txt` file with word-level OCR boxes in the format:  
  `x1,y1,x2,y1,x2,y2,x1,y2,text`  
- These OCR boxes are used as the basis for forgery generation  

---

### Forgery Generation  

Since no large-scale ground-truth forged dataset exists, forgeries are generated programmatically by modifying OCR-aligned regions.

**Forgery types:**
- `price_replace` — inflate or partially modify line-item amounts  
- `total_replace` — modify the grand total field  
- `date_replace` — alter date values  
- `smudge` — apply blur/noise to obscure values  
- `copy_move` — copy-paste regions within the same image  

**Realism techniques:**
- Background-aware patching (avoids visible rectangles)  
- Slight ink variation and positional jitter  
- Noise, blur, and brightness/contrast adjustments  
- JPEG recompression to simulate scan artifacts  

---

### Final Dataset  

Total images: 1,903  
- Real: 973 (label = 0)  
- Forged: 930 (label = 1) — 43 skipped (no matching OCR boxes)  
- Masks: 930 (pixel-level tamper masks for all forged images)

Class balance: ~1.05:1 (real:forged)  
Dataset size: ~1.04 GB  

**Forgery breakdown:**
- total_replace: 330 (35.5%)
- price_replace: 310 (33.3%)
- date_replace: 290 (31.2%)

**Splits (stratified):**
- Train: 1,426 (697 forged / 729 real)
- Val: 286  
- Test: 191  


## Training

- Trained on Google Colab (T4 GPU)  
- Notebooks:
  - `SROIE_Forgery_Training_Colab.ipynb` — data + baseline
  - `Improved_Detection_v2.ipynb` — final models + evaluation  

---

### Models

**1. EfficientNet-B3 (Classifier)**
- Task: real vs forged classification  
- ImageNet pretrained backbone + custom head  
- Input: 320×320  
- Key techniques: differential LR, OneCycleLR, class weighting, label smoothing, TTA  
- Forgery-aware augmentations (JPEG artifacts, dropout, distortions)

**2. U-Net (Segmentation)**
- Task: pixel-level tamper localization  
- EfficientNet-B3 encoder (pretrained)  
- Loss: Focal + Dice + BCE  
- Output converted to tamper score using thresholded area

**3. OCR Anomaly Detector**
- Task: detect statistical/semantic inconsistencies from OCR  
- 13 engineered features (amounts, dates, text structure, confidence)  
- Model: Isolation Forest (trained on real receipts only)  
- Combines rule-based checks + anomaly score  

---

### Final Pipeline

- Visual classifier (global decision)  
- Segmentation model (local evidence)  
- OCR anomaly detector (semantic validation)  
- Outputs fused via cross-signal decision logic  

## Inference Pipeline

Multi-signal pipeline combining visual, semantic, and anomaly-based detection.

### Components

- **Physical checks (OpenCV):** brightness blobs, borders, ELA, texture, lighting inconsistencies  
- **Segmentation (U-Net):** localizes tampered regions  
- **Classifier (EfficientNet):** global real vs forged prediction  
- **OCR + logic:** validates totals, taxes, and field consistency  
- **Anomaly detector:** flags statistical irregularities in OCR features  

---

### Key Idea

- Combines **where (segmentation)** + **what (OCR logic)** + **how (physical artifacts)**  
- Cross-signal reasoning links spatial tampering with financial fields (e.g., TOTAL region)

---

### Decision Logic

- Rule-based overrides for strong signals (logic errors, clear artifacts)  
- Multi-signal agreement boosts confidence  
- Final ensemble score combines all signals  
- Outputs: **clean / suspicious / forged**  

## Results

- Dataset: 1,903 receipts (973 real, 930 forged)

### Performance (Test Set)

| Model | AUC | Accuracy |
|---|---|---|
| Classifier | 0.67 | 53% |
| **Ensemble** | **0.81** | **76%** |

### Key Insight

- Single model performance is limited  
- Combining **visual + segmentation + OCR reasoning** significantly improves detection  
- Ensemble gives **+13.7 pp AUC gain** over classifier alone  





