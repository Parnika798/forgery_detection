# Receipt Forgery Detection System

A multi-signal AI system for detecting tampered receipts using **image classification, pixel-level segmentation, physical artifact detection, and OCR-based logical validation**, served via a FastAPI API.

---

## Highlights

- Detects **subtle forgeries (<5% pixel tampering)**
- Combines **visual + spatial + semantic signals**
- Achieves **0.81 AUC (vs 0.67 baseline)** on test set
- Handles cases where **image looks clean but numbers are inconsistent**

---
<img width="1361" height="852" alt="image" src="https://github.com/user-attachments/assets/dadf7220-7d76-4f91-af6c-31bad197355b" />
<img width="1261" height="903" alt="image" src="https://github.com/user-attachments/assets/c4c88bd0-856f-4095-8972-b7c42673350c" />
<img width="1266" height="915" alt="image" src="https://github.com/user-attachments/assets/d82c88c9-9f15-4164-a2f4-d1a947b91d03" />



##  Results

- Dataset: 1,903 receipts (973 real, 930 forged)

### Performance (Test Set)

| Model | AUC | Accuracy |
|---|---|---|
| Classifier (EfficientNet-B3) | 0.67 | 53% |
| **Ensemble (multi-signal)** | **0.81** | **76%** |

**Key takeaway:**  
Single-model approaches struggle on localized tampering. Combining **classification + segmentation + OCR reasoning** improves robustness significantly (**+13.7 pp AUC**).

---

##  Problem Statement

Receipt fraud often involves:
- Small edits (1–2 digits changed)
- Localized tampering (<5% of image)
- High-quality edits with minimal visual artifacts  

Pure CNN-based approaches fail because most of the image remains authentic.

---

##  Approach

The system combines multiple complementary signals:

- **Classifier (EfficientNet-B3):** global real vs forged prediction  
- **Segmentation (U-Net):** localizes tampered regions  
- **Physical checks (OpenCV):** detects blobs, borders, ELA, lighting inconsistencies  
- **OCR + logic:** validates totals, taxes, and field consistency  
- **Anomaly detector:** flags statistical irregularities in extracted text  

These signals are fused using a **rule-based + ensemble decision engine**.

---

##  System Flow






---

##  Dataset

- Source: **SROIE 2019 (Kaggle)**
- 973 real receipts with OCR annotations  
- Programmatically generated forged samples using OCR-aligned edits  

**Final dataset:**
- 1,903 images (973 real, 930 forged)  
- Pixel-level masks for all forged images  
- Near-balanced (~1.05:1)  
- Split: 1,426 train / 286 val / 191 test  

---

##  Training

- Trained on **Google Colab (T4 GPU)**  
- Notebooks:
  - `SROIE_Forgery_Training_Colab.ipynb` — data + baseline  
  - `Improved_Detection_v2.ipynb` — final models  

### Models

- **EfficientNet-B3:** classification (320×320, pretrained, TTA, class weighting)  
- **U-Net (EfficientNet-B3 encoder):** tamper localization (Focal + Dice + BCE)  
- **Isolation Forest:** anomaly detection on OCR features (trained on real-only data)

---

## Inference Pipeline

- Runs **multi-signal detection**:
  - Visual (classifier)
  - Spatial (segmentation)
  - Physical (image artifacts)
  - Semantic (OCR + logical validation)

- Uses:
  - Rule-based overrides for strong signals  
  - Multi-signal agreement  
  - Final ensemble scoring  

**Output:**
- `clean / suspicious / forged`
- Confidence score  
- Tamper heatmap  
- Parsed receipt fields  

---

## Key Insight

Forgery detection is not purely a vision problem.

Combining:
- **Where** (segmentation),
- **What** (OCR logic),
- **How** (physical artifacts)

leads to significantly better performance than any single model.

---

## Tech Stack

- PyTorch, EfficientNet, U-Net  
- OpenCV (image forensics)  
- Tesseract OCR  
- Scikit-learn (Isolation Forest)  
- FastAPI (deployment)

---

## Future Improvements

- Improve OCR robustness for multi-currency receipts  
- Train on real-world forged data (not synthetic)  
- Replace rule-based fusion with learned meta-model  

---
