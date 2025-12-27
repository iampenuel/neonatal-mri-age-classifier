# Neonatal Brain MRI Age Classification (0–12 months)

In this project, I built and evaluated a deep learning model that predicts an infant’s **developmental age bin (0–12 months)** from a **single neonatal/infant brain MRI slice**. I trained transfer learning CNNs on a public Kaggle dataset and added **Grad-CAM** to provide a simple visual explanation of what influenced the model’s prediction.

> ⚠️ This model is a **research/educational demo only**. It is not a clinical diagnostic
> tool and should not be used for real medical decisions.

---

## Problem Statement
Infant brain development changes rapidly during the first year of life. In research and clinical settings, developmental staging can require expert interpretation and can be time-consuming. This project explores whether a machine learning model can learn useful developmental patterns from MRI slices and produce an interpretable age-bin prediction.

---

## Key Results
**Task:** 14-class age-bin classification aligned to the dataset’s structure:
- `0–10 days`, `11–20 days`, `21–30 days`, then `2–12 months`.

**Models trained**
- **Baseline:** ResNet-18 (ImageNet pretrained, fine-tuned)
  - **Test macro-F1:** **0.8162**
- **Improved (final):** EfficientNet-B0 (ImageNet pretrained, fine-tuned) + **Grad-CAM**
  - **Test macro-F1:** **0.8775**

**Saved outputs (see `reports/`)**
- Confusion matrices (PNG)
- Per-class precision/recall/F1 (CSV)
- Grad-CAM overlays (PNG)

---

## Methodology

### 1) Dataset loading & inspection
- Downloaded the “Neonatal Brain Development MRI” dataset from Kaggle.
- Parsed folder structure into a metadata table:
  - modality (T1/T2), sex, age-bin label, file path.

### 2) Label scheme
Used the dataset’s native 14 age bins:
- `0_10d`, `11_20d`, `21_30d`, `2m`, `3m`, …, `12m`

### 3) Split strategy (leakage-aware)
Because explicit patient identifiers were not available in the file paths, I used a conservative strategy to reduce leakage:
- Created **group IDs** using perceptual hashing (pHash) to reduce duplicate/near-duplicate leakage.
- Performed **group-based splitting** so identical/near-identical images are not shared across train/val/test.

> Limitation: Without true patient IDs, perfect patient-level leakage prevention cannot be guaranteed.

### 4) Model training (transfer learning CNNs)
- **ResNet-18** baseline and **EfficientNet-B0** improved model (both ImageNet pretrained).
- Used **weighted cross-entropy** to address minor class imbalance.
- Selected best checkpoint using **validation macro-F1**.

### 5) Evaluation
Reported:
- **Per-class F1**
- **Macro-F1**
- **Confusion matrix**

### 6) Explainability (Grad-CAM)
Implemented **Grad-CAM** to visualize which regions most influenced the predicted age bin. Saved overlay examples for inspection and for a future demo app.

---
## How to Run (Colab)

1. Open `notebooks/01_data_and_splits.ipynb`
   - Downloads data from Kaggle
   - Builds metadata
   - Creates leakage-aware splits and saves artifacts

2. Open `notebooks/02_train_eval_gradcam_demo.ipynb`
   - Loads saved artifacts
   - Trains baseline + improved model
   - Exports metrics, confusion matrices, and Grad-CAM overlays

---

## Results Artifacts

- EfficientNet-B0 confusion matrix: `reports/cm_effnetb0.png`  
- ResNet-18 confusion matrix: `reports/cm_resnet18.png`  
- Per-class metrics:
  - `reports/per_class_metrics_effnetb0.csv`
  - `reports/per_class_metrics_resnet18.csv`
- Grad-CAM overlays: `reports/gradcam_gallery/`

---

## Data Sources (Provenance)

- Kaggle: *Neonatal Brain Development MRI* dataset  
- 2025 paper: MDPI *Journal of Clinical Medicine* (references the Kaggle dataset)

---

## Technologies Used

- **Language:** Python  
- **Core libraries:** NumPy, Pandas, PyTorch, Torchvision, scikit-learn, Matplotlib  
- **Tools / Platforms:**
  - Google Colab (training and experimentation)
  - Kaggle (dataset hosting)
  - Git & GitHub (version control and collaboration)
  - Streamlit (planned web demo)

---


## Disclaimer

This project is for **educational and research demonstration only**.  
It is **not** a medical device and must **not** be used for clinical diagnosis or medical decision-making.

---

## Author

Completed by **Penuel Stanley-Zebulon** (GitHub: `@iampenuel`, Email: `pcs5301@psu.edu`)

---

## Repository Structure
```text
neonatal-mri-age-classifier/
  notebooks/
    01_data_and_splits.ipynb
    02_train_eval_gradcam_demo.ipynb
  reports/
    cm_resnet18.png
    cm_effnetb0.png
    per_class_metrics_resnet18.csv
    per_class_metrics_effnetb0.csv
    gradcam_gallery/
  README.md
  requirements.txt
