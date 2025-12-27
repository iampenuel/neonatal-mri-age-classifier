# Neonatal Brain MRI Age Classification (0–12 months)

Educational ML project: classify neonatal/infant brain MRI slices into developmental age bins.

## Disclaimer
**Educational demo only. Not a medical device. Not for clinical diagnosis or medical decisions.**

## Models
- Baseline: ResNet-18 (transfer learning) — test macro-F1 ≈ 0.816
- Improved: EfficientNet-B0 (transfer learning) + Grad-CAM — test macro-F1 ≈ 0.878

## Repo Structure
- `notebooks/` data prep + training notebooks
- `reports/` confusion matrices, per-class metrics, Grad-CAM examples

## Data
Kaggle: Neonatal Brain Development MRI dataset (see citations in report/README).

## How to run
Open the notebooks in Google Colab and run top-to-bottom.
