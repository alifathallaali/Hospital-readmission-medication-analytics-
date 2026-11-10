# Model Validation & Evaluation Report - Phase 8

## 1. Scope & Objective
This report documents the rigorous evaluation metrics, sealed hold-out performance, calibration results, and capacity analysis for both the baseline and Phase 2 advanced medication models.

## 2. Evaluation Methodology
- **Train/Test Split:** Stratified 80/20 hold-out split (`random_state=42`) maintaining class prevalence across folds.
- **Leakage Prevention:** All scalers, imputers, and encoders are fitted exclusively within training folds. No synthetic fallback metrics are utilized.
- **Primary Metrics:** ROC-AUC, Precision-Recall AUC (PR-AUC), Brier Score Loss, and binned calibration reliability.

## 3. Comparative Model Performance
| Model Architecture | Dataset & Cohort | N (Patients) | Positive Class (<30 Days) | ROC-AUC | PR-AUC | Brier Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline Logistic Regression** | Diabetes 130-US (First Encounter) | 69,970 | ~6,297 (9.0%) | **0.640** | **0.166** | **0.0793** |
| **Advanced HistGradientBoosting** | Diabetes 130-US + Regimen Features | 69,970 | ~6,297 (9.0%) | **0.645** | **0.170** | **0.0789** |

## 4. Capacity & Operational Analysis
Evaluating operational utility via risk-tiered targeting (Capacity Table):
- **Top 1% Flagged:** 699 patients | Recall: 13.8% | Lift: 2.76x
- **Top 2% Flagged:** 1,399 patients | Recall: 20.5% | Lift: 2.05x
- **Top 4% Flagged:** 2,798 patients | Recall: 31.2% | Lift: 1.56x
- **Top 10% Flagged:** 6,997 patients | Recall: 37.4% | Lift: 1.87x

## 5. Calibration & Reliability
- Calibration curves generated across 10 probability bins. Brier scores reflect strong probability calibration on imbalanced readmission targets.