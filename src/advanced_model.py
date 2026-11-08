import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    brier_score_loss, 
    precision_recall_curve, 
    roc_auc_score, 
    average_precision_score
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

def run_advanced_modeling():
    print("--- Starting Phase 3: Advanced Modeling & Calibration from diabetic_data.csv ---")
    
    data_path = Path("data/diabetic_data.csv")
    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path}")
        return
    
    print("Loading raw dataset...")
    df = pd.read_csv(data_path)
    
    # Basic data preprocessing & cohort filtering (matching project standards)
    # Replace '?' with NaN
    df = df.replace('?', np.nan)
    
    # Drop unnecessary columns or columns with massive missing data
    drop_cols = ["weight", "payer_code", "patient_nbr", "encounter_id"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    
    # Exclude hospice/death discharge dispositions if present (e.g., 11, 13, 14, 19, 20, 21)
    if "discharge_disposition_id" in df.columns:
        exclusions = [11, 13, 14, 19, 20, 21]
        df = df[~df["discharge_disposition_id"].isin(exclusions)]
        
    # Define target: readmitted within 30 days (<30) vs others
    if "readmitted" in df.columns:
        y = df["readmitted"].apply(lambda val: 1 if str(val).strip() == "<30" else 0)
        X = df.drop(columns=["readmitted"], errors="ignore")
    else:
        print("Error: 'readmitted' column not found in dataset.")
        return
        
    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)
    
    print(f"Dataset shape after preprocessing: X={X.shape}, y={y.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Using HistGradientBoosting (No external internet installation needed)
    hgb = HistGradientBoostingClassifier(random_state=42)
    param_distributions = {
        "max_iter": [50, 100],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1]
    }
    
    print("Running Randomized Search for Hyperparameter Tuning...")
    search = RandomizedSearchCV(
        estimator=hgb,
        param_distributions=param_distributions,
        n_iter=3,
        scoring="roc_auc",
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    
    print(f"Model Evaluation -> ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Brier: {brier:.4f}")
    
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
    
    metrics_data = {
        "model_name": "HistGradientBoosting_Tuned_With_Advanced_Meds",
        "best_params": search.best_params_,
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "brier_score": round(float(brier), 4),
        "calibration_curve": {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist()
        }
    }
    
    output_dir = Path("app_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "advanced_model_metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=4)
        
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='HGB Tuned')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve - Phase 3')
    plt.legend()
    plt.savefig(figures_dir / 'advanced_calibration_curve.png')
    plt.close()
    
    print("Advanced modeling completed successfully and files saved!")

if __name__ == "__main__":
    run_advanced_modeling()