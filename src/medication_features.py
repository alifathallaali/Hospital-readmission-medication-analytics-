"""Advanced medication and regimen complexity feature engineering module for PharmaLens."""
import pandas as pd
import numpy as np


def compute_regimen_complexity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes advanced medication change metrics, polypharmacy indices, 
    and regimen stability scores for hospital readmission analytics.
    """
    out = df.copy()
    
    # 1. Regimen volatility score based on change indicators and caps
    if "any_dose_change" in out.columns and "drugs_changed_cap" in out.columns:
        out["regimen_volatility_score"] = (
            out["drugs_changed_cap"] * out["any_dose_change"].astype(float)
        )
    else:
        out["regimen_volatility_score"] = 0.0
        
    # 2. Polypharmacy stratification flag (>= 5 medications is standard threshold)
    if "num_medications" in out.columns:
        out["is_high_polypharmacy"] = out["num_medications"] >= 5
    else:
        out["is_high_polypharmacy"] = False
        
    # 3. Regimen stability score (ratio of unchanged medications)
    if "num_medications" in out.columns and "drugs_changed_cap" in out.columns:
        denom = out["num_medications"].replace(0, 1)
        out["regimen_stability_score"] = np.clip(
            1.0 - (out["drugs_changed_cap"].astype(float) / denom), 0.0, 1.0
        )
    else:
        out["regimen_stability_score"] = 1.0
        
    return out


def flag_high_risk_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flags encounters involving critical chronic or glycemic agent transitions.
    """
    out = df.copy()
    
    # Identify columns corresponding to key glycemic/chronic agents
    glycemic_cols = [
        c for c in out.columns 
        if any(k in c.lower() for k in ["insulin", "metformin", "glyburide", "pioglitazone"])
    ]
    
    if glycemic_cols and "any_dose_change" in out.columns:
        out["high_risk_glycemic_transition"] = out["any_dose_change"] & out[glycemic_cols].any(axis=1)
    else:
        out["high_risk_glycemic_transition"] = False
        
    return out