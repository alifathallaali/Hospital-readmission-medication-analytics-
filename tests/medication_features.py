"""Unit tests for PharmaLens advanced medication analytics pipeline."""
import pandas as pd
import pytest
from src.medication_features import compute_regimen_complexity, flag_high_risk_transitions


def test_compute_regimen_complexity_basic():
    data = {
        "num_medications": [3, 6, 2],
        "any_dose_change": [True, False, True],
        "drugs_changed_cap": [1, 0, 2]
    }
    df = pd.DataFrame(data)
    res = compute_regimen_complexity(df)
    
    assert "regimen_volatility_score" in res.columns
    assert "is_high_polypharmacy" in res.columns
    assert "regimen_stability_score" in res.columns
    
    # Check polypharmacy flag logic (6 meds >= 5 is True, others False)
    assert list(res["is_high_polypharmacy"]) == [False, True, False]
    
    # Check stability score bounds [0.0, 1.0]
    assert (res["regimen_stability_score"] >= 0.0).all()
    assert (res["regimen_stability_score"] <= 1.0).all()


def test_compute_regimen_complexity_missing_columns():
    """Ensure graceful handling when required raw columns are missing."""
    df = pd.DataFrame({"random_col": [1, 2, 3]})
    res = compute_regimen_complexity(df)
    assert "regimen_volatility_score" in res.columns
    assert "is_high_polypharmacy" in res.columns
    assert not res["is_high_polypharmacy"].any()


def test_flag_high_risk_transitions():
    data = {
        "any_dose_change": [True, False, True],
        "insulin": [True, False, False]
    }
    df = pd.DataFrame(data)
    res = flag_high_risk_transitions(df)
    
    assert "high_risk_glycemic_transition" in res.columns
    assert list(res["high_risk_glycemic_transition"]) == [True, False, False]