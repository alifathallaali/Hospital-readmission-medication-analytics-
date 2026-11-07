"""Consistency tests: exports vs. each other, vs. the README, and data-governance checks.
They need only files that are in the repository (no raw data), so they run in a fresh clone."""
import json
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app_data"

REQUIRED = [
    "readmission_by_group.csv", "capacity_curve.csv", "capacity_table.csv", "calibration.csv",
    "pr_curve.csv", "subgroups.csv", "model_comparison.csv", "kpis.json",
    "mimic_diagnoses.csv", "mimic_top_drugs.csv", "mimic_diabetic_drugs.csv",
    "mimic_los.csv", "mimic_insulin.csv", "mimic_kpis.json",
]
FORBIDDEN_COLUMNS = {"subject_id", "hadm_id", "stay_id", "patient_nbr", "encounter_id"}


def kpis():
    return json.loads((APP / "kpis.json").read_text())


def mimic_kpis():
    return json.loads((APP / "mimic_kpis.json").read_text())


def test_required_files_exist():
    missing = [f for f in REQUIRED if not (APP / f).exists()]
    assert not missing, f"Missing exports in app_data/: {missing}"


def test_no_patient_identifiers_in_exports():
    bad = {}
    for p in APP.glob("*.csv"):
        cols = {c.lower() for c in pd.read_csv(p, nrows=0).columns}
        if cols & FORBIDDEN_COLUMNS:
            bad[p.name] = sorted(cols & FORBIDDEN_COLUMNS)
    assert not bad, f"Identifier columns found in exports: {bad}"


def test_gitignore_protects_raw_data():
    lines = {l.strip() for l in (ROOT / ".gitignore").read_text().splitlines()}
    assert "data/" in lines and "*.db" in lines


def test_kpis_are_sane():
    k = kpis()
    assert 0 < k["readmit_rate_pct"] < 100
    assert 0.5 < k["holdout_roc_auc"] < 1.0
    assert 0 < k["holdout_pr_auc"] < 1.0
    assert k["patients"] <= k["encounters_raw"]


def test_capacity_table_matches_curve_and_lift():
    k = kpis()
    cap = pd.read_csv(APP / "capacity_curve.csv")
    cap["flag_pct"] = cap["flag_pct"].round().astype(int)
    tbl = pd.read_csv(APP / "capacity_table.csv")
    for _, r in tbl.iterrows():
        pct = int(str(r["flag top"]).rstrip("%"))
        curve_recall = float(cap.loc[cap["flag_pct"] == pct, "recall_pct"].iloc[0])
        assert abs(curve_recall - r["recall %"]) < 0.6, f"curve/table disagree at {pct}%"
        assert abs(r["lift"] - r["recall %"] / pct) < 0.06, f"lift != recall/share at {pct}%"
        assert abs(r["patients flagged"] - k["holdout_n"] * pct / 100) <= 1


def test_mimic_kpis_are_consistent():
    m = mimic_kpis()
    assert m["table_mismatches"] == 0
    assert m["readmit_unplanned"] <= m["readmit_any"] <= m["discharged_alive"] <= m["admissions"]
    assert m["with_diabetes"] <= m["admissions"]
    assert m["patients"] <= m["admissions"]


def test_model_comparison_schema_and_values():
    """Ensure model comparison export has required columns and valid performance ranges."""
    df = pd.read_csv(APP / "model_comparison.csv")
    expected_cols = {"model", "roc_auc", "roc_sd", "pr_auc", "pr_sd"}
    assert expected_cols.issubset(df.columns), f"Missing columns in model_comparison.csv: {expected_cols - set(df.columns)}"
    assert len(df) >= 4, "Model comparison should include at least 4 baseline/trained models."
    assert ((df["roc_auc"] >= 0.5) & (df["roc_auc"] <= 1.0)).all()
    assert ((df["pr_auc"] >= 0.0) & (df["pr_auc"] <= 1.0)).all()


def test_subgroups_schema_and_validity():
    """Verify subgroups export schema and structural integrity."""
    df = pd.read_csv(APP / "subgroups.csv")
    expected_cols = {"variable", "group", "n", "readmitted", "prevalence %", "ROC-AUC"}
    assert expected_cols.issubset(df.columns), f"Missing columns in subgroups.csv: {expected_cols - set(df.columns)}"
    assert len(df) > 0, "Subgroups export is empty."
    assert (df["n"] > 0).all()


def test_calibration_and_pr_curves_valid():
    """Verify calibration and PR curve export files are non-empty and well-formed."""
    cal = pd.read_csv(APP / "calibration.csv")
    assert len(cal) > 0, "calibration.csv is empty"
    pr = pd.read_csv(APP / "pr_curve.csv")
    assert len(pr) > 0, "pr_curve.csv is empty"


def test_readme_matches_exports():
    """If the model is re-run and the exports change, the README numbers must be updated too."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    k = kpis()
    tbl = pd.read_csv(APP / "capacity_table.csv")
    expected = [
        f"{k['holdout_roc_auc']:.3f}",
        f"{k['holdout_pr_auc']:.3f}",
        f"{k['holdout_brier']:.4f}",
        f"{k['patients']:,}",
        f"{k['encounters_raw']:,}",
    ]
    for _, r in tbl.iterrows():
        expected += [f"{int(r['patients flagged']):,}", f"{r['recall %']:.1f}%", f"{r['lift']:.2f}x"]
    missing = [e for e in expected if e not in readme]
    assert not missing, f"README does not contain current export values: {missing}"


def test_readme_does_not_overclaim():
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    for phrase in ["clinically validated model", "improves patient outcomes", "causes readmission"]:
        assert phrase not in readme


def test_streamlit_app_runs_without_exception():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "Home.py"), default_timeout=60).run()
    assert not at.exception, [str(e.value) for e in at.exception]