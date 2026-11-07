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