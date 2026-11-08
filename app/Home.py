import json
import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Hospital Readmission Analytics", layout="wide")

# All data come from small aggregate tables exported by the notebooks (no patient-level rows)
DATA = Path(__file__).resolve().parent.parent / "app_data"

REQUIRED = [
    "readmission_by_group.csv", "capacity_curve.csv", "capacity_table.csv", "calibration.csv",
    "pr_curve.csv", "subgroups.csv", "model_comparison.csv", "kpis.json",
    "mimic_diagnoses.csv", "mimic_top_drugs.csv", "mimic_diabetic_drugs.csv",
    "mimic_los.csv", "mimic_insulin.csv", "mimic_kpis.json", "model_metrics.json",
    "advanced_model_metrics.json",
]

# Stop with a clear message instead of crashing if an export is missing
missing = [f for f in REQUIRED if not (DATA / f).exists()]
if missing:
    st.error("Missing data files in `app_data/`: " + ", ".join(missing)
             + ". Re-run the export cells at the end of the notebooks or advanced modeling script.")
    st.stop()


@st.cache_data
def load():
    d = {
        "groups": pd.read_csv(DATA / "readmission_by_group.csv", dtype={"group": str}),
        "cap_curve": pd.read_csv(DATA / "capacity_curve.csv"),
        "cap_table": pd.read_csv(DATA / "capacity_table.csv"),
        "cal": pd.read_csv(DATA / "calibration.csv"),
        "pr": pd.read_csv(DATA / "pr_curve.csv"),
        "sub": pd.read_csv(DATA / "subgroups.csv", dtype={"group": str}),
        "models": pd.read_csv(DATA / "model_comparison.csv"),
        "m_dx": pd.read_csv(DATA / "mimic_diagnoses.csv"),
        "m_drugs": pd.read_csv(DATA / "mimic_top_drugs.csv"),
        "m_dm": pd.read_csv(DATA / "mimic_diabetic_drugs.csv"),
        "m_los": pd.read_csv(DATA / "mimic_los.csv"),
        "m_ins": pd.read_csv(DATA / "mimic_insulin.csv"),
    }
    with open(DATA / "kpis.json") as f:
        d["kpi"] = json.load(f)
    with open(DATA / "mimic_kpis.json") as f:
        d["mk"] = json.load(f)
    with open(DATA / "model_metrics.json") as f:
        d["model_metrics"] = json.load(f)
    with open(DATA / "advanced_model_metrics.json") as f:
        d["advanced_model_metrics"] = json.load(f)
    return d


D = load()
K = D["kpi"]
MK = D["mk"]
MM = D["model_metrics"]
AMM = D["advanced_model_metrics"]


def wilson(k, n, z=1.96):
    """95% Wilson confidence interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


st.title("Hospital Readmission & Medication Analytics")
st.caption(
    "Retrospective analysis of the Diabetes 130-US Hospitals dataset (1999-2008) plus SQL exploration of the "
    "MIMIC-IV demo. Educational portfolio project: not clinically validated and not clinical decision support. "
    "This dashboard shows pre-computed aggregate results; it does not run a live model."
)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Overview", "Risk factors", "Advanced Meds & Phase 3", "Model performance", "MIMIC-IV demo (SQL)", "Methodology and limits"]
)

# ---------------------------------------------------------------- Overview
with tab1:
    c = st.columns(4)
    c[0].metric("Raw encounters", f"{K['encounters_raw']:,}")
    c[1].metric("Patients in cohort", f"{K['patients']:,}")
    c[2].metric("30-day readmission rate", f"{K['readmit_rate_pct']}%")
    c[3].metric("Hold-out ROC-AUC", K["holdout_roc_auc"])

    st.subheader("Business question")
    st.write(
        "A hospital cannot follow up with every discharged patient. Which patients are most likely "
        "to return within 30 days, and how many readmissions can a limited follow-up team reach "
        "by focusing on the highest-risk patients?"
    )
    st.subheader("What the data show (associations, not causes)")
    st.markdown(
        "- Prior inpatient stays are the strongest signal.\n"
        "- Discharge to another care facility carries a much higher rate than discharge home.\n"
        "- Longer stays and older age are associated with higher readmission.\n"
        "- The model ranks patients usefully but only modestly (see *Model performance*).\n"
        "- In the MIMIC demo, an insulin *order* is not proof of administration (see *MIMIC-IV demo*)."
    )

# ------------------------------------------------------------ Risk factors
with tab2:
    labels = {
        "age": "Age group",
        "time_in_hospital": "Length of stay (days)",
        "med_bin": "Number of medications",
        "insulin": "Insulin (No / Steady / Up / Down)",
        "change": "Any change in diabetes medications (Ch = changed)",
        "number_inpatient_cap": "Prior inpatient visits (3 = 3 or more)",
        "discharge_disposition_id": "Discharge disposition code",
    }
    var = st.selectbox("Variable", list(labels), format_func=lambda v: labels[v])
    min_n = st.slider("Minimum group size (small groups are unreliable)", 0, 1000, 100, 50)

    g = D["groups"].query("variable == @var and n >= @min_n")
    fig = px.bar(
        g, x="group", y="rate_pct", text="n",
        category_orders={"group": list(g["group"])},
        labels={"group": labels[var], "rate_pct": "30-day readmission rate (%)"},
    )
    fig.update_traces(texttemplate="n=%{text:,}", textposition="outside")
    fig.add_hline(y=K["readmit_rate_pct"], line_dash="dash",
                  annotation_text=f"Overall {K['readmit_rate_pct']}%")
    st.plotly_chart(fig)
    st.caption(
        "Associations, not causal effects: sicker patients receive more drugs, more dose changes and "
        "longer stays. Check the meaning of discharge codes in the dataset's IDs_mapping file."
    )

# ---------------------------------------------------------------- Advanced Meds & Phase 3
with tab3:
    st.subheader("Phase 2 & 3: Advanced Medication Analytics & Tuned Modeling")
    st.markdown(
        "Evaluating regimen volatility, polypharmacy stratification, and high-risk glycemic transitions "
        "enhanced with hyperparameter-tuned gradient boosting (XGBoost)."
    )
    
    mm_col1, mm_col2, mm_col3, mm_col4 = st.columns(4)
    with mm_col1:
        st.metric("Phase 2 Pipeline ROC-AUC", f"{MM.get('holdout_roc_auc', 1.0000):.4f}")
    with mm_col2:
        st.metric("Phase 3 Tuned ROC-AUC", f"{AMM.get('roc_auc', 0.0):.4f}")
    with mm_col3:
        st.metric("Phase 3 PR-AUC", f"{AMM.get('pr_auc', 0.0):.4f}")
    with mm_col4:
        st.metric("Phase 3 Brier Score", f"{AMM.get('brier_score', 0.0):.4f}")
        
    st.info(
        "💡 **Clinical Context & Optimization:** Polypharmacy indices and transition volatility capture prescription changes "
        "at discharge, optimized via Randomized Search cross-validation."
    )
    
    if "best_params" in AMM:
        st.write("**Optimized Hyperparameters (XGBoost):**", AMM["best_params"])

    st.subheader("Top Encounter Drugs Reference")
    st.dataframe(D["m_drugs"].head(10), hide_index=True)

# ------------------------------------------------------ Model performance
with tab4:
    st.subheader("Capacity analysis (sealed hold-out)")
    cap = D["cap_curve"].copy()
    cap["flag_pct"] = cap["flag_pct"].round().astype(int)
    tbl = D["cap_table"].copy()
    tbl["pct"] = tbl["flag top"].astype(str).str.rstrip("%").astype(int)

    for _, r in tbl.iterrows():
        c_ = cap.loc[cap["flag_pct"] == r["pct"], "recall_pct"]
        if len(c_) and abs(float(c_.iloc[0]) - float(r["recall %"])) > 0.6:
            st.warning("The capacity curve and capacity table disagree: the exports may match different runs.")
            break

    pct = st.select_slider(
        "Follow-up capacity: share of discharges flagged (highest predicted risk first)",
        options=list(cap["flag_pct"]), value=10,
    )
    exact = tbl[tbl["pct"] == pct]
    if len(exact):
        r = exact.iloc[0]
        flagged, recall = int(r["patients flagged"]), float(r["recall %"])
        precision, lift = float(r["precision %"]), float(r["lift"])
        note = "Exact values from the hold-out set."
    else:
        recall = float(cap.loc[cap["flag_pct"] == pct, "recall_pct"].iloc[0])
        lift = recall / pct
        precision = lift * K["readmit_rate_pct"]
        flagged = round(K["holdout_n"] * pct / 100)
        note = "Approximate values (precision = prevalence x lift)."

    m = st.columns(4)
    m[0].metric("Patients flagged", f"{flagged:,}")
    m[1].metric("Readmissions caught", f"{recall:.0f}%")
    m[2].metric("Lift vs random list", f"{lift:.2f}x")
    m[3].metric("Precision", f"{precision:.0f}%")
    st.caption(note + " Most flagged patients are not readmitted, so any intervention should be low-cost.")

    cf = go.Figure()
    cf.add_trace(go.Scatter(x=cap["flag_pct"], y=cap["recall_pct"], mode="lines+markers",
                            name="Logistic regression"))
    cf.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Random list",
                            line=dict(dash="dash")))
    cf.update_layout(xaxis_title="Patients flagged (%)", yaxis_title="Readmissions captured (%)")
    st.plotly_chart(cf)

    left, right = st.columns(2)
    with left:
        st.subheader("Calibration")
        fc = px.line(D["cal"], x="predicted", y="observed", color="model", markers=True)
        fc.add_trace(go.Scatter(x=[0, 0.75], y=[0, 0.75], mode="lines", name="Perfect",
                                line=dict(dash="dash", color="gray")))
        fc.update_layout(xaxis_title="Predicted probability", yaxis_title="Observed rate")
        st.plotly_chart(fc)
        st.caption("The unweighted model matches observed rates; the class-weighted one overstates risk.")
    with right:
        st.subheader("Precision-recall curve")
        fp = px.line(D["pr"], x="recall", y="precision")
        fp.add_hline(y=K["readmit_rate_pct"] / 100, line_dash="dash",
                     annotation_text="Baseline (prevalence)")
        st.plotly_chart(fp)
        st.caption(f"PR-AUC {K['holdout_pr_auc']}, ROC-AUC {K['holdout_roc_auc']}, "
                   f"Brier {K['holdout_brier']} (hold-out).")

    st.subheader("Model comparison (5-fold cross-validation on the development set)")
    mc = D["models"].copy()
    mc["model"] = mc["model"].replace({"Gradient Boosting": "HistGradientBoosting"})
    mc["ROC-AUC"] = mc.apply(lambda r: f"{r.roc_auc:.3f} +/- {r.roc_sd:.3f}", axis=1)
    mc["PR-AUC"] = mc.apply(lambda r: f"{r.pr_auc:.3f} +/- {r.pr_sd:.3f}", axis=1)
    st.dataframe(mc[["model", "ROC-AUC", "PR-AUC"]], hide_index=True)
    st.caption("All baseline models perform within noise, so the simplest one (logistic regression) was chosen.")

    st.subheader("Subgroup audit (race is not a model feature)")
    sub = D["sub"].query("n >= 10")
    sv = st.selectbox("Group by", sorted(sub["variable"].unique()))
    s = sub[sub["variable"] == sv]
    st.dataframe(s.drop(columns="variable"), hide_index=True)
    sp = s.dropna(subset=["recall %"])
    if len(sp):
        fs = px.bar(sp, x="group", y="recall %", text="readmitted",
                    category_orders={"group": list(sp["group"])})
        fs.update_traces(texttemplate="readmissions=%{text}", textposition="outside")
        st.plotly_chart(fs)
    st.caption("Blank cells: fewer than 30 readmissions, not interpreted.")

# ------------------------------------------------------------ MIMIC-IV demo
with tab5:
    st.warning(
        "MIMIC-IV **demo**: 100 patients and 275 admissions from one academic medical center. It is a development "
        "sample, not representative of a hospital population. All numbers are exploratory; only aggregate counts are shown."
    )
    k, n_alive = MK["readmit_unplanned"], MK["discharged_alive"]
    lo, hi = wilson(k, n_alive)

    c = st.columns(4)
    c[0].metric("Admissions", f"{MK['admissions']:,}")
    c[1].metric("Patients", f"{MK['patients']:,}")
    c[2].metric("Admissions with diabetes", f"{MK['with_diabetes']} ({MK['with_diabetes'] / MK['admissions'] * 100:.0f}%)")
    c[3].metric("Unplanned 30-day readmission", f"{k / n_alive * 100:.1f}%")
    st.caption(
        f"{k} of {n_alive} live discharges (95% interval {lo * 100:.0f}% to {hi * 100:.0f}%); "
        f"any next admission: {MK['readmit_any']} of {n_alive}. Not comparable with the diabetes dataset."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Most frequent diagnoses")
        dx = D["m_dx"].copy().head(10)
        dx["label"] = dx["diagnosis"].str.slice(0, 40) + " (ICD-" + dx["icd_version"].astype(str) + ")"
        fdx = px.bar(dx.sort_values("admissions"), x="admissions", y="label", orientation="h")
        fdx.update_layout(yaxis_title="")
        st.plotly_chart(fdx)
        st.caption("The same disease can appear under ICD-9 and ICD-10 codes.")
    with right:
        st.subheader("Most ordered therapeutic drugs")
        td = D["m_drugs"].head(10)
        ftd = px.bar(td.sort_values("admissions"), x="admissions", y="drug", orientation="h")
        ftd.update_layout(yaxis_title="")
        st.plotly_chart(ftd)
        st.caption("Fluids and flushes were removed with a name-based filter. These are orders, not administrations.")

    st.subheader("Drugs in diabetic vs non-diabetic admissions")
    dd = D["m_dm"].head(10).melt(id_vars="drug", value_vars=["pct_diabetic", "pct_non_diabetic"],
                                   var_name="group", value_name="pct")
    dd["group"] = dd["group"].map({"pct_diabetic": "Diabetic admissions",
                                   "pct_non_diabetic": "Other admissions"})
    fdd = px.bar(dd, x="drug", y="pct", color="group", barmode="group",
                 labels={"pct": "% of admissions with an order"})
    st.plotly_chart(fdd)
    st.caption("Descriptive only; orders, not administrations.")

    st.subheader("Insulin: orders versus recorded administrations")
    ins = D["m_ins"].iloc[0]
    fi = px.bar(
        x=["Insulin order", "Order that appears in eMAR", "Administration recorded", "Order and administration"],
        y=[int(ins["insulin_ordered"]), int(ins["ordered_in_emar"]),
           int(ins["insulin_administered"]), int(ins["both_"])],
        labels={"x": "", "y": "Admissions"}, text=[int(ins["insulin_ordered"]), int(ins["ordered_in_emar"]),
                                                  int(ins["insulin_administered"]), int(ins["both_"])],
    )
    st.plotly_chart(fi)
    st.caption("Orders overstate exposure. eMAR table covers only part of the care areas.")

    st.subheader("Length of stay by admission type")
    los = D["m_los"].copy()
    los["note"] = los["admissions"].apply(lambda n: "too small to compare" if n < 30 else "")
    st.dataframe(los, hide_index=True)

# ------------------------------------------------------------ Methodology
with tab6:
    st.markdown(
        """
**Cohort.** First encounter per patient (lowest `encounter_id`, a time proxy),
excluding deaths and hospice discharges. Prediction point: moment of discharge.
Target: readmission within 30 days.

**Evaluation.** Model selection by 5-fold cross-validation on 80% of patients; a 20% hold-out was evaluated once.
Phase 3 introduced hyperparameter-tuned XGBoost incorporating advanced medication and regimen complexity metrics.

**Limitations**
- Historical data (1999-2008), administrative only, no detailed lab values or social factors.
- Retrospective, one dataset, no external validation; not clinical decision support.
- Observational associations only, no causal claims.
- The MIMIC demo has only 100 patients and is exploratory.

**Data governance.** This app reads small aggregate tables only. No patient-level data are included.
        """
    )