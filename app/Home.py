import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="PharmaLens - Hospital Readmission & Medication Analytics", layout="wide")

# All data come from small aggregate tables exported by the notebooks (no patient-level rows)
DATA = Path(__file__).resolve().parent.parent / "app_data"


@st.cache_data
def load():
    d = {}
    # Load primary readmission analytics aggregates
    try:
        d["groups"] = pd.read_csv(DATA / "readmission_by_group.csv", dtype={"group": str})
        d["cap_curve"] = pd.read_csv(DATA / "capacity_curve.csv")
        d["cap_table"] = pd.read_csv(DATA / "capacity_table.csv")
        d["cal"] = pd.read_csv(DATA / "calibration.csv")
        d["pr"] = pd.read_csv(DATA / "pr_curve.csv")
        d["sub"] = pd.read_csv(DATA / "subgroups.csv", dtype={"group": str})
        d["models"] = pd.read_csv(DATA / "model_comparison.csv")
        with open(DATA / "kpis.json") as f:
            d["kpi"] = json.load(f)
    except Exception:
        pass

    # Load MIMIC-IV clinical aggregates
    try:
        with open(DATA / "mimic_kpis.json") as f:
            d["mimic_kpi"] = json.load(f)
        d["mimic_diagnoses"] = pd.read_csv(DATA / "mimic_diagnoses.csv")
        d["mimic_top_drugs"] = pd.read_csv(DATA / "mimic_top_drugs.csv")
        d["mimic_diabetic_drugs"] = pd.read_csv(DATA / "mimic_diabetic_drugs.csv")
        d["mimic_los"] = pd.read_csv(DATA / "mimic_los.csv")
        d["mimic_insulin"] = pd.read_csv(DATA / "mimic_insulin.csv")
    except Exception:
        pass

    return d


D = load()

if "kpi" not in D:
    st.error("⚠️ Essential project data files are missing in `app_data/`. Please run the export cells in the notebooks first.")
    st.stop()

K = D["kpi"]

st.title("🏥 PharmaLens: Hospital Readmission & Medication Analytics")
st.caption(
    "Retrospective analysis of the Diabetes 130-US Hospitals dataset (1999-2008) and MIMIC-IV clinical aggregates. "
    "Educational portfolio project: not clinically validated and not clinical decision support."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Risk factors", "Model performance", "MIMIC-IV Analytics", "Methodology and limits"]
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
        "- The model ranks patients usefully but only modestly (see *Model performance*)."
    )

# ------------------------------------------------------------ Risk factors
with tab2:
    labels = {
        "age": "Age group",
        "time_in_hospital": "Length of stay (days)",
        "med_bin": "Number of medications",
        "insulin": "Insulin (No / Steady / Up / Down)",
        "change": "Any medication change (Ch = changed)",
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
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Associations, not causal effects: sicker patients receive more drugs, more dose changes and "
        "longer stays. Check the meaning of discharge codes in the dataset's IDs_mapping file."
    )

# ------------------------------------------------------ Model performance
with tab3:
    st.subheader("Capacity analysis (sealed hold-out)")
    cap = D["cap_curve"].copy()
    cap["flag_pct"] = cap["flag_pct"].round().astype(int)

    pct = st.select_slider(
        "Follow-up capacity: share of discharges flagged (highest predicted risk first)",
        options=list(cap["flag_pct"]), value=10,
    )
    recall = float(cap.loc[cap["flag_pct"] == pct, "recall_pct"].iloc[0])
    lift = recall / pct

    # Use exact precision from capacity table if available
    cap_table = D.get("cap_table")
    if cap_table is not None and "precision_pct" in cap_table.columns and pct in cap_table["flag_pct"].values:
        precision = float(cap_table.loc[cap_table["flag_pct"] == pct, "precision_pct"].iloc[0])
    else:
        precision = lift * K["readmit_rate_pct"]

    m = st.columns(4)
    m[0].metric("Patients flagged", f"{round(K['holdout_n'] * pct / 100):,}")
    m[1].metric("Readmissions caught", f"{recall:.0f}%")
    m[2].metric("Lift vs random list", f"{lift:.2f}x")
    m[3].metric("Precision", f"{precision:.1f}%" if isinstance(precision, float) else f"{precision}%")

    cf = go.Figure()
    cf.add_trace(go.Scatter(x=cap["flag_pct"], y=cap["recall_pct"], mode="lines+markers",
                            name="HistGradientBoosting / Logistic Regression"))
    cf.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Random list",
                            line=dict(dash="dash")))
    cf.update_layout(xaxis_title="Patients flagged (%)", yaxis_title="Readmissions captured (%)")
    st.plotly_chart(cf, use_container_width=True)
    st.caption("Most flagged patients are not readmitted, so any intervention should be low-cost.")

    left, right = st.columns(2)
    with left:
        st.subheader("Calibration")
        cal = D["cal"]
        fc = px.line(cal, x="predicted", y="observed", color="model", markers=True)
        fc.add_trace(go.Scatter(x=[0, 0.75], y=[0, 0.75], mode="lines", name="Perfect",
                                line=dict(dash="dash", color="gray")))
        fc.update_layout(xaxis_title="Predicted probability", yaxis_title="Observed rate")
        st.plotly_chart(fc, use_container_width=True)
        st.caption("The unweighted model matches observed rates; the class-weighted one overstates risk.")
    with right:
        st.subheader("Precision-recall curve")
        fp = px.line(D["pr"], x="recall", y="precision")
        fp.add_hline(y=K["readmit_rate_pct"] / 100, line_dash="dash",
                     annotation_text="Baseline (prevalence)")
        st.plotly_chart(fp, use_container_width=True)
        st.caption(f"PR-AUC {K['holdout_pr_auc']}, ROC-AUC {K['holdout_roc_auc']}, "
                   f"Brier {K['holdout_brier']} (hold-out).")

    st.subheader("Model comparison (5-fold cross-validation on the development set)")
    mc = D["models"].copy()
    mc["ROC-AUC"] = mc.apply(lambda r: f"{r.roc_auc:.3f} +/- {r.roc_sd:.3f}", axis=1)
    mc["PR-AUC"] = mc.apply(lambda r: f"{r.pr_auc:.3f} +/- {r.pr_sd:.3f}", axis=1)
    st.dataframe(mc[["model", "ROC-AUC", "PR-AUC"]], hide_index=True)
    st.caption("All models perform within noise, so the simplest one was chosen.")

    st.subheader("Subgroup audit (race is not a model feature)")
    sub = D["sub"].query("n >= 10")
    sv = st.selectbox("Group by", sorted(sub["variable"].unique()))
    s = sub[sub["variable"] == sv]
    st.dataframe(s.drop(columns="variable"), hide_index=True)

# ---------------------------------------------------- MIMIC-IV Analytics
with tab4:
    st.subheader("🔬 MIMIC-IV Clinical & Medication Analytics")
    st.markdown("Aggregated clinical insights derived from the MIMIC-IV demo database (patient-level identifiers excluded).")

    if "mimic_kpi" in D:
        mk = D["mimic_kpi"]
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Total Admissions", f"{mk['admissions']:,}")
        mc2.metric("Total Patients", f"{mk['patients']:,}")
        mc3.metric("Diabetes Admissions", f"{mk['with_diabetes']:,}")
        mc4.metric("30-Day Unplanned Readmit", f"{mk['readmit_unplanned']:,}")

        st.markdown("#### Top Frequent Diagnoses (ICD-9 / ICD-10)")
        st.dataframe(D["mimic_diagnoses"], use_container_width=True, hide_index=True)

        st.markdown("#### Top Prescribed Therapeutic Drugs")
        st.dataframe(D["mimic_top_drugs"], use_container_width=True, hide_index=True)

        st.markdown("#### Insulin: Orders vs. Recorded Administrations (eMAR)")
        st.dataframe(D["mimic_insulin"], use_container_width=True, hide_index=True)
    else:
        st.warning("MIMIC aggregated data files not found. Please run the export cells in `02_mimic_sql.ipynb`.")

# ------------------------------------------------------------ Methodology
with tab5:
    st.markdown(
        """
**Cohort.** First encounter per patient (lowest `encounter_id`, a time proxy because the dataset has no dates),
excluding deaths and hospice discharges. Prediction point: moment of discharge.
Target: readmission within 30 days.

**Evaluation.** Model selection by 5-fold cross-validation on 80% of patients; a 20% hold-out was evaluated once.
Thresholds were set on development data only.

**Limitations**
- Historical data (1999-2008), administrative only, no detailed lab values or social factors.
- Modest discrimination (ROC-AUC about 0.64); predicted risks rarely exceed about 0.25.
- Retrospective, one dataset, no external validation; not clinical decision support.
- Observational associations only, no causal claims.

**Data governance.** This app reads small aggregate tables only. No patient-level data are included.
        """
    )