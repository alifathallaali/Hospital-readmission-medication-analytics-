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
    "advanced_model_metrics.json", "medicaid_metrics.json", "medicaid_filtered_summary.csv",
    "faers_metrics.json", "synthea_metrics.json", "hrrp_metrics.json",
]

# Stop with a clear message instead of crashing if an export is missing
missing = [f for f in REQUIRED if not (DATA / f).exists()]
if missing:
    st.error("Missing data files in `app_data/`: " + ", ".join(missing)
             + ". Re-run the export cells or analysis scripts.")
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
        "medicaid_filtered": pd.read_csv(DATA / "medicaid_filtered_summary.csv"),
    }
    with open(DATA / "kpis.json") as f:
        d["kpi"] = json.load(f)
    with open(DATA / "mimic_kpis.json") as f:
        d["mk"] = json.load(f)
    with open(DATA / "model_metrics.json") as f:
        d["model_metrics"] = json.load(f)
    with open(DATA / "advanced_model_metrics.json") as f:
        d["advanced_model_metrics"] = json.load(f)
    with open(DATA / "medicaid_metrics.json") as f:
        d["medicaid_metrics"] = json.load(f)
    with open(DATA / "faers_metrics.json") as f:
        d["faers_metrics"] = json.load(f)
    with open(DATA / "synthea_metrics.json") as f:
        d["synthea_metrics"] = json.load(f)
    with open(DATA / "hrrp_metrics.json") as f:
        d["hrrp_metrics"] = json.load(f)
    return d


D = load()
K = D["kpi"]
MK = D["mk"]
MM = D["model_metrics"]
AMM = D["advanced_model_metrics"]
MED = D["medicaid_metrics"]
FAERS = D["faers_metrics"]
SYN = D["synthea_metrics"]
HRRP = D["hrrp_metrics"]


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
    "MIMIC-IV demo, Medicaid Pharma Market, FDA FAERS, Synthea EHR/FHIR, and CMS HRRP hospital penalties. Educational portfolio project."
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(
    ["Overview", "Risk factors", "Advanced Meds & Phase 3", "Model performance", "MIMIC-IV demo (SQL)", "Methodology and limits", "Pharma Market (Phase 4)", "FAERS Safety (Phase 5)", "EHR & FHIR (Synthea)", "CMS HRRP Penalties (Phase 7)"]
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
        "enhanced with hyperparameter-tuned gradient boosting."
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
        st.write("**Optimized Hyperparameters:**", AMM["best_params"])

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
Phase 3 introduced hyperparameter-tuned gradient boosting incorporating advanced medication and regimen complexity metrics.

**Limitations**
- Historical data (1999-2008), administrative only, no detailed lab values or social factors.
- Retrospective, one dataset, no external validation; not clinical decision support.
- Observational associations only, no causal claims.
- The MIMIC demo has only 100 patients and is exploratory.

**Data governance.** This app reads small aggregate tables only. No patient-level data are included.
        """
    )

# ------------------------------------------------------------ Pharma Market (Phase 4)
with tab7:
    st.subheader("Phase 4: Medicaid Pharma Market & Utilization Analytics")
    st.markdown(
        "Analyzing large-scale drug utilization, prescription patterns, and economic indices from Medicaid data."
    )
    
    p4_col1, p4_col2, p4_col3 = st.columns(3)
    with p4_col1:
        st.metric("Analyzed Records Sample", f"{MED.get('total_records_analyzed', 0):,}")
    with p4_col2:
        st.metric("Unique Drugs Tracked", f"{MED.get('unique_drugs', 0):,}")
    with p4_col3:
        st.metric("Regions / States", f"{MED.get('unique_states', 0):,}")
        
    st.markdown("---")
    st.subheader("Interactive Drug Utilization Explorer & Regional Filtering")
    
    df_med = D["medicaid_filtered"]
    selected_state = st.selectbox("Filter by State / Region", options=sorted(df_med['state'].unique()))
    
    filtered_df = df_med[df_med['state'] == selected_state].sort_values(by='utilization_count', ascending=False).head(10)
    
    fig_med = px.bar(
        filtered_df, 
        x='utilization_count', 
        y='drug', 
        orientation='h',
        labels={'utilization_count': 'Utilization Count', 'drug': 'Drug / Product Code'},
        title=f"Top 10 Utilized Drugs in {selected_state}"
    )
    fig_med.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_med, use_container_width=True)
    
    st.info(
        "💡 **Enterprise Insight:** Dynamic filtering enables health economists and policy makers to drill down into "
        "regional prescription volumes and identify high-demand pharmaceutical classes instantly."
    )

# ------------------------------------------------------------ FAERS Safety (Phase 5)
with tab8:
    st.subheader("Phase 5: FDA FAERS Pharmacovigilance & Safety Analytics")
    st.markdown(
        "Analyzing adverse event reporting systems, post-market drug safety signals, and incident reports."
    )
    
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        st.metric("Total Archive Files", FAERS.get("total_files", 0))
    with f_col2:
        st.metric("Sample Rows Processed", f"{FAERS.get('sample_rows', 0):,}")
    with f_col3:
        st.metric("Top Tracked Products", len(FAERS.get("top_reported_drugs", [])))
        
    st.subheader("Top Reported Drugs in FAERS Sample")
    top_drugs_list = FAERS.get("top_reported_drugs", [])
    if top_drugs_list:
        df_faers_top = pd.DataFrame({"Drug Name": top_drugs_list})
        st.dataframe(df_faers_top, hide_index=True)
    else:
        st.info("No drug samples found in the current FAERS extract.")
        
    st.info(
        "💡 **Pharmacovigilance Insight:** Integrating FDA FAERS post-market safety data allows clinical researchers "
        "to monitor adverse drug reactions and evaluate real-world safety profiles alongside hospital readmission patterns."
    )

# ------------------------------------------------------------ EHR & FHIR (Synthea - Phase 6)
with tab9:
    st.subheader("Phase 6: Synthea EHR & FHIR Healthcare Standards Analytics")
    st.markdown(
        "Exploring synthetic electronic health record data adhering to HL7 FHIR R4 and US Core standards, "
        "capturing comprehensive patient journeys, clinical conditions, and encounters."
    )
    
    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        st.metric("Total Synthea Patients", f"{SYN.get('total_synthea_patients', 0):,}")
    with s_col2:
        st.metric("Supported FHIR Standards", len(SYN.get('fhir_standards_supported', [])))
    with s_col3:
        st.metric("Top Conditions Tracked", len(SYN.get('top_fhir_conditions', [])))
        
    st.subheader("Top Recorded Clinical Conditions (FHIR Standards)")
    top_conds = SYN.get("top_fhir_conditions", [])
    if top_conds:
        df_conds = pd.DataFrame({"Condition Description": top_conds})
        st.dataframe(df_conds, hide_index=True)
    else:
        st.info("No clinical conditions found in the current Synthea sample.")
        
    st.info(
        "💡 **Interoperability Insight:** Leveraging FHIR-compliant synthetic datasets allows healthcare systems "
        "to test multi-source data integration, ensuring seamless exchange of clinical summaries and continuity of care."
    )

# ------------------------------------------------------------ CMS HRRP Penalties (Phase 7)
with tab10:
    st.subheader("Phase 7: CMS HRRP Hospital Penalties & Financial Impact")
    st.markdown(
        "Analyzing Hospital Readmissions Reduction Program (HRRP) metrics, excess readmission ratios, "
        "and reimbursement penalties under CMS guidelines."
    )
    
    h_col1, h_col2, h_col3 = st.columns(3)
    with h_col1:
        st.metric("Hospitals Evaluated", f"{HRRP.get('total_hospitals_evaluated', 0):,}")
    with h_col2:
        st.metric("Program Evaluation Cycle", HRRP.get("hrrp_program_year", "FY 2026"))
    with h_col3:
        st.metric("Sample Tracked Facilities", len(HRRP.get("sample_top_hospitals", [])))
        
    st.subheader("Sample Evaluated Hospitals Reference")
    sample_hospitals = HRRP.get("sample_top_hospitals", [])
    if sample_hospitals:
        df_hrrp_top = pd.DataFrame({"Hospital Name": sample_hospitals})
        st.dataframe(df_hrrp_top, hide_index=True)
    else:
        st.info("No hospital sample records found in current HRRP extract.")
        
    st.info(
        "💡 **Financial & Regulatory Insight:** Connecting predictive readmission risk models with CMS financial penalties "
        "empowers health systems to proactively mitigate revenue risks and optimize care coordination resources."
    )