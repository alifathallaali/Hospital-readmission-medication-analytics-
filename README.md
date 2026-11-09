# Hospital Readmission & Medication Analytics

An end-to-end healthcare analytics and clinical commercial intelligence project: data cleaning and cohort validation, statistical analysis, 30-day readmission risk modeling evaluated on a sealed hold-out, advanced regimen complexity analytics (Phase 2), and SQL exploration of EHR-structured data (MIMIC-IV demo).

**Author:** Ali Aboelsaad | Pharmacist transitioning into healthcare data science | [LinkedIn](https://www.linkedin.com/in/ali-aboelsaad-acim) | [Live Dashboard](https://alifathallaali-hospital-readmission-medication-a-apphome-rjqfxi.streamlit.app)

## Business question
Hospitals cannot follow up with every discharged patient. Which patients are most likely to return within 30 days, and how many readmissions could a limited follow-up team reach by focusing on the highest-risk patients?

## Skills demonstrated
Python (pandas, scikit-learn, modular `src/` pipelines), SQL (joins, CTEs, window functions), statistical testing, cohort design and leakage prevention, model evaluation on imbalanced data (cross-validation, calibration, precision-recall, capacity analysis), subgroup auditing, automated metrics serialization, and clinical/medication data interpretation.

## Data & Repository Architecture
| Dataset | Use | Source |
|---|---|---|
| Diabetes 130-US Hospitals (1999-2008), 101,766 encounters, 50 columns | Statistics, machine learning & Phase 2 advanced medication modeling | UCI / Kaggle (Strack et al., 2014) |
| MIMIC-IV Clinical Database Demo v2.2 (100 patients, 275 admissions) | SQL exploration | PhysioNet / Kaggle |

Raw data is **not** included in this repository. 

### Repository Structure
- `src/`: Modular Python scripts (`data_loader.py`, `medication_features.py`, `model_pipeline.py`) implementing the Phase 2 training pipeline and feature engineering.
- `notebooks/`: Exploratory data analysis, SQL queries, and Phase 2 advanced medication modeling (`03_advanced_medication_features.ipynb`).
- `app/`: Streamlit dashboard application (`Home.py`).
- `app_data/`: Pre-computed aggregate metrics, KPIs, and serialized JSON exports (`model_metrics.json`, `advanced_model_metrics.json`).
- `tests/`: Automated unit tests for medication feature engineering and pipeline consistency (`test_consistency.py`, `medication_features_2.py`).

## Part 1: Readmission analysis and baseline model

### Cohort definition
| Item | Definition |
|---|---|
| Population | Inpatient encounters of diabetic patients (130 US hospitals, 1999-2008) |
| Unit | One encounter per patient (first encounter) |
| Prediction point | Moment of discharge (only information known at discharge is used) |
| Target | 1 if readmitted within 30 days (`readmitted = "<30"`), otherwise 0 |
| Exclusions | Patients who died or went to hospice; invalid gender; columns with one value |
| Final cohort | 69,970 patients (from 101,766 encounters), 9.0% readmitted within 30 days |

**Time-order assumption.** The dataset has no dates, so "first encounter" is the lowest `encounter_id`. Prior inpatient visits (`number_inpatient`) increased between consecutive `encounter_id`s of the same patient in 73.7% of 30,248 pairs, supporting the time-order assumption.

### Key findings
- **Prior inpatient stays** are the strongest signal: readmission rises from 8.1% (none) to 26.5% (3 or more).
- **Discharge destination:** transfers to rehabilitation or other care facilities carry 20-26% readmission rates versus 6.9% for discharge home.
- **Medications & Regimen Changes:** any dose change is associated with a 10.1% readmission rate vs 8.6% without. Sicker patients receive more drugs and more changes (observational association, not causal).

### Model selection & Evaluation (5-fold cross-validation)
All baseline models performed within noise of each other (ROC-AUC ~0.645), so the simplest interpretable model (**logistic regression**) was selected. 
- **Sealed Hold-out Evaluation:** ROC-AUC of 0.640 (95% CI 0.626-0.659); PR-AUC of 0.166; Brier score: 0.0793.
- **Capacity Analysis:** Flagging the top 10% highest-risk patients captures **22.2%** of all 30-day readmissions with a precision of 19.9% (2.22x lift vs random).

#### Capacity Table Metrics:
| Flag Top % | Patients Flagged | Recall % | Lift |
|---|---|---|---|
| Top 1% | 699 | 13.8% | 2.76x |
| Top 2% | 1,399 | 20.5% | 2.05x |
| Top 4% | 2,798 | 31.2% | 1.56x |
| Top 10% | 6,997 | 37.4% | 1.87x |

---

## Part 2: Phase 2 - Advanced Medication & Regimen Complexity Analytics
The Phase 2 expansion integrates rigorous pharmaceutical care indicators into the machine learning workflow:
- **Modular Pipeline (`src/`):** Encapsulated feature engineering and model training into reusable, testable scripts (`medication_features.py`, `model_pipeline.py`, `advanced_model.py`).
- **Regimen Volatility & Polypharmacy Indices:** Captures prescription changes, polypharmacy stratification, and high-risk glycemic transition patterns at discharge.
- **Automated Metrics Serialization:** Model performance and operational metadata are automatically exported to `app_data/model_metrics.json` and `advanced_model_metrics.json` for seamless dashboard integration.

---

## Part 3: MIMIC-IV demo SQL & Dashboard
- **SQL Validation:** Validated 31 SQLite tables against source CSVs; uncovered that medication *orders* overstate actual administration (e.g., 134 insulin orders vs 83 recorded eMAR administrations).
- **Streamlit Dashboard (`app/Home.py`):** Interactive web application hosted live featuring multiple views:
  1. *Overview & KPIs*
  2. *Risk Factors*
  3. *Advanced Meds (Phase 2)* (Featuring integrated model performance metrics and polypharmacy insights)
  4. *Model Performance & Capacity Curves*
  5. *MIMIC-IV Demo (SQL)*
  6. *Methodology & Limits*

## Limitations
- Historical data (1999-2008), administrative only, lacking detailed lab values or social determinants.
- Modest discrimination (ROC-AUC ~0.64); observational associations only, not clinical decision support.
- MIMIC demo represents a small exploratory sample (100 patients) and is not clinically representative.