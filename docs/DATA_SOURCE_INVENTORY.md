# Data Source Inventory & Metadata Catalog - Phase 8

## Overview
This document inventories all six healthcare data sources integrated into the platform, defining their source origin, version, access conditions, license, and processing status.

## 1. Diabetes 130-US Hospitals (1999–2008)
- **Source:** UCI Machine Learning Repository / Kaggle (Strack et al., 2014)
- **License / Access:** Open academic use.
- **Version / Scale:** 101,766 raw encounters, 50 columns.
- **Processing Status:** Cleaned, missing `?` handled, deduplicated to 69,970 unique patient encounters (first encounter), hospice/death exclusions applied.
- **Outputs:** Primary 30-day readmission model, capacity curves, subgroup audits.

## 2. MIMIC-IV Clinical Database Demo v2.2
- **Source:** PhysioNet / Kaggle.
- **License / Access:** Credentialed PhysioNet open demo license.
- **Version / Scale:** 100 patients, 275 admissions, 31 SQLite relational tables.
- **Processing Status:** Validated row counts across all tables; compared prescription orders against eMAR administrations.
- **Outputs:** SQL exploration exports (`mimic_*.csv`, `mimic_kpis.json`).

## 3. Medicaid State Drug Utilization Data (SDUD)
- **Source:** Medicaid.gov public domain.
- **Version / Scale:** Multi-state federal drug utilization records.
- **Processing Status:** Chunked parsing implemented to prevent memory truncation.
- **Outputs:** State-level spending and utilization summaries (`medicaid_filtered_summary.csv`).

## 4. FDA FAERS (Adverse Event Reporting System)
- **Source:** U.S. Food and Drug Administration.
- **Version / Scale:** Quarterly ASCII extracts (DEMO, DRUG, REAC, OUTC, THER).
- **Processing Status:** Case-level deduplication (`PRIMARYID`), drug-role normalization.
- **Outputs:** Signal detection metrics (`faers_metrics.json`).

## 5. Synthea EHR / FHIR Synthetic Data
- **Source:** The Mitre Corporation (Open source synthetic generator).
- **Version / Scale:** Synthetic patient bundles (Patient, Encounter, Condition, Medication, Observation, Procedure).
- **Processing Status:** Relationship mapping and resource coverage summary.
- **Outputs:** FHIR metrics summary (`synthea_metrics.json`).

## 6. CMS Hospital Readmissions Reduction Program (HRRP)
- **Source:** Centers for Medicare & Medicaid Services.
- **Version / Scale:** National hospital penalty and excess readmission ratio tables.
- **Processing Status:** Provider ID (CCN) normalization and metric calculations.
- **Outputs:** Penalty distributions and tiering (`hrrp_metrics.json`).