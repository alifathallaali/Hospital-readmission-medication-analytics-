# Phase 8 Execution Plan - Platform Audit & Data Integrity Recovery

## 1. Objectives & Scope
Phase 8 establishes an end-to-end audit of data hygiene, methodological rigor, and model validation across all project pipelines. No UI redesigns or speculative features are permitted until data foundations are fully verified.

## 2. Step-by-Step Execution Roadmap
- **Step 1: Repository Safety & Branching**
  - Branch safely into `feature/phase-8-audit-recovery` from baseline commit `660d6f1f7b50f2dc80e1bbb6d103db7a3e571dac`.
  - Enforce strict `.gitignore` rules for raw restricted datasets (`data/`, `*.db`).
- **Step 2: Critical Modeling Audit & Correction**
  - Remove all random synthetic target fallbacks in `src/model_pipeline.py`.
  - Isolate MIMIC aggregate tables from patient-level modeling data.
  - Shift preprocessing downstream of train/test splits to eliminate data leakage.
  - Unify cohort exclusions (removing hospice/death codes) to anchor the 69,970 patient cohort.
- **Step 3: Source-by-Source Data Audit**
  - Document and validate all six healthcare data sources (Diabetes 130-US, MIMIC-IV, Medicaid SDUD, FDA FAERS, Synthea, CMS HRRP).
- **Step 4: Unified Data Foundation & Manifest**
  - Establish Medallion Architecture guidelines and generate `app_data/manifest.json`.
- **Step 5: Testing & CI Validation**
  - Execute automated `pytest` suite ensuring 12/12 passing checks for consistency, schema contracts, and KPI bounds.