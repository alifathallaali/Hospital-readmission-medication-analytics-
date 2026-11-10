# Target Data Architecture & Medallion Foundation - Phase 8

## 1. Architectural Vision
To transition PharmaLens into an enterprise-grade Clinical Intelligence Platform, we implement a structured Medallion Architecture (Bronze, Silver, Gold) ensuring robust data provenance, quality reporting, and reproducible feature extraction.

## 2. Medallion Layers
- **Bronze Layer (Raw Ingestion):** Immutable storage of raw sources (CSV extracts, SQLite dumps, FHIR bundles, ASCII FAERS files). Strict schema preservation without destructive transformations.
- **Silver Layer (Cleaning & Standardization):** Type casting, missing value handling (`?` to `NaN`), deduplication (`PRIMARYID`, `encounter_id`), and normalization of terminologies across disparate sources.
- **Gold Layer (Analytical Marts):** Aggregated feature stores, cohort definitions, and optimized model-ready tables (e.g., 69,970 patient readmission cohort, regimen complexity features, capacity curves).

## 3. Governance & Provenance
- **Source Registry (`app_data/manifest.json`):** Machine-readable lineage ledger tracking dataset version, license, download source, ingestion timestamp, and corresponding output artifacts.
- **Data Quality & Contracts:** Automated `pytest` suites validating schema constraints, missingness bounds, and preventing target leakage prior to downstream model ingestion.