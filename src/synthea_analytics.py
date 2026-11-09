import json
from pathlib import Path
import zipfile
import pandas as pd

def analyze_synthea_data():
    print("--- Starting Synthea EHR/FHIR Healthcare Analytics ---")
    
    zip_path = Path("data/synthea_sample_data_csv_latest.zip")
    if not zip_path.exists():
        print(f"Synthea zip file not found at {zip_path}")
        return
    
    output_dir = Path("app_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        file_list = z.namelist()
        print(f"Total files in Synthea archive: {len(file_list)}")
        
        patient_file = next((f for f in file_list if 'patient' in f.lower() and f.endswith('.csv')), None)
        condition_file = next((f for f in file_list if 'condition' in f.lower() and f.endswith('.csv')), None)
        
        total_patients = 0
        top_conditions = []
        
        if patient_file:
            with z.open(patient_file) as f:
                df_patients = pd.read_csv(f)
                total_patients = len(df_patients)
                
        if condition_file:
            with z.open(condition_file) as f:
                df_conditions = pd.read_csv(f)
                # إعطاء الأولوية القصوى لعمود الوصف (description) لجلب النصوص الطبية لا الأكواد الرقمية
                desc_col = next((c for c in df_conditions.columns if 'description' in c.lower()), None)
                if not desc_col:
                    desc_col = next((c for c in df_conditions.columns if 'code' in c.lower()), df_conditions.columns[0])
                
                top_conditions = df_conditions[desc_col].dropna().value_counts().head(5).index.tolist()
                
        metrics = {
            "total_synthea_patients": total_patients,
            "top_fhir_conditions": top_conditions,
            "fhir_standards_supported": ["US Core Implementation Guide", "FHIR R4", "Clinical Encounters"]
        }
        
        with open(output_dir / "synthea_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
            
    print("Synthea EHR/FHIR analytics processed and saved successfully!")

if __name__ == "__main__":
    analyze_synthea_data()