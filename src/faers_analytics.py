import json
from pathlib import Path
import zipfile
import pandas as pd

def analyze_faers_data():
    print("--- Starting FAERS Pharmacovigilance & Adverse Event Analytics ---")
    
    zip_path = Path("data/faers_ascii_2026q3.zip")
    if not zip_path.exists():
        print(f"FAERS zip file not found at {zip_path}")
        return
    
    output_dir = Path("app_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        file_list = z.namelist()
        drug_files = [f for f in file_list if 'drug' in f.lower() and (f.endswith('.txt') or f.endswith('.DAT'))]
        reac_files = [f for f in file_list if 'reac' in f.lower() and (f.endswith('.txt') or f.endswith('.DAT'))]
        
        sample_rows = 0
        top_drugs = []
        
        if drug_files:
            with z.open(drug_files[0]) as f_drug:
                df_drug = pd.read_csv(f_drug, sep='$', encoding='latin1', low_memory=False, nrows=10000)
                sample_rows = len(df_drug)
                
                # البحث عن عمود يحتوي على اسم الدواء صراحة
                col_name = next((c for c in df_drug.columns if 'drugname' in c.lower() or 'name' in c.lower()), df_drug.columns[1])
                
                top_drugs = df_drug[col_name].value_counts().head(5).index.tolist()
                
        metrics = {
            "total_files": len(file_list),
            "sample_rows": sample_rows,
            "top_reported_drugs": top_drugs
        }
        
        with open(output_dir / "faers_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
            
    print("FAERS metrics successfully processed and saved!")

if __name__ == "__main__":
    analyze_faers_data()