import json
from pathlib import Path
import pandas as pd

def analyze_hrrp_data():
    print("--- Starting CMS HRRP FY 2026 Hospital Penalties Analytics ---")
    
    # البحث عن ملف الـ Excel الخاص بعقوبات المستشفيات لعام 2026 في مجلد data
    data_dir = Path("data")
    excel_files = list(data_dir.glob("*Hospital_Readmissions_Redu*.xlsx")) + list(data_dir.glob("*Hospital_Readmissions_Redu*.csv"))
    
    if not excel_files:
        print("CMS HRRP file not found in data/ directory.")
        return
    
    file_path = excel_files[0]
    print(f"Reading file: {file_path.name}")
    
    # قراءة الملف (إذا كان إكسل أو سي إس في)
    if file_path.suffix.endswith('.xlsx'):
        df = pd.read_excel(file_path, nrows=5000)
    else:
        df = pd.read_csv(file_path, nrows=5000, low_memory=False)
        
    output_dir = Path("app_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # استخراج بعض المؤشرات الأساسية (مثل عدد المستشفيات، متوسط نسب العقوبات أو معدلات إعادة التنويم الزائدة)
    total_hospitals = len(df)
    
    # البحث عن أعمدة تخص اسم المستشفي، الولاية، أو نسبة التخفيض إن وجدت
    col_hospital = next((c for c in df.columns if 'hospital' in c.lower() or 'facility' in c.lower()), df.columns[0])
    col_state = next((c for c in df.columns if 'state' in c.lower()), None)
    
    top_penalized = df[col_hospital].dropna().head(5).tolist() if col_hospital else []
    
    metrics = {
        "total_hospitals_evaluated": total_hospitals,
        "sample_top_hospitals": top_penalized,
        "hrrp_program_year": "FY 2026",
        "description": "CMS Hospital Readmissions Reduction Program financial penalties and excess readmission ratios."
    }
    
    with open(output_dir / "hrrp_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("CMS HRRP metrics successfully processed and saved!")

if __name__ == "__main__":
    analyze_hrrp_data()