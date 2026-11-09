import json
from pathlib import Path
import pandas as pd

def analyze_medicaid_data():
    print("--- Starting Enhanced Phase 4: Medicaid Drug Utilization Analytics ---")
    
    data_path = Path("data/medicaid_sdud2025_updatedjuly2026.csv")
    if not data_path.exists():
        print(f"Medicaid data file not found at {data_path}")
        return
    
    print("Reading and processing Medicaid dataset sample...")
    chunk_size = 100000
    for chunk in pd.read_csv(data_path, chunksize=chunk_size, low_memory=False):
        df = chunk.head(50000).copy()
        break
        
    # توحيد أسماء الأعمدة لتصبح آمنة وسهلة المعالجة
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    
    drug_col = next((c for c in df.columns if 'drug' in c or 'product' in c or 'name' in c), df.columns[0])
    state_col = next((c for c in df.columns if 'state' in c), None)
    
    summary_df = df[[state_col, drug_col]].dropna() if state_col else df[[drug_col]].dropna()
    summary_df.columns = ['state', 'drug'] if state_col else ['drug']
    if not state_col:
        summary_df['state'] = 'National'
        
    # تجميع البيانات لاستخراج الأكثر طلباً حسب الدولة والدواء
    agg_df = summary_df.groupby(['state', 'drug']).size().reset_index(name='utilization_count')
    agg_df = agg_df.sort_values(by='utilization_count', ascending=False)
    
    output_dir = Path("app_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # حفظ الملف لتتم قراءته تفاعلياً في Streamlit
    agg_df.to_csv(output_dir / "medicaid_filtered_summary.csv", index=False)
    
    metrics = {
        "total_records_analyzed": len(df),
        "unique_drugs": int(agg_df['drug'].nunique()),
        "unique_states": int(agg_df['state'].nunique()) if state_col else 1
    }
    with open(output_dir / "medicaid_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Phase 4 interactive data processed and saved successfully!")

if __name__ == "__main__":
    analyze_medicaid_data()