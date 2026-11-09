import json
from pathlib import Path
import pandas as pd

def analyze_medicaid_data():
    print("--- Starting Phase 4: Medicaid Drug Utilization & Economic Analytics ---")
    
    # تحديد مسار ملف البيانات الضخمة (تأكد من مطابقة الاسم الدقيق لديك)
    data_path = Path("data/medicaid_sdud2025_updatedjuly2026.csv")
    if not data_path.exists():
        print(f"Medicaid data file not found at {data_path}")
        return
    
    print("Loading a sample/chunk of Medicaid data for analysis...")
    # نظراً لحجم الملف الكبير، نقوم بقراءته على دفعات أو استخراج أعمدة الاهتمام
    # الأعمدة الشائعة غالباً تشمل: Utilization Type, State, Product Name (Drug Name), Totals, etc.
    chunk_size = 100000
    chunks = []
    
    for chunk in pd.read_csv(data_path, chunksize=chunk_size, low_memory=False):
        # تصفية البيانات لتشمل أدوية السكري أو الفئات المزمنة أو أخذ عينة تحليلية
        # كمثال: تصفية البيانات المتعلقة بالأدوية المتاحة أو التركيز على إجمالي المدفوعات واستخدامات الوحدات
        chunks.append(chunk.head(5000)) # نأخذ عينة تمثيلية لكل جزء لتفادي الضغط على الذاكرة محلياً
        break # نكتفي بالجزء الأول كمرحلة أولية للاستكشاف
        
    df_sample = chunks[0]
    
    # استخراج ملخص تحليلي (KPIs and Top Drugs by Utilization/Cost)
    # البحث عن أعمدة الأسماء والتكاليف ديناميكياً
    col_drug = next((c for c in df_sample.columns if 'drug' in c.lower() or 'product' in c.lower() or 'name' in c.lower()), df_sample.columns[0])
    col_cost = next((c for c in df_sample.columns if 'amount' in c.lower() or 'cost' in c.lower() or 'reimb' in c.lower()), None)
    
    summary_data = {
        "total_records_analyzed": len(df_sample),
        "primary_drug_column": col_drug,
        "cost_column_available": col_cost is not None
    }
    
    # حفظ النتائج في app_data لاستخدامها لاحقاً في لوحة التحكم Streamlit
    output_dir = Path("app_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "medicaid_metrics.json", "w") as f:
        json.dump(summary_data, f, indent=4)
        
    print("Phase 4 exploratory metrics generated and saved successfully!")

if __name__ == "__main__":
    analyze_medicaid_data()