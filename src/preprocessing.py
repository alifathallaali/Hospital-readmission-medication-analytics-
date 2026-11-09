import pandas as pd
import numpy as np

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    تحميل وتنظيف بيانات مستشفيات السكري أو بيانات السجلات الطبية.
    """
    # قراءة البيانات مع اعتبار '?' قيم مفقودة
    df = pd.read_csv(filepath, na_values=['?'])
    
    # 1. إزالة التكرارات بناءً على معرف المريض (إذا وجد)
    if 'patient_nbr' in df.columns:
        df = df.drop_duplicates(subset=['patient_nbr'], keep='first')
        
    # 2. استبعاد الحالات الحرجة أو الوفيات إذا لزم الأمر (مثلاً كود الوفاة أو الرعاية التلطيفية)
    # مثال: إذا كان هناك عمود لرمز الخروج (discharge_disposition_id)
    if 'discharge_disposition_id' in df.columns:
        # استبعاد الوفيات أو دور الرعاية (مثلاً القيم 11، 19، 20، 21)
        hospice_death_codes = [11, 19, 20, 21]
        df = df[~df['discharge_disposition_id'].isin(hospice_death_codes)]
        
    # 3. معالجة الأعمدة النصية أو الرقمية الناقصة الأساسية
    # ملء القيم العددية الناقصة أو تركها للنموذج حسب الحاجة
    
    return df

if __name__ == "__main__":
    # تجربة الوظيفة محلياً عبر سطر الأوامر (cmd/powershell)
    print("preprocessing module ready.")