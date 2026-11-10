# test_api.py
import requests

# عنوان الـ API المحلي اللي شغال عليه FastAPI
url = "http://127.0.0.1:8000/predict"

# بيانات تجريبية لمريض سنرسلها لفحصها
patient_payload = {
    "age": 55,
    "time_in_hospital": 6,
    "num_lab_procedures": 42,
    "num_procedures": 2,
    "num_medications": 18,
    "number_inpatient": 1
}

print("إرسال طلب فحص بيانات المريض إلى السيرفر...")
response = requests.post(url, json=patient_payload)

print(f"رمز الاستجابة (Status Code): {response.status_code}")
print("رد السيرفر (Response JSON):")
print(response.json())