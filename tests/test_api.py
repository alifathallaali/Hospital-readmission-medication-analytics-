# tests/test_api.py
from fastapi.testclient import TestClient
from api.main import app

# إنشاء عميل اختبار للـ FastAPI
client = TestClient(app)

def test_read_root():
    """اختبار الصفحة الرئيسية للـ API والتأكد من أنها تعمل"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "PharmaLens" in data["message"]

def test_predict_readmission_success():
    """اختبار نقطة التنبؤ ببيانات صحيحة ومطابقة لشروط Pydantic"""
    payload = {
        "age": 60,
        "time_in_hospital": 4,
        "num_lab_procedures": 30,
        "num_procedures": 1,
        "num_medications": 12,
        "number_inpatient": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "predicted_readmission_risk" in data
    assert isinstance(data["high_risk_flag"], bool)

def test_predict_readmission_validation_error():
    """اختبار رفض البيانات غير الصالحة (مثل عمر غير منطقي أو سالب) بفضل Pydantic"""
    invalid_payload = {
        "age": 150,  # عمر أكبر من المسموح (أقصى حد 120 حسب الـ schema)
        "time_in_hospital": 2,
        "num_lab_procedures": 20,
        "num_procedures": 0,
        "num_medications": 5,
        "number_inpatient": 0
    }
    response = client.post("/predict", json=invalid_payload)
    # يجب أن يرجع خطأ 422 Unprocessable Entity بسبب فشل التحقق في Pydantic
    assert response.status_code == 422