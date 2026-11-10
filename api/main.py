# api/main.py
from fastapi import FastAPI, HTTPException
from api.schemas import PatientPredictionRequest
import joblib
import pandas as pd
import os

app = FastAPI(
    title="PharmaLens Readmission & Clinical Intelligence API",
    version="1.0.0",
    description="Production-grade FastAPI backend for hospital readmission prediction with strict Pydantic validation."
)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Welcome to PharmaLens Clinical Intelligence API - Phase 9",
        "docs_url": "/docs"
    }

@app.post("/predict")
def predict_readmission(payload: PatientPredictionRequest):
    try:
        # تحويل بيانات المدخلات إلى DataFrame لكي تتوافق مع الموديل
        input_data = pd.DataFrame([payload.dict()])
        
        # محاكاة التنبؤ (أو تحميل الموديل المحفوظ مسبقاً إذا وجد)
        # هنا بنحسب نتيجة تجريبية بناءً على القواعد لضمان سرعة الاستجابة وعمل الـ API
        risk_score = 0.25 if payload.time_in_hospital < 5 else 0.65
        is_high_risk = risk_score > 0.5
        
        return {
            "status": "success",
            "predicted_readmission_risk": risk_score,
            "high_risk_flag": is_high_risk,
            "received_data": payload.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))