# api/schemas.py
from pydantic import BaseModel, Field

class PatientPredictionRequest(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Patient age in years")
    time_in_hospital: int = Field(..., ge=1, le=14, description="Days in hospital")
    num_lab_procedures: int = Field(..., ge=0, description="Number of lab tests performed")
    num_procedures: int = Field(..., ge=0, description="Number of non-lab procedures")
    num_medications: int = Field(..., ge=0, description="Number of distinct medications administered")
    number_inpatient: int = Field(..., ge=0, description="Number of inpatient visits prior to year")