from fastapi import FastAPI
import requests

SUPABASE_URL = "https://kzqcznveyxallyonedls.supabase.co"
SUPABASE_KEY = "sbpublishablerUhjaGNhHHlkwHis22FqkgmG2Fswbz"

app = FastAPI()

# قراءة بيانات مريض واحد من Supabase
def get_patient_by_id(pat_id: int):
    url = f"{SUPABASE_URL}/rest/v1/tblpatient?pat_id=eq.{pat_id}&select=*"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }
    response = requests.get(url, headers=headers)
    return response.json()

# تحديث حالة المريض في Supabase
def update_patient_status(pat_id: int, status: str):
    url = f"{SUPABASE_URL}/rest/v1/tblpatient?pat_id=eq.{pat_id}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"pat_status": status}
    requests.patch(url, headers=headers, json=payload)

@app.post("/predict/ecg/{pat_id}")
def predict_ecg(pat_id: int):
    patient = get_patient_by_id(pat_id)
    if not patient:
        return {"error": "المريض غير موجود"}

    age = patient[0].get("pat_age")
    if age is None:
        return {"error": "لا توجد بيانات عمر"}

    # منطق مبسط للتجربة: إذا العمر أكبر من 30 → خطر
    status = "خطر" if age > 30 else "طبيعي"

    # تحديث Supabase
    update_patient_status(pat_id, status)

    return {"pat_id": pat_id, "status": status}

# Endpoint GET إضافي حتى تقدر تستدعيه من المتصفح مباشرة
@app.get("/predict/ecg/{pat_id}")
def predict_ecg_get(pat_id: int):
    return predict_ecg(pat_id)    if age is None:
        return {"error": "لا توجد بيانات عمر"}

    # منطق مبسط للتجربة: إذا العمر أكبر من 30 → خطر
    status = "خطر" if age > 30 else "طبيعي"

    # تحديث Supabase
    update_patient_status(pat_id, status)

    return {"pat_id": pat_id, "status": status}

# Endpoint GET إضافي حتى تقدر تستدعيه من المتصفح مباشرة
@app.get("/predict/ecg/{pat_id}")
def predict_ecg_get(pat_id: int):
    return predict_ecg(pat_id)
