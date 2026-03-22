from fastapi import FastAPI
import requests

SUPABASE_URL = "https://kzqcznveyxallyonedls.supabase.co"
SUPABASE_KEY = "sbpublishablerUhjaGNhHHlkwHis22FqkgmG2Fswbz"

app = FastAPI()

# قراءة بيانات مريض واحد من Supabase
def get_patient_by_id(patid: int):
    url = f"{SUPABASE_URL}/rest/v1/tblpatient?patid=eq.{patid}&select=*"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }
    response = requests.get(url, headers=headers)
    return response.json()

# تحديث حالة المريض في Supabase
def update_patient_status(patid: int, status: str):
    url = f"{SUPABASE_URL}/rest/v1/tblpatient?patid=eq.{patid}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"patstatus": status}
    requests.patch(url, headers=headers, json=payload)

# Endpoint للتنبؤ بحالة ECG
@app.post("/predict/ecg/{patid}")
def predict_ecg(patid: int):
    patient = get_patient_by_id(patid)
    if not patient:
        return {"error": "المريض غير موجود"}

    age = patient[0].get("patage")
    if age is None:
        return {"error": "لا توجد بيانات عمر"}

    # منطق مبسط للتجربة: إذا العمر أكبر من 30 → خطر
    status = "خطر" if age > 30 else "طبيعي"

    # تحديث Supabase
    update_patient_status(patid, status)

    return {"patid": patid, "status": status}

# Endpoint GET إضافي حتى تقدر تستدعيه من المتصفح مباشرة
@app.get("/predict/ecg/{patid}")
def predict_ecg_get(patid: int):
    return predict_ecg(patid)    # مثال: تدريب نموذج بسيط على العمر مقابل حالة الطوارئ
    data = get_patient_by_id(patid)
    X = [[p["patage"]] for p in data if p["patage"] is not None]
    y = [1 if p["isemergency"] else 0 for p in data if p["patage"] is not None]

    if not X or not y:
        return {"error": "لا توجد بيانات كافية"}

    model = LogisticRegression().fit(X, y)

    pred = model.predict([[patient[0]["patage"]]])[0]
    status = "خطر" if pred == 1 else "طبيعي"

    update_patient_status(patid, status)

    return {"patid": patid, "prediction": int(pred), "status": status}
