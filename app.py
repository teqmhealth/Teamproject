from fastapi import FastAPI
import requests, json
from datetime import datetime, timezone
import ecg_model

app = FastAPI()

# إعدادات Supabase
SUPABASE_URL = "https://kzqcznveyxallyonedls.supabase.co"
API_KEY = "sb_publishable_rUhjaGNhHHlkwHis22Fqkg_mG2Fswbz"
HEADERS = {
    "apikey": API_KEY,
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# قراءة القراءات الخام من جدول tbl_reading
def fetch_readings(pat_id):
    url = f"{SUPABASE_URL}/rest/v1/tbl_reading?pat_id=eq.{pat_id}"
    response = requests.get(url, headers=HEADERS)
    return response.json()

# حفظ تقرير جديد في جدول tbl_report
def save_report(pat_id, diagnosis, recommendation):
    url = f"{SUPABASE_URL}/rest/v1/tbl_report"
    data = {
        "pat_id": pat_id,
        "rep_date": datetime.now(timezone.utc).isoformat(),
        "rep_diagnosis": diagnosis,
        "rep_recommendation": recommendation
    }
    requests.post(url, headers=HEADERS, data=json.dumps(data))

# قراءة التقارير النهائية من جدول tbl_report
def fetch_reports(pat_id):
    url = f"{SUPABASE_URL}/rest/v1/tbl_report?pat_id=eq.{pat_id}"
    response = requests.get(url, headers=HEADERS)
    return response.json()

# 🟢 مسار POST: تحليل بيانات المريض وحفظ تقرير جديد
@app.post("/process_patient/{pat_id}")
def process_patient(pat_id: int):
    readings = fetch_readings(pat_id)
    results = []
    for r in readings:
        diagnosis = ecg_model.predict_ecg(
            r["pulse_rate"], r["oxygen_saturation"], r["temperature"]
        )
        recommendation = "متابعة الطبيب" if diagnosis == "غير طبيعي" else "استمرار المراقبة"
        save_report(pat_id, diagnosis, recommendation)
        results.append({"diagnosis": diagnosis, "recommendation": recommendation})
    return {"reports": results}

# 🟢 مسار GET: قراءة القراءات الخام
@app.get("/patients/{pat_id}")
def get_patient_readings(pat_id: int):
    readings = fetch_readings(pat_id)
    return {"readings": readings}

# 🟢 مسار GET: قراءة التقارير النهائية
@app.get("/reports/{pat_id}")
def get_patient_reports(pat_id: int):
    reports = fetch_reports(pat_id)
    return {"reports": reports}
