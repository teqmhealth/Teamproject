from fastapi import FastAPI
import requests
from datetime import datetime

SUPABASE_URL = "https://kzqcznveyxallyonedls.supabase.co"
SUPABASE_KEY = "sb_publishable_rUhjaGNhHHlkwHis22Fqkg_mG2Fswbz"

app = FastAPI()

def supabase_request(method, endpoint, payload=None):
    url = f"{SUPABASE_URL}/rest/v1/{endpoint}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    if method == "GET":
        return requests.get(url, headers=headers).json()
    elif method == "POST":
        return requests.post(url, headers=headers, json=payload).json()

def get_reading_by_id(read_id: int):
    return supabase_request("GET", f"tbl_reading?read_id=eq.{read_id}&select=*")

def save_report(pat_id: int, diagnosis: str, recommendation: str):
    payload = {
        "pat_id": pat_id,
        "rep_date": datetime.utcnow().isoformat() + "Z",  # تاريخ اليوم
        "rep_diagnosis": diagnosis,
        "rep_recommendation": recommendation
    }
    return supabase_request("POST", "tbl_report", payload)

@app.post("/analyze_reading/{read_id}")
def analyze_reading(read_id: int):
    readings = get_reading_by_id(read_id)
    if not readings:
        return {"error": "لا توجد قراءة بهذا الرقم"}

    reading = readings[0]
    pat_id = reading.get("pat_id")
    oxygen = reading.get("oxygen_saturation")
    pulse = reading.get("pulse_rate")
    temp = reading.get("temperature")

    if oxygen is not None and oxygen < 90:
        diagnosis = "انخفاض الأكسجين"
        recommendation = "يجب مراجعة الطبيب فورًا"
    elif pulse is not None and pulse > 120:
        diagnosis = "معدل نبض مرتفع"
        recommendation = "ينصح بالراحة ومراجعة الطبيب"
    elif temp is not None and temp > 38:
        diagnosis = "حرارة مرتفعة"
        recommendation = "تناول سوائل وخافض حرارة"
    else:
        diagnosis = "بخير"
        recommendation = "لا تهمل نفسك"

    report = save_report(pat_id, diagnosis, recommendation)

    return {
        "read_id": read_id,
        "pat_id": pat_id,
        "rep_date": report[0].get("rep_date") if report else None,
        "rep_diagnosis": diagnosis,
        "rep_recommendation": recommendation
    }

@app.get("/reports")
def get_reports():
    reports = supabase_request("GET", "tbl_report?select=*")
    return {"reports": reports}    if not readings:
        return {"error": "لا توجد قراءة بهذا الرقم"}

    reading = readings[0]
    pat_id = reading.get("pat_id")
    oxygen = reading.get("oxygen_saturation")
    pulse = reading.get("pulse_rate")
    temp = reading.get("temperature")

    if oxygen is not None and oxygen < 90:
        diagnosis = "انخفاض الأكسجين"
        recommendation = "يجب مراجعة الطبيب فورًا"
    elif pulse is not None and pulse > 120:
        diagnosis = "معدل نبض مرتفع"
        recommendation = "ينصح بالراحة ومراجعة الطبيب"
    elif temp is not None and temp > 38:
        diagnosis = "حرارة مرتفعة"
        recommendation = "تناول سوائل وخافض حرارة"
    else:
        diagnosis = "بخير"
        recommendation = "لا تهمل نفسك"

   report = save_report(pat_id, diagnosis, recommendation)

    return {
        "read_id": read_id,
        "pat_id": pat_id,
        "rep_diagnosis": diagnosis,
        "rep_recommendation": recommendation
    }

@app.get("/reports")
def get_reports():
    reports = supabase_request("GET", "tbl_report?select=*")
    return {"reports": reports}
