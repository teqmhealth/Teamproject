from fastapi import FastAPI
import requests

SUPABASE_URL = "https://kzqcznveyxallyonedls.supabase.co"
SUPABASE_KEY = "sb_publishable_rUhjaGNhHHlkwHis22Fqkg_mG2Fswbz"

app = FastAPI()

# دالة عامة للاتصال بـ Supabase
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
    elif method == "PATCH":
        return requests.patch(url, headers=headers, json=payload).json()

# قراءة بيانات مريض من جدول القراءات
def get_reading(pat_id: int):
    return supabase_request("GET", f"tbl_reading?pat_id=eq.{pat_id}&select=*")

# حفظ تقرير جديد في جدول التقارير
def save_report(pat_id: int, result: str):
    payload = {
        "pat_id": pat_id,
        "report_result": result
    }
    return supabase_request("POST", "tbl_report", payload)

# Endpoint: تحليل قراءة وحفظ تقرير
@app.post("/analyze/{pat_id}")
def analyze_patient(pat_id: int):
    readings = get_reading(pat_id)
    if not readings:
        return {"error": "لا توجد قراءات لهذا المريض"}

    reading = readings[0]
    oxygen = reading.get("oxygen_saturation")
    pulse = reading.get("pulse_rate")
    temp = reading.get("temperature")

    # منطق مبسط للتجربة
    if oxygen is not None and oxygen < 90:
        result = "خطر: انخفاض الأكسجين"
    elif pulse is not None and pulse > 120:
        result = "خطر: معدل نبض مرتفع"
    elif temp is not None and temp > 38:
        result = "خطر: حرارة مرتفعة"
    else:
        result = "طبيعي"

    # حفظ التقرير
    save_report(pat_id, result)

    return {"pat_id": pat_id, "result": result}

# Endpoint: استعراض كل التقارير
@app.get("/reports")
def get_reports():
    reports = supabase_request("GET", "tbl_report?select=*")
    return {"reports": reports}
