from fastapi import FastAPI
import requests

SUPABASE_URL = "https://kzqcznveyxallyonedls.supabase.co"
SUPABASE_KEY = "sb_publishable_rUhjaGNhHHlkwHis22Fqkg_mG2Fswbz"

app = FastAPI()

# دالة تجيب كل المرضى
def get_all_patients():
    url = f"{SUPABASE_URL}/rest/v1/tbl_patient?select=*"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }
    response = requests.get(url, headers=headers)
    return response.json()
def get_all_reports():
    url = f"{SUPABASE_URL}/rest/v1/tbl_report?select=*"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }
    response = requests.get(url, headers=headers)
    return response.json()
def get_all_reading():
    url = f"{SUPABASE_URL}/rest/v1/tbl_reading?select=*"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }
    response = requests.get(url, headers=headers)
    return response.json()

# Endpoint يعيد كل المرضى
@app.get("/patients")
def read_patients():
    patients = get_all_patients()
    return {"patients": patients}
@app.get("/reports")
def read_reports():
    reports = get_all_reports()
    return {"reports": reports}
    @app.get("/reading")
def read_readers():
    reports = get_all_reading()
    return {"readers": reports}
