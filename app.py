from fastapi import FastAPI
import requests

SUPABASE_URL = "https://kzqcznveyxallyonedls.supabase.co"
SUPABASE_KEY = "sb_publishable_rUhjaGNhHHlkwHis22Fqkg_mG2Fswbz"

app = FastAPI()

# دالة عامة لجلب بيانات مريض
def get_patient_by_id(pat_id: int):
    url = f"{SUPABASE_URL}/rest/v1/tbl_patient?pat_id=eq.{pat_id}&select=*"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }
    response = requests.get(url, headers=headers)
    return response.json()

# دالة عامة لتحديث حالة المريض
def update_patient_status(pat_id: int, status: str):
    url = f"{SUPABASE_URL}/rest/v1/tbl_patient?pat_id=eq.{pat_id}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"pat_status": status}
    requests.patch(url, headers=headers, json=payload)

# منطق مبسط للتنبؤ
def simple_logic(patient, mode: str):
    age = patient[0].get("pat_age")
    emergency = patient[0].get("is_emergency")

    if age is None:
        return "بيانات غير مكتملة"

    if mode == "ecg":
        return "خطر" if age > 30 else "طبيعي"
    elif mode == "heart_attack":
        return "خطر" if emergency else "طبيعي"
    elif mode == "oxygen":
        return "خطر" if age > 40 else "طبيعي"
    elif mode == "arrhythmia":
        return "خطر" if emergency else "طبيعي"
    elif mode == "fall":
        return "خطر" if age > 65 else "طبيعي"
    else:
        return "غير معروف"

# ECG
@app.post("/predict/ecg/{pat_id}")
@app.get("/predict/ecg/{pat_id}")
def predict_ecg(pat_id: int):
    patient = get_patient_by_id(pat_id)
    if not patient:
        return {"error": "المريض غير موجود"}
    status = simple_logic(patient, "ecg")
    update_patient_status(pat_id, status)
    return {"pat_id": pat_id, "status": status}

# Heart Attack
@app.post("/predict/heart_attack/{pat_id}")
@app.get("/predict/heart_attack/{pat_id}")
def predict_heart_attack(pat_id: int):
    patient = get_patient_by_id(pat_id)
    if not patient:
        return {"error": "المريض غير موجود"}
    status = simple_logic(patient, "heart_attack")
    update_patient_status(pat_id, status)
    return {"pat_id": pat_id, "status": status}

# Oxygen
@app.post("/predict/oxygen/{pat_id}")
@app.get("/predict/oxygen/{pat_id}")
def predict_oxygen(pat_id: int):
    patient = get_patient_by_id(pat_id)
    if not patient:
        return {"error": "المريض غير موجود"}
    status = simple_logic(patient, "oxygen")
    update_patient_status(pat_id, status)
    return {"pat_id": pat_id, "status": status}

# Arrhythmia
@app.post("/predict/arrhythmia/{pat_id}")
@app.get("/predict/arrhythmia/{pat_id}")
def predict_arrhythmia(pat_id: int):
    patient = get_patient_by_id(pat_id)
    if not patient:
        return {"error": "المريض غير موجود"}
    status = simple_logic(patient, "arrhythmia")
    update_patient_status(pat_id, status)
    return {"pat_id": pat_id, "status": status}

# Fall
@app.post("/predict/fall/{pat_id}")
@app.get("/predict/fall/{pat_id}")
def predict_fall(pat_id: int):
    patient = get_patient_by_id(pat_id)
    if not patient:
        return {"error": "المريض غير موجود"}
    status = simple_logic(patient, "fall")
    update_patient_status(pat_id, status)
    return {"pat_id": pat_id, "status": status}    if age is None:
        return {"error": "لا توجد بيانات عمر"}

    # منطق مبسط للتجربة: إذا العمر أكبر من 30 → خطر
    status = "خطر" if age > 30 else "طبيعي"

    # تحديث Supabase
    update_patient_status(pat_id, status)

    return {"pat_id": pat_id, "status": status}

@app.get("/predict/ecg/{pat_id}")
def predict_ecg_get(pat_id: int):
    patient = get_patient_by_id(pat_id)
    if not patient:
        return {"error": "المريض غير موجود"}

    age = patient[0].get("pat_age")
    if age is None:
        return {"error": "لا توجد بيانات عمر"}

    status = "خطر" if age > 30 else "طبيعي"
    update_patient_status(pat_id, status)

    return {"pat_id": pat_id, "status": status}
