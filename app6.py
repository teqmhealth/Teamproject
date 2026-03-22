from fastapi import FastAPI
import requests
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import os
from dotenv import load_dotenv

# تحميل القيم من ملف .env
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

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

# -------------------------------
# دوال عرض جميع البيانات
# -------------------------------
@app.get("/patients")
def get_patients():
    return {"patients": supabase_request("GET", "tbl_patient?select=*")}

@app.get("/users")
def get_users():
    return {"users": supabase_request("GET", "tbl_user?select=*")}

@app.get("/reports")
def get_reports():
    return {"reports": supabase_request("GET", "tbl_report?select=*")}

@app.get("/readings")
def get_readings():
    return {"readings": supabase_request("GET", "tbl_reading?select=*")}

# -------------------------------
# عرض القراءات حسب رقم المريض
# -------------------------------
@app.get("/readings/{pat_id}")
def get_readings_by_patient(pat_id: int):
    readings = supabase_request("GET", f"tbl_reading?pat_id=eq.{pat_id}&select=*")
    if not readings:
        return {"error": f"لا توجد قراءات للمريض رقم {pat_id}"}
    return {"pat_id": pat_id, "readings": readings}

# -------------------------------
# عرض التقارير حسب رقم المريض
# -------------------------------
@app.get("/reports/{pat_id}")
def get_reports_by_patient(pat_id: int):
    reports = supabase_request("GET", f"tbl_report?pat_id=eq.{pat_id}&select=*")
    if not reports:
        return {"error": f"لا توجد تقارير للمريض رقم {pat_id}"}
    return {"pat_id": pat_id, "reports": reports}

# -------------------------------
# تدريب النموذج على جميع القراءات
# -------------------------------
@app.get("/train_model")
def train_model():
    readings = supabase_request("GET", "tbl_reading?select=*")
    return train_on_readings(readings, "عام")

# -------------------------------
# تدريب النموذج على بيانات مريض محدد
# -------------------------------
@app.get("/train_model/{pat_id}")
def train_model_by_patient(pat_id: int):
    readings = supabase_request("GET", f"tbl_reading?pat_id=eq.{pat_id}&select=*")
    return train_on_readings(readings, f"المريض رقم {pat_id}", pat_id)

# -------------------------------
# دالة مساعدة للتدريب
# -------------------------------
def train_on_readings(readings, label, pat_id=None):
    if not readings:
        return {"error": f"لا توجد بيانات للتدريب ({label})"}

    X, y = [], []
    for r in readings:
        oxygen = r.get("oxygen_saturation")
        pulse = r.get("pulse_rate")
        temp = r.get("temperature")
        if oxygen is None or pulse is None or temp is None:
            continue

        score = 0
        if oxygen < 90: score += 2
        elif oxygen < 95: score += 1
        if pulse > 120 or pulse < 50: score += 2
        elif pulse > 100: score += 1
        if temp > 38 or temp < 35: score += 2
        elif temp > 37.5: score += 1

        if score >= 4:
            label_class = "خطر عالي"
        elif score >= 2:
            label_class = "خطر متوسط"
        else:
            label_class = "طبيعي"

        X.append([oxygen, pulse, temp])
        y.append(label_class)

    if not X:
        return {"error": "لا توجد قراءات مكتملة للتدريب"}

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    if not hasattr(app.state, "models"):
        app.state.models = {}
    if pat_id:
        app.state.models[pat_id] = clf
    else:
        app.state.model = clf

    accuracy = clf.score(X_test, y_test)

    return {
        "message": f"تم تدريب النموذج بنجاح ({label})",
        "accuracy": accuracy,
        "samples": len(readings)
    }

# -------------------------------
# استخدام النموذج لإعطاء النتيجة فقط (بدون تحديث جدول التقارير)
# -------------------------------
@app.post("/predict/{read_id}")
def predict(read_id: int):
    readings = supabase_request("GET", f"tbl_reading?read_id=eq.{read_id}&select=*")
    if not readings:
        return {"error": "لا توجد قراءة بهذا الرقم"}

    reading = readings[0]
    pat_id = reading.get("pat_id")
    oxygen = reading.get("oxygen_saturation")
    pulse = reading.get("pulse_rate")
    temp = reading.get("temperature")

    if oxygen is None or pulse is None or temp is None:
        return {"error": "القراءة غير مكتملة"}

    # اختيار النموذج الخاص بالمريض إذا موجود، وإلا النموذج العام
    clf = None
    if hasattr(app.state, "models") and pat_id in app.state.models:
        clf = app.state.models[pat_id]
    elif hasattr(app.state, "model"):
        clf = app.state.model
    else:
        return {"error": "لا يوجد نموذج مدرب، نفذ /train_model أو /train_model/{pat_id}"}

    prediction = clf.predict([[oxygen, pulse, temp]])[0]

    if prediction == "خطر عالي":
        recommendation = "يجب مراجعة الطبيب فورًا"
    elif prediction == "خطر متوسط":
        recommendation = "ينصح بالراحة والمتابعة"
    else:
        recommendation = "استمر بالمراقبة"

    current_date = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    return {
        "read_id": read_id,
        "pat_id": pat_id,
        "rep_date": current_date,
        "rep_diagnosis": prediction,
        "rep_recommendation": recommendation
    }
