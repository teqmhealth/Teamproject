from fastapi import FastAPI
import requests
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np

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

# -------------------------------
# دوال عرض جميع البيانات
# -------------------------------
@app.get("/patients")
def get_patients():
    patients = supabase_request("GET", "tbl_patient?select=*")
    return {"patients": patients}

@app.get("/users")
def get_users():
    users = supabase_request("GET", "tbl_user?select=*")
    return {"users": users}

@app.get("/reports")
def get_reports():
    reports = supabase_request("GET", "tbl_report?select=*")
    return {"reports": reports}

@app.get("/readings")
def get_readings():
    readings = supabase_request("GET", "tbl_reading?select=*")
    return {"readings": readings}

# -------------------------------
# تدريب النموذج على بيانات القراءات
# -------------------------------
@app.get("/train_model")
def train_model():
    readings = supabase_request("GET", "tbl_reading?select=*")
    if not readings:
        return {"error": "لا توجد بيانات للتدريب"}

    X, y = [], []

    for r in readings:
        oxygen = r.get("oxygen_saturation")
        pulse = r.get("pulse_rate")
        temp = r.get("temperature")

        if oxygen is None or pulse is None or temp is None:
            continue

        # منطق مبسط لتصنيف الحالة
        score = 0
        if oxygen < 90: score += 2
        elif oxygen < 95: score += 1
        if pulse > 120 or pulse < 50: score += 2
        elif pulse > 100: score += 1
        if temp > 38 or temp < 35: score += 2
        elif temp > 37.5: score += 1

        if score >= 4:
            label = "خطر عالي"
        elif score >= 2:
            label = "خطر متوسط"
        else:
            label = "طبيعي"

        X.append([oxygen, pulse, temp])
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    # تقسيم البيانات إلى تدريب واختبار
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # تدريب النموذج
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # حساب الدقة
    accuracy = clf.score(X_test, y_test)

    # نخزن النموذج في التطبيق (ذاكرة مؤقتة)
    app.state.model = clf

    return {
        "message": "تم تدريب النموذج بنجاح",
        "accuracy": accuracy,
        "samples": len(readings)
    }

# -------------------------------
# استخدام النموذج لإعطاء النتيجة لقراءة جديدة
# -------------------------------
@app.post("/predict/{read_id}")
def predict(read_id: int):
    # تأكد أن النموذج مدرب
    if not hasattr(app.state, "model"):
        return {"error": "النموذج غير مدرب بعد، نفذ /train_model أولاً"}

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

    # توقع باستخدام النموذج المدرب
    prediction = app.state.model.predict([[oxygen, pulse, temp]])[0]

    # توصية بناءً على النتيجة
    if prediction == "خطر عالي":
        recommendation = "يجب مراجعة الطبيب فورًا"
    elif prediction == "خطر متوسط":
        recommendation = "ينصح بالراحة والمتابعة"
    else:
        recommendation = "استمر بالمراقبة"

    # حفظ التقرير في جدول tbl_report
    payload = {
        "pat_id": pat_id,
        "rep_date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        "rep_diagnosis": prediction,
        "rep_recommendation": recommendation
    }
    report = supabase_request("POST", "tbl_report", payload)

    # Supabase يرجع قائمة من الكائنات بعد الإدخال
    rep_date = None
    if isinstance(report, list) and len(report) > 0:
        rep_date = report[0].get("rep_date")

    return {
        "read_id": read_id,
        "pat_id": pat_id,
        "rep_date": rep_date,
        "rep_diagnosis": prediction,
        "rep_recommendation": recommendation
    }
