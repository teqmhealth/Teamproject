from fastapi import FastAPI
import os, requests
from dotenv import load_dotenv

# تحميل القيم من .env
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

app = FastAPI()

# إعداد الهيدر الأساسي
headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation"
}

# 🟢 قراءة المرضى
@app.get("/patients")
def get_patients():
    url = f"{SUPABASE_URL}/rest/v1/tbl_patient?select=*"
    r = requests.get(url, headers=headers)
    return r.json()

# 🟢 إضافة مريض جديد
@app.post("/patients")
def add_patient(patient: dict):
    url = f"{SUPABASE_URL}/rest/v1/tbl_patient"
    r = requests.post(url, headers=headers, json=patient)
    return r.json()

# 🟢 قراءة المستخدمين
@app.get("/users")
def get_users():
    url = f"{SUPABASE_URL}/rest/v1/tbl_user?select=*"
    r = requests.get(url, headers=headers)
    return r.json()

# 🟢 إضافة مستخدم جديد
@app.post("/users")
def add_user(user: dict):
    url = f"{SUPABASE_URL}/rest/v1/tbl_user"
    r = requests.post(url, headers=headers, json=user)
    return r.json()

# 🟢 قراءة القراءات
@app.get("/readings")
def get_readings():
    url = f"{SUPABASE_URL}/rest/v1/tbl_reading?select=*"
    r = requests.get(url, headers=headers)
    return r.json()

# 🟢 إضافة قراءة جديدة
@app.post("/readings")
def add_reading(reading: dict):
    url = f"{SUPABASE_URL}/rest/v1/tbl_reading"
    r = requests.post(url, headers=headers, json=reading)
    return r.json()

# 🟢 قراءة التقارير
@app.get("/reports")
def get_reports():
    url = f"{SUPABASE_URL}/rest/v1/tbl_report?select=*"
    r = requests.get(url, headers=headers)
    return r.json()

# 🟢 قراءة التنبيهات
@app.get("/alerts")
def get_alerts():
    url = f"{SUPABASE_URL}/rest/v1/tbl_alert?select=*"
    r = requests.get(url, headers=headers)
    return r.json()def fetch_reports(pat_id):
    url = f"{SUPABASE_URL}/rest/v1/tbl_report?pat_id=eq.{pat_id}&select=*"
    response = requests.get(url, headers=HEADERS)
    return response.json()

@app.post("/process_patient/{pat_id}")
def process_patient(pat_id: int):
    readings = fetch_readings(pat_id)
    results = []
    for r in readings:
        diagnosis = ecg_model.predict_ecg(r["pulse_rate"], r["oxygen_saturation"])
        recommendation = "متابعة الطبيب" if diagnosis != "بخير" else "استمر على نفس النمط"
        save_report(pat_id, diagnosis, recommendation)
        results.append({"diagnosis": diagnosis, "recommendation": recommendation})
    return {"reports": results}

@app.get("/patients/{pat_id}")
def get_patient_readings(pat_id: int):
    return {"readings": fetch_readings(pat_id)}

@app.get("/reports/{pat_id}")
def get_patient_reports(pat_id: int):
    return {"reports": fetch_reports(pat_id)}    X, y = [], []
    for r in readings:
        row = []
        for f in features:
            if r.get(f) is None:
                break
            row.append(r.get(f))
        else:
            if r.get(label_field) is None:
                continue
            y.append(r[label_field])
            X.append(row)

    if not X:
        return {"error": "لا توجد بيانات مكتملة للتدريب"}

    X, y = np.array(X), np.array(y)
    y_cat = to_categorical(y, num_classes=num_classes)
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.3)

    model = Sequential([
        Dense(12, activation="relu", input_shape=(len(features),)),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=25, batch_size=4, verbose=0)
    model.save(filename)

    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return {"message": f"تم تدريب النموذج وحفظه في {filename}", "accuracy": float(accuracy)}

# -------------------------------
# دالة التنبؤ وإرجاع تقرير + تنبيه عند الطوارئ
# -------------------------------
def predict_and_report(model_name, features, pat_id, read_id):
    filename = f"{model_name}_model.keras"
    if not os.path.exists(filename):
        return {"error": f"النموذج {filename} غير موجود، درّبه أولاً"}

    model = load_model(filename)
    probs = model.predict(np.array([features]))
    pred_class = int(np.argmax(probs))

    # تحويل النتيجة إلى تشخيص نصي وتوصية
    if model_name == "oxygen" and pred_class == 1:
        diagnosis = "انخفاض الأكسجين"
        recommendation = "يجب مراجعة الطبيب فورًا"
        severity = "high"
    elif model_name == "temperature" and pred_class == 1:
        diagnosis = "ارتفاع الحرارة"
        recommendation = "اشرب سوائل وراجع الطبيب"
        severity = "medium"
    elif pred_class == 0:
        diagnosis = "بخير"
        recommendation = "استمر على نفس النمط"
        severity = "low"
    else:
        diagnosis = "غير طبيعي"
        recommendation = "لا تهمل نفسك"
        severity = "medium"

    # تقرير JSON
    report = {
        "rep_id": next(rep_counter),
        "pat_id": pat_id,
        "rep_date": datetime.utcnow().isoformat(),
        "rep_diagnosis": diagnosis,
        "rep_recommendation": recommendation
    }

    # إذا الحالة طارئة فقط → إنشاء تنبيه
    if pred_class == 1:
        alert = {
            "pat_id": pat_id,
            "alert_type": model_name,
            "alert_message": f"حالة طارئة من نموذج {model_name} للقراءة {read_id}",
            "severity": severity,
            "is_seen": False
        }
        return {"report": report, "alert": alert}
    else:
        return {"report": report}

# -------------------------------
# مسار التدريب الجماعي
# -------------------------------
@app.get("/train/all/{pat_id}")
def train_all_models(pat_id: int):
    results = {}
    results["gps"] = train_model_generic("tbl_reading", ["location"], "is_emergency", "gps_model.keras", 2)
    results["heart_attack"] = train_model_generic("tbl_reading", ["pulse_rate"], "is_emergency", "heart_attack_model.keras", 2)
    results["arrhythmia"] = train_model_generic("tbl_reading", ["pulse_rate"], "is_emergency", "arrhythmia_model.keras", 2)
    results["ecg"] = train_model_generic("tbl_ecg", ["signal_value"], "diagnosis_label", "ecg_model.keras", 2)
    results["oxygen"] = train_model_generic("tbl_reading", ["oxygen_saturation"], "is_emergency", "oxygen_model.keras", 2)
    results["fall"] = train_model_generic("tbl_reading", ["location"], "is_emergency", "fall_model.keras", 2)
    results["temperature"] = train_model_generic("tbl_reading", ["temperature"], "is_emergency", "temperature_model.keras", 2)
    return {"training_results": results}

# -------------------------------
# مسار الفحص الجماعي
# -------------------------------
@app.get("/predict/all/{read_id}")
def predict_all(read_id: int):
    reports = []
    reading = supabase_request(f"tbl_reading?read_id=eq.{read_id}&select=*")
    ecg = supabase_request(f"tbl_ecg?read_id=eq.{read_id}&select=*")

    if not reading and not ecg:
        return {"error": f"لا توجد قراءة بالرقم {read_id}"}

    if reading:
        r = reading[0]
        pat_id = r["pat_id"]

        if r.get("location"):
            reports.append(predict_and_report("gps", [r["location"]], pat_id, read_id))
            reports.append(predict_and_report("fall", [r["location"]], pat_id, read_id))

        if r.get("pulse_rate"):
            reports.append(predict_and_report("heart_attack", [r["pulse_rate"]], pat_id, read_id))
            reports.append(predict_and_report("arrhythmia", [r["pulse_rate"]], pat_id, read_id))

        if r.get("oxygen_saturation"):
            reports.append(predict_and_report("oxygen", [r["oxygen_saturation"]], pat_id, read_id))

        if r.get("temperature"):
            reports.append(predict_and_report("temperature", [r["temperature"]], pat_id, read_id))

    if ecg:
        e = ecg[0]
        if e.get("signal_value"):
            reports.append(predict_and_report("ecg", [e["signal_value"]], e["pat_id"], read_id))

    return reports
