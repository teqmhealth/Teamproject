from fastapi import FastAPI
import os, pandas as pd, joblib, base64, requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from supabase import create_client, Client
from dotenv import load_dotenv

app = FastAPI()

# تحميل القيم من .env
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 🟢 رفع النموذج إلى GitHub
def upload_to_github(filepath, filename):
    with open(filepath, "rb") as f:
        content = f.read()
    b64_content = base64.b64encode(content).decode("utf-8")

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/models/{filename}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    check = requests.get(url, headers=headers)
    if check.status_code == 200:
        sha = check.json()["sha"]
        data = {"message": f"Update model {filename}", "content": b64_content, "sha": sha}
    else:
        data = {"message": f"Add model {filename}", "content": b64_content}

    return requests.put(url, headers=headers, json=data).json()

# 🟢 جلب البيانات
def fetch_readings_data():
    response = supabase.table("tbl_reading").select("*").execute()
    return response.data

# 🟢 تدريب وحفظ نموذج
def train_and_save(feature, filename):
    readings = fetch_readings_data()
    df = pd.DataFrame(readings)

    if df.empty:
        return {"error": "لا توجد بيانات للتدريب"}

    df["is_emergency"] = ((df["temperature"] > 38) | (df["oxygen_saturation"] < 90)).astype(int)

    X = df[[feature]]
    y = df["is_emergency"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    joblib.dump(model, save_path)

    github_result = upload_to_github(save_path, filename)

    return {"message": f"تم تدريب النموذج {filename} ورفعه إلى GitHub", "accuracy": acc, "github_response": github_result}

# 🟢 توليد تقرير ديناميكي
def generate_report(rep_id, pat_id, diagnosis, recommendation):
    return {
        "rep_id": rep_id,
        "pat_id": pat_id,
        "rep_date": pd.Timestamp.now(tz="UTC").isoformat(),
        "rep_diagnosis": diagnosis,
        "rep_recommendation": recommendation
    }

# 🟢 توليد تنبيه ديناميكي
def generate_alert(alert_id, pat_id, emergency_flag):
    if emergency_flag == 1:
        return {
            "alert_id": alert_id,
            "pat_id": pat_id,
            "alert_type": "Critical Condition",
            "alert_message": "الحالة حرجة، يجب مراجعة الطبيب فورًا وعدم التأخير.",
            "alert_timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "is_seen": False
        }
    else:
        return {
            "alert_id": alert_id,
            "pat_id": pat_id,
            "alert_type": "Healthy",
            "alert_message": "الوضع مستقر، لا ترهق نفسك ولا تهمل أدويتك.",
            "alert_timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "is_seen": True
        }

# 🟢 التنبؤ بحسب رقم القراءة
@app.get("/predict/reading/{read_id}")
def predict_by_reading(read_id: int):
    reading = supabase.table("tbl_reading").select("*").eq("read_id", read_id).execute()
    if not reading.data:
        return {"error": f"لا توجد قراءة بالرقم {read_id}"}

    selected_reading = reading.data[0]
    results = {}

    models_dir = "models"
    for fname, feature in {
        "temperature_model.pkl": "temperature",
        "oxygen_model.pkl": "oxygen_saturation",
        "pulse_model.pkl": "pulse_rate"
    }.items():
        path = os.path.join(models_dir, fname)
        if os.path.exists(path):
            model = joblib.load(path)
            results[feature] = int(model.predict([[selected_reading[feature]]])[0])

    emergency_flag = int((selected_reading["temperature"] > 38) or (selected_reading["oxygen_saturation"] < 90))

    diagnosis = "خطر عالي" if emergency_flag == 1 else "الوضع مستقر"
    recommendation = "يجب مراجعة الطبيب فورًا" if emergency_flag == 1 else "استمر في المتابعة الدورية فقط"

    report = generate_report(100 + read_id, selected_reading.get("pat_id", 0), diagnosis, recommendation)
    alert = generate_alert(200 + read_id, selected_reading.get("pat_id", 0), emergency_flag)

    return {
        "read_id": read_id,
        "reading": selected_reading,
        "prediction_results": results,
        "emergency_flag": emergency_flag,
        "report": report,
        "alert": alert
    }

# 🟢 التنبؤ بحسب المريض (آخر قراءة له)
@app.get("/predict/patient/{pat_id}")
def predict_by_patient(pat_id: int):
    reading = supabase.table("tbl_reading").select("*").eq("pat_id", pat_id).order("created_at", desc=True).limit(1).execute()
    if not reading.data:
        return {"error": f"لا توجد قراءات للمريض {pat_id}"}

    selected_reading = reading.data[0]
    results = {}

    models_dir = "models"
    for fname, feature in {
        "temperature_model.pkl": "temperature",
        "oxygen_model.pkl": "oxygen_saturation",
        "pulse_model.pkl": "pulse_rate"
    }.items():
        path = os.path.join(models_dir, fname)
        if os.path.exists(path):
            model = joblib.load(path)
            results[feature] = int(model.predict([[selected_reading[feature]]])[0])

    emergency_flag = int((selected_reading["temperature"] > 38) or (selected_reading["oxygen_saturation"] < 90))

    diagnosis = "خطر عالي" if emergency_flag == 1 else "الوضع مستقر"
    recommendation = "يجب مراجعة الطبيب فورًا" if emergency_flag == 1 else "استمر في المتابعة الدورية فقط"

    report = generate_report(1000 + pat_id, pat_id, diagnosis, recommendation)
    alert = generate_alert(2000 + pat_id, pat_id, emergency_flag)

    return {
        "pat_id": pat_id,
        "reading": selected_reading,
        "prediction_results": results,
        "emergency_flag": emergency_flag,
        "report": report,
        "alert": alert
    }
