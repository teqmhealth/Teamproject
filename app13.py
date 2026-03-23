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

# إعداد الهيدر الأساسي
headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation"
}

# 🟢 المرضى
@app.get("/patients")
def get_patients():
    url = f"{SUPABASE_URL}/rest/v1/tbl_patient?select=*"
    return requests.get(url, headers=headers).json()

@app.post("/patients")
def add_patient(patient: dict):
    url = f"{SUPABASE_URL}/rest/v1/tbl_patient"
    return requests.post(url, headers=headers, json=patient).json()

# 🟢 المستخدمين
@app.get("/users")
def get_users():
    url = f"{SUPABASE_URL}/rest/v1/tbl_user?select=*"
    return requests.get(url, headers=headers).json()

@app.post("/users")
def add_user(user: dict):
    url = f"{SUPABASE_URL}/rest/v1/tbl_user"
    return requests.post(url, headers=headers, json=user).json()

# 🟢 القراءات
@app.get("/readings")
def get_readings():
    url = f"{SUPABASE_URL}/rest/v1/tbl_reading?select=*"
    return requests.get(url, headers=headers).json()

@app.post("/readings")
def add_reading(reading: dict):
    url = f"{SUPABASE_URL}/rest/v1/tbl_reading"
    return requests.post(url, headers=headers, json=reading).json()

# 🟢 التقارير
@app.get("/reports")
def get_reports():
    url = f"{SUPABASE_URL}/rest/v1/tbl_report?select=*"
    return requests.get(url, headers=headers).json()

@app.get("/reports/{pat_id}")
def fetch_reports(pat_id: int):
    url = f"{SUPABASE_URL}/rest/v1/tbl_report?pat_id=eq.{pat_id}"
    return requests.get(url, headers=headers).json()

# 🟢 التنبيهات
@app.get("/alerts")
def get_alerts():
    url = f"{SUPABASE_URL}/rest/v1/tbl_alert?select=*"
    return requests.get(url, headers=headers).json()

# 🟢 رفع النموذج إلى GitHub (مع دعم التحديث)
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

# 🟢 جلب البيانات من Supabase
def fetch_readings_data():
    response = supabase.table("tbl_reading").select("*").execute()
    return response.data

# 🟢 تدريب وحفظ نموذج
def train_and_save(feature, filename):
    readings = fetch_readings_data()
    df = pd.DataFrame(readings)

    if df.empty:
        return {"error": "لا توجد بيانات للتدريب"}

    # توليد العمود الهدف داخليًا
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

# 🟢 مسارات التدريب
@app.get("/train/temperature")
def train_temperature():
    return train_and_save("temperature", "temperature_model.pkl")

@app.get("/train/oxygen")
def train_oxygen():
    return train_and_save("oxygen_saturation", "oxygen_model.pkl")

@app.get("/train/pulse")
def train_pulse():
    return train_and_save("pulse_rate", "pulse_model.pkl")

@app.get("/train/all")
def train_all():
    return {
        "temperature": train_and_save("temperature", "temperature_model.pkl"),
        "oxygen": train_and_save("oxygen_saturation", "oxygen_model.pkl"),
        "pulse": train_and_save("pulse_rate", "pulse_model.pkl")
    }

# 🟢 التنبؤ من آخر قراءة
@app.get("/predict")
def predict():
    latest = supabase.table("tbl_reading").select("*").order("created_at", desc=True).limit(1).execute()
    if not latest.data:
        return {"error": "لا توجد قراءات"}

    latest_reading = latest.data[0]
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
            results[feature] = int(model.predict([[latest_reading[feature]]])[0])

    emergency_flag = int((latest_reading["temperature"] > 38) or (latest_reading["oxygen_saturation"] < 90))

    return {"latest_reading": latest_reading, "prediction_results": results, "emergency_flag": emergency_flag}

# 🟢 التنبؤ بحسب رقم القراءة
@app.get("/predict/{read_id}")
def predict_by_id(read_id: int):
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

    return {"read_id": read_id, "reading": selected_reading, "prediction_results": results, "emergency_flag": emergency_flag}
