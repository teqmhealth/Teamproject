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
    return r.json()

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
    data = {
        "message": f"Add model {filename}",
        "content": b64_content
    }
    response = requests.put(url, headers=headers, json=data)
    return response.json()

# 🟢 جلب البيانات من Supabase
def fetch_readings():
    response = supabase.table("tbl_reading").select("*").execute()
    return response.data

# 🟢 تدريب وحفظ نموذج
def train_and_save(feature, filename):
    readings = fetch_readings()
    df = pd.DataFrame(readings)

    if df.empty or "is_emergency" not in df.columns:
        return {"error": "البيانات غير صالحة للتدريب"}

    X = df[[feature]]
    y = df["is_emergency"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    # حفظ محليًا مؤقتًا
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    joblib.dump(model, save_path)

    # رفع إلى GitHub
    github_result = upload_to_github(save_path, filename)

    return {
        "message": f"تم تدريب النموذج {filename} ورفعه إلى GitHub",
        "accuracy": acc,
        "github_response": github_result
    }

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

    return {
        "latest_reading": latest_reading,
        "prediction_results": results
    }
