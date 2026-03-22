from fastapi import FastAPI
import requests
import numpy as np
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from dotenv import load_dotenv

# تحميل القيم من ملف .env
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

app = FastAPI()

def supabase_request(endpoint):
    url = f"{SUPABASE_URL}/rest/v1/{endpoint}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    return requests.get(url, headers=headers).json()

# -------------------------------
# دوال عرض البيانات العامة
# -------------------------------
@app.get("/patients")
def get_patients():
    return {"patients": supabase_request("tbl_patient?select=*")}

@app.get("/users")
def get_users():
    return {"users": supabase_request("tbl_user?select=*")}

@app.get("/reports")
def get_reports():
    return {"reports": supabase_request("tbl_report?select=*")}

@app.get("/readings")
def get_readings():
    return {"readings": supabase_request("tbl_reading?select=*")}

# -------------------------------
# دالة عامة لتدريب أي نموذج وحفظه H5
# -------------------------------
def train_model_generic(table_name, features, label_field, filename, num_classes=2):
    readings = supabase_request(f"{table_name}?select=*")
    if not readings:
        return {"error": f"لا توجد بيانات في {table_name}"}

    X, y = [], []
    for r in readings:
        row = []
        for f in features:
            if r.get(f) is None:
                break
            row.append(r.get(f))
        else:
            y.append(r.get(label_field))
            X.append(row)

    if not X:
        return {"error": "لا توجد بيانات مكتملة للتدريب"}

    X = np.array(X)
    y = np.array(y)
    y_cat = to_categorical(y, num_classes=num_classes)

    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.3, random_state=42)

    model = Sequential([
        Dense(16, activation='relu', input_shape=(len(features),)),
        Dense(8, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    model.save(filename)

    return {"message": f"تم تدريب النموذج وحفظه في {filename}", "accuracy": float(accuracy), "samples": len(readings)}

# -------------------------------
# تدريبات النماذج المختلفة
# -------------------------------
@app.get("/train/ecg")
def train_ecg():
    return train_model_generic("tbl_ecg", ["signal_value"], "diagnosis_label", "ecg_model.h5", num_classes=2)

@app.get("/train/oxygen")
def train_oxygen():
    return train_model_generic("tbl_oxygen", ["oxygen_saturation"], "diagnosis_label", "oxygen_model.h5", num_classes=3)

@app.get("/train/temperature")
def train_temperature():
    return train_model_generic("tbl_temperature", ["temperature"], "diagnosis_label", "temperature_model.h5", num_classes=3)

@app.get("/train/fall")
def train_fall():
    return train_model_generic("tbl_fall", ["acceleration_x", "acceleration_y", "acceleration_z"], "diagnosis_label", "fall_model.h5", num_classes=2)

@app.get("/train/heart_attack")
def train_heart_attack():
    return train_model_generic("tbl_heart_attack", ["cholesterol", "blood_pressure"], "diagnosis_label", "heart_attack_model.h5", num_classes=2)

@app.get("/train/arrhythmia")
def train_arrhythmia():
    return train_model_generic("tbl_arrhythmia", ["ecg_signal"], "diagnosis_label", "arrhythmia_model.h5", num_classes=2)

@app.get("/train/gps")
def train_gps():
    return train_model_generic("tbl_gps", ["latitude", "longitude"], "diagnosis_label", "gps_model.h5", num_classes=2)

@app.get("/train/maigghn")
def train_maigghn():
    return train_model_generic("tbl_maigghn", ["feature1", "feature2"], "diagnosis_label", "maigghn_model.h5", num_classes=2)

# -------------------------------
# التنبؤ باستخدام نموذج محفوظ
# -------------------------------
@app.post("/predict/{model_name}/{read_id}")
def predict(model_name: str, read_id: int):
    filename = f"{model_name}_model.h5"
    if not os.path.exists(filename):
        return {"error": f"النموذج {filename} غير موجود، درّبه أولاً"}

    model = load_model(filename)

    # مثال: قراءة من جدول عام tbl_reading
    readings = supabase_request(f"tbl_reading?read_id=eq.{read_id}&select=*")
    if not readings:
        return {"error": "لا توجد قراءة بهذا الرقم"}

    reading = readings[0]
    oxygen = reading.get("oxygen_saturation")
    pulse = reading.get("pulse_rate")
    temp = reading.get("temperature")

    if oxygen is None or pulse is None or temp is None:
        return {"error": "القراءة غير مكتملة"}

    prediction_probs = model.predict(np.array([[oxygen, pulse, temp]]))
    prediction_class = np.argmax(prediction_probs)

    if prediction_class == 2:
        diagnosis = "خطر عالي"
        recommendation = "يجب مراجعة الطبيب فورًا"
    elif prediction_class == 1:
        diagnosis = "خطر متوسط"
        recommendation = "ينصح بالراحة والمتابعة"
    else:
        diagnosis = "طبيعي"
        recommendation = "استمر بالمراقبة"

    current_date = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    return {
        "model": model_name,
        "read_id": read_id,
        "rep_date": current_date,
        "rep_diagnosis": diagnosis,
        "rep_recommendation": recommendation
  }
