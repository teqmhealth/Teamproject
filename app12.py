import os
os.environ["KERAS_BACKEND"] = "torch"

import requests, numpy as np
from sklearn.model_selection import train_test_split
from keras_core import Sequential
from keras_core.layers import Dense
from keras_core.utils import to_categorical
from keras_core.models import load_model
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

app = FastAPI()

def supabase_request(endpoint: str, method="GET", data=None):
    url = f"{SUPABASE_URL}/rest/v1/{endpoint}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    if method == "GET":
        return requests.get(url, headers=headers).json()
    elif method == "POST":
        return requests.post(url, headers=headers, json=data).json()

# -------------------------------
# دالة عامة للتدريب
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
    return {"message": f"تم تدريب النموذج وحفظه في {filename}", "accuracy": float(accuracy), "samples": len(readings)}

# -------------------------------
# دالة عامة للتنبؤ + تقرير + تنبيه
# -------------------------------
def predict_and_log(model_name, features, pat_id, read_id):
    filename = f"{model_name}_model.keras"
    if not os.path.exists(filename):
        return {"error": f"النموذج {filename} غير موجود، درّبه أولاً"}

    model = load_model(filename)
    probs = model.predict(np.array([features]))
    pred_class = int(np.argmax(probs))

    # إنشاء تقرير
    supabase_request("tbl_report", method="POST", data={
        "pat_id": pat_id,
        "rep_diagnosis": f"{model_name} prediction={pred_class}",
        "rep_recommendation": "راجع الطبيب إذا الحالة طارئة"
    })

    # إذا الحالة طارئة → إنشاء تنبيه
    if pred_class == 1:
        supabase_request("tbl_alert", method="POST", data={
            "pat_id": pat_id,
            "alert_type": model_name,
            "alert_message": f"حالة طارئة من نموذج {model_name} للقراءة {read_id}",
            "is_seen": False
        })

    return {
        "model": model_name,
        "read_id": read_id,
        "features": features,
        "prediction_class": pred_class,
        "probabilities": probs.tolist()
    }

# -------------------------------
# مسارات التنبؤ باستخدام read_id
# -------------------------------
@app.get("/predict/gps/{read_id}")
def predict_gps(read_id: int):
    reading = supabase_request(f"tbl_reading?read_id=eq.{read_id}&select=*")
    if not reading: return {"error": "لا توجد قراءة"}
    r = reading[0]
    if r.get("location"):
        return predict_and_log("gps", [r["location"]], r["pat_id"], read_id)
    return {"error": "لا توجد بيانات location"}

@app.get("/predict/heart_attack/{read_id}")
def predict_heart_attack(read_id: int):
    reading = supabase_request(f"tbl_reading?read_id=eq.{read_id}&select=*")
    if not reading: return {"error": "لا توجد قراءة"}
    r = reading[0]
    if r.get("pulse_rate"):
        return predict_and_log("heart_attack", [r["pulse_rate"]], r["pat_id"], read_id)
    return {"error": "لا توجد بيانات pulse_rate"}

@app.get("/predict/arrhythmia/{read_id}")
def predict_arrhythmia(read_id: int):
    reading = supabase_request(f"tbl_reading?read_id=eq.{read_id}&select=*")
    if not reading: return {"error": "لا توجد قراءة"}
    r = reading[0]
    if r.get("pulse_rate"):
        return predict_and_log("arrhythmia", [r["pulse_rate"]], r["pat_id"], read_id)
    return {"error": "لا توجد بيانات pulse_rate"}

@app.get("/predict/ecg/{read_id}")
def predict_ecg(read_id: int):
    ecg = supabase_request(f"tbl_ecg?read_id=eq.{read_id}&select=*")
    if not ecg: return {"error": "لا توجد قراءة ECG"}
    e = ecg[0]
    if e.get("signal_value"):
        return predict_and_log("ecg", [e["signal_value"]], e["pat_id"], read_id)
    return {"error": "لا توجد بيانات signal_value"}

@app.get("/predict/oxygen/{read_id}")
def predict_oxygen(read_id: int):
    reading = supabase_request(f"tbl_reading?read_id=eq.{read_id}&select=*")
    if not reading: return {"error": "لا توجد قراءة"}
    r = reading[0]
    if r.get("oxygen_saturation"):
        return predict_and_log("oxygen", [r["oxygen_saturation"]], r["pat_id"], read_id)
    return {"error": "لا توجد بيانات oxygen_saturation"}

@app.get("/predict/fall/{read_id}")
def predict_fall(read_id: int):
    reading = supabase_request(f"tbl_reading?read_id=eq.{read_id}&select=*")
    if not reading: return {"error": "لا توجد قراءة"}
    r = reading[0]
    if r.get("location"):
        return predict_and_log("fall", [r["location"]], r["pat_id"], read_id)
    return {"error": "لا توجد بيانات location"}

@app.get("/predict/temperature/{read_id}")
def predict_temperature(read_id: int):
    reading = supabase_request(f"tbl_reading?read_id=eq.{read_id}&select=*")
    if not reading: return {"error": "لا توجد قراءة"}
    r = reading[0]
    if r.get("temperature"):
        return predict_and_log("temperature", [r["temperature"]], r["pat_id"], read_id)
    return {"error": "لا توجد بيانات temperature"}
