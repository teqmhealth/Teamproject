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
def train_model_generic(table_name, features, label_field, filename, num_classes=2, filter_query=None):
    endpoint = f"{table_name}?select=*"
    if filter_query:
        endpoint = f"{table_name}?{filter_query}&select=*"

    readings = supabase_request(endpoint)
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
        Dense(8, activation="relu", input_shape=(len(features),)),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=20, batch_size=4, verbose=0)
    model.save(filename)

    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return {"message": f"تم تدريب النموذج وحفظه في {filename}", "accuracy": float(accuracy), "samples": len(readings)}

# -------------------------------
# دالة عامة للتنبؤ
# -------------------------------
def predict_model_generic(model_name, features):
    filename = f"{model_name}_model.keras"
    if not os.path.exists(filename):
        return {"error": f"النموذج {filename} غير موجود، درّبه أولاً"}

    model = load_model(filename)
    prediction_probs = model.predict(np.array([features]))
    prediction_class = int(np.argmax(prediction_probs))

    return {
        "model": model_name,
        "features": features,
        "prediction_class": prediction_class,
        "probabilities": prediction_probs.tolist()
    }

# -------------------------------
# مسارات تدريب وتنبؤ لكل نموذج
# -------------------------------

@app.get("/train/gps/{pat_id}")
def train_gps(pat_id: int):
    return train_model_generic("tbl_reading", ["location"], "is_emergency",
                               "gps_model.keras", 2, filter_query=f"pat_id=eq.{pat_id}")

@app.get("/predict/gps/{pat_id}")
def predict_gps(pat_id: int):
    readings = supabase_request(f"tbl_reading?pat_id=eq.{pat_id}&select=*")
    return {"predictions": [predict_model_generic("gps", [r["location"]]) for r in readings if r.get("location")]}

@app.get("/train/heart_attack/{pat_id}")
def train_heart_attack(pat_id: int):
    return train_model_generic("tbl_reading", ["pulse_rate"], "is_emergency",
                               "heart_attack_model.keras", 2, filter_query=f"pat_id=eq.{pat_id}")

@app.get("/predict/heart_attack/{pat_id}")
def predict_heart_attack(pat_id: int):
    readings = supabase_request(f"tbl_reading?pat_id=eq.{pat_id}&select=*")
    return {"predictions": [predict_model_generic("heart_attack", [r["pulse_rate"]]) for r in readings if r.get("pulse_rate")]}

@app.get("/train/arrhythmia/{pat_id}")
def train_arrhythmia(pat_id: int):
    return train_model_generic("tbl_reading", ["pulse_rate"], "is_emergency",
                               "arrhythmia_model.keras", 2, filter_query=f"pat_id=eq.{pat_id}")

@app.get("/predict/arrhythmia/{pat_id}")
def predict_arrhythmia(pat_id: int):
    readings = supabase_request(f"tbl_reading?pat_id=eq.{pat_id}&select=*")
    return {"predictions": [predict_model_generic("arrhythmia", [r["pulse_rate"]]) for r in readings if r.get("pulse_rate")]}

@app.get("/train/ecg/{pat_id}")
def train_ecg(pat_id: int):
    return train_model_generic("tbl_ecg", ["signal_value"], "diagnosis_label",
                               "ecg_model.keras", 2, filter_query=f"pat_id=eq.{pat_id}")

@app.get("/predict/ecg/{pat_id}")
def predict_ecg(pat_id: int):
    readings = supabase_request(f"tbl_ecg?pat_id=eq.{pat_id}&select=*")
    return {"predictions": [predict_model_generic("ecg", [r["signal_value"]]) for r in readings if r.get("signal_value")]}

@app.get("/train/oxygen/{pat_id}")
def train_oxygen(pat_id: int):
    return train_model_generic("tbl_reading", ["oxygen_saturation"], "is_emergency",
                               "oxygen_model.keras", 2, filter_query=f"pat_id=eq.{pat_id}")

@app.get("/predict/oxygen/{pat_id}")
def predict_oxygen(pat_id: int):
    readings = supabase_request(f"tbl_reading?pat_id=eq.{pat_id}&select=*")
    return {"predictions": [predict_model_generic("oxygen", [r["oxygen_saturation"]]) for r in readings if r.get("oxygen_saturation")]}

@app.get("/train/fall/{pat_id}")
def train_fall(pat_id: int):
    return train_model_generic("tbl_reading", ["location"], "is_emergency",
                               "fall_model.keras", 2, filter_query=f"pat_id=eq.{pat_id}")

@app.get("/predict/fall/{pat_id}")
def predict_fall(pat_id: int):
    readings = supabase_request(f"tbl_reading?pat_id=eq.{pat_id}&select=*")
    return {"predictions": [predict_model_generic("fall", [r["location"]]) for r in readings if r.get("location")]}

@app.get("/train/temperature/{pat_id}")
def train_temperature(pat_id: int):
    return train_model_generic("tbl_reading", ["temperature"], "is_emergency",
                               "temperature_model.keras", 2, filter_query=f"pat_id=eq.{pat_id}")

@app.get("/predict/temperature/{pat_id}")
def predict_temperature(pat_id: int):
    readings = supabase_request(f"tbl_reading?pat_id=eq.{pat_id}&select=*")
    return {"predictions": [predict_model_generic("temperature", [r["temperature"]]) for r in readings if r.get("temperature")]}
