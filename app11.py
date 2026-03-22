import os
os.environ["KERAS_BACKEND"] = "torch"   # استخدام PyTorch كـ backend

import requests, numpy as np
from sklearn.model_selection import train_test_split
from keras_core import Sequential
from keras_core.layers import Dense
from keras_core.utils import to_categorical
from keras_core.models import load_model
from dotenv import load_dotenv
from fastapi import FastAPI

# تحميل القيم من ملف .env
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

app = FastAPI()

# -------------------------------
# دالة عامة للتعامل مع Supabase
# -------------------------------
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
# تدريب شبكة عصبية صغيرة على جدول القراءات
# -------------------------------
@app.get("/train/reading")
def train_reading_model():
    readings = supabase_request("tbl_reading?select=*")
    X, y = [], []
    for r in readings:
        if r.get("oxygen_saturation") and r.get("pulse_rate") and r.get("temperature"):
            X.append([r["oxygen_saturation"], r["pulse_rate"], r["temperature"]])
            # مثال مبسط: إذا الأكسجين أقل من 90 → حالة طارئة
            y.append(1 if r["oxygen_saturation"] < 90 else 0)

    if not X:
        return {"error": "لا توجد بيانات مكتملة"}

    X, y = np.array(X), np.array(y)
    y_cat = to_categorical(y, num_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.3)

    model = Sequential([
        Dense(8, activation="relu", input_shape=(3,)),
        Dense(2, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=20, batch_size=4, verbose=0)
    model.save("reading_model.keras")

    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return {"message": "تم تدريب النموذج", "accuracy": float(accuracy), "samples": len(readings)}

# -------------------------------
# التنبؤ + إنشاء تقرير وتنبيه
# -------------------------------
@app.get("/predict/reading/{pat_id}")
def predict_reading(pat_id: int):
    readings = supabase_request(f"tbl_reading?pat_id=eq.{pat_id}&select=*")
    if not os.path.exists("reading_model.keras"):
        return {"error": "النموذج غير موجود، درّبه أولاً"}

    model = load_model("reading_model.keras")
    predictions = []
    for r in readings:
        if r.get("oxygen_saturation") and r.get("pulse_rate") and r.get("temperature"):
            features = [r["oxygen_saturation"], r["pulse_rate"], r["temperature"]]
            probs = model.predict(np.array([features]))
            pred_class = int(np.argmax(probs))
            predictions.append({
                "features": features,
                "prediction_class": pred_class,
                "probabilities": probs.tolist()
            })

            # إنشاء تقرير في tbl_report
            supabase_request("tbl_report", method="POST", data={
                "pat_id": pat_id,
                "report_text": f"Prediction: {pred_class}, Oxygen={features[0]}, Pulse={features[1]}, Temp={features[2]}"
            })

            # إذا الحالة طارئة → إنشاء تنبيه في tbl_alert
            if pred_class == 1:
                supabase_request("tbl_alert", method="POST", data={
                    "pat_id": pat_id,
                    "alert_text": "حالة طارئة: انخفاض الأكسجين",
                    "state": True
                })

    return {"predictions": predictions}    X, y = [], []
    for r in readings:
        if r.get("oxygen_saturation") and r.get("pulse_rate") and r.get("temperature"):
            X.append([r["oxygen_saturation"], r["pulse_rate"], r["temperature"]])
            # مثال مبسط: إذا الأكسجين أقل من 90 → حالة طارئة
            y.append(1 if r["oxygen_saturation"] < 90 else 0)

    if not X:
        return {"error": "لا توجد بيانات مكتملة"}

    X, y = np.array(X), np.array(y)
    y_cat = to_categorical(y, num_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.3)

    model = Sequential([
        Dense(8, activation="relu", input_shape=(3,)),
        Dense(2, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=20, batch_size=4, verbose=0)
    model.save("reading_model.keras")

    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return {"message": "تم تدريب النموذج", "accuracy": float(accuracy), "samples": len(readings)}

# -------------------------------
# التنبؤ + إنشاء تقرير وتنبيه
# -------------------------------
@app.get("/predict/reading/{pat_id}")
def predict_reading(pat_id: int):
    readings = supabase_request(f"tbl_reading?pat_id=eq.{pat_id}&select=*")
    if not os.path.exists("reading_model.keras"):
        return {"error": "النموذج غير موجود، درّبه أولاً"}

    model = load_model("reading_model.keras")
    predictions = []
    for r in readings:
        if r.get("oxygen_saturation") and r.get("pulse_rate") and r.get("temperature"):
            features = [r["oxygen_saturation"], r["pulse_rate"], r["temperature"]]
            probs = model.predict(np.array([features]))
            pred_class = int(np.argmax(probs))
            predictions.append({
                "features": features,
                "prediction_class": pred_class,
                "probabilities": probs.tolist()
            })

            # إنشاء تقرير في tbl_report
            supabase_request("tbl_report", method="POST", data={
                "pat_id": pat_id,
                "report_text": f"Prediction: {pred_class}, Oxygen={features[0]}, Pulse={features[1]}, Temp={features[2]}"
            })

            # إذا الحالة طارئة → إنشاء تنبيه في tbl_alert
            if pred_class == 1:
                supabase_request("tbl_alert", method="POST", data={
                    "pat_id": pat_id,
                    "alert_text": "حالة طارئة: انخفاض الأكسجين",
                    "state": True
                })

    return {"predictions": predictions}
