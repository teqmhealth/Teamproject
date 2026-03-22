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

def supabase_request(endpoint: str):
    url = f"{SUPABASE_URL}/rest/v1/{endpoint}"
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
    return requests.get(url, headers=headers).json()

# -------------------------------
# تدريب شبكة عصبية صغيرة على جدول القراءات
# -------------------------------
@app.get("/train/reading")
def train_reading_model():
    readings = supabase_request("tbl_reading?select=*")
    X, y = [], []
    for r in readings:
        if r.get("oxygen_saturation") and r.get("pulse_rate") and r.get("temperature") and r.get("pat_id"):
            X.append([r["oxygen_saturation"], r["pulse_rate"], r["temperature"]])
            # هنا نفترض أن label هو حالة الطوارئ من جدول المريض (مثال مبسط)
            y.append(1 if r.get("oxygen_saturation") < 90 else 0)

    if not X:
        return {"error": "لا توجد بيانات مكتملة"}

    X, y = np.array(X), np.array(y)
    y_cat = to_categorical(y, num_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.3)

    # شبكة عصبية صغيرة جدًا
    model = Sequential([
        Dense(8, activation="relu", input_shape=(3,)),   # 3 مدخلات فقط
        Dense(2, activation="softmax")                   # إخراج ثنائي
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=20, batch_size=4, verbose=0)
    model.save("reading_model.keras")

    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return {"message": "تم تدريب النموذج", "accuracy": float(accuracy), "samples": len(readings)}

# -------------------------------
# التنبؤ باستخدام النموذج
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
            predictions.append({
                "features": features,
                "prediction_class": int(np.argmax(probs)),
                "probabilities": probs.tolist()
            })
    return {"predictions": predictions}    if filter_query:
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
        Dense(16, activation="relu", input_shape=(len(features),)),
        Dense(8, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)
    model.save(filename)

    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return {
        "message": f"تم تدريب النموذج وحفظه في {filename}",
        "accuracy": float(accuracy),
        "samples": len(readings)
    }

# -------------------------------
# دالة عامة للتنبؤ
# -------------------------------
def predict_model_generic(model_name: str, features: list):
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
# أمثلة لمسارات التدريب والتنبؤ
# -------------------------------
@app.get("/train/ecg/by_patient/{pat_id}")
def train_ecg_by_patient(pat_id: int):
    return train_model_generic(
        "tbl_ecg", ["signal_value"], "diagnosis_label",
        "ecg_model.keras", num_classes=2,
        filter_query=f"pat_id=eq.{pat_id}"
    )

@app.get("/predict/ecg/by_patient/{pat_id}")
def predict_ecg_by_patient(pat_id: int):
    readings = supabase_request(f"tbl_ecg?pat_id=eq.{pat_id}&select=*")
    return {
        "predictions": [
            predict_model_generic("ecg", [r["signal_value"]])
            for r in readings if r.get("signal_value")
        ]
    }        row = []
        for f in features:
            if r.get(f) is None:
                break
            row.append(r.get(f))
        else:
            if r.get(label_field) is None:
                continue
            y.append(r.get(label_field))
            X.append(row)

    if not X:
        return {"error": "لا توجد بيانات مكتملة للتدريب"}

    X, y = np.array(X), np.array(y)
    y_cat = to_categorical(y, num_classes=num_classes)
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.3)

    model = Sequential([
        Dense(16, activation='relu', input_shape=(len(features),)),
        Dense(8, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)
    model.save(filename)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return {"message": f"تم تدريب النموذج وحفظه في {filename}", "accuracy": float(accuracy), "samples": len(readings)}

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
# أمثلة لمسارات التدريب والتنبؤ
# -------------------------------
@app.get("/train/ecg/by_patient/{pat_id}")
def train_ecg_by_patient(pat_id: int):
    return train_model_generic("tbl_ecg", ["signal_value"], "diagnosis_label", "ecg_model.keras", 2, filter_query=f"pat_id=eq.{pat_id}")

@app.get("/predict/ecg/by_patient/{pat_id}")
def predict_ecg_by_patient(pat_id: int):
    readings = supabase_request(f"tbl_ecg?pat_id=eq.{pat_id}&select=*")
    return {"predictions": [predict_model_generic("ecg", [r["signal_value"]]) for r in readings if r.get("signal_value")]}
