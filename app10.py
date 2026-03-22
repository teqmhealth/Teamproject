from fastapi import FastAPI
import requests, os, numpy as np
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.models import load_model
from dotenv import load_dotenv

# تحميل القيم من ملف .env
load_dotenv()
SUPABASE_URL, SUPABASE_KEY = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")

app = FastAPI()

# -------------------------------
# دالة عامة لاستدعاء بيانات من Supabase
# -------------------------------
def supabase_request(endpoint):
    url = f"{SUPABASE_URL}/rest/v1/{endpoint}"
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
    return requests.get(url, headers=headers).json()

# -------------------------------
# دوال عامة للتدريب والتنبؤ
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
# مسارات قراءة البيانات
# -------------------------------
@app.get("/patients")
def get_all_patients():
    return {"patients": supabase_request("tbl_patient?select=*")}

@app.get("/users")
def get_all_users():
    return {"users": supabase_request("tbl_user?select=*")}

@app.get("/reports")
def get_all_reports():
    return {"reports": supabase_request("tbl_report?select=*")}

@app.get("/readings")
def get_all_readings():
    return {"readings": supabase_request("tbl_reading?select=*")}

@app.get("/ecg")
def get_all_ecg():
    return {"ecg": supabase_request("tbl_reading?select=*")}

@app.get("/oxygen")
def get_all_oxygen():
    return {"oxygen": supabase_request("tbl_reading?select=*")}

@app.get("/temperature")
def get_all_temperature():
    return {"temperature": supabase_request("tbl_reading?select=*")}

@app.get("/fall")
def get_all_fall():
    return {"fall": supabase_request("tbl_reading?select=*")}

@app.get("/heart_attack")
def get_all_heart_attack():
    return {"heart_attack": supabase_request("tbl_reading?select=*")}

@app.get("/arrhythmia")
def get_all_arrhythmia():
    return {"arrhythmia": supabase_request("tbl_reading?select=*")}

@app.get("/gps")
def get_all_gps():
    return {"gps": supabase_request("tbl_reading?select=*")}

@app.get("/maigghn")
def get_all_maigghn():
    return {"maigghn": supabase_request("tbl_reading?select=*")}

# -------------------------------
# أمثلة لمسارات التدريب والتنبؤ
# -------------------------------
@app.get("/train/ecg/by_patient/{pat_id}")
def train_ecg_by_patient(pat_id: int):
    return train_model_generic("tbl_reading", ["signal_value"], "diagnosis_label", "ecg_model.keras", 2, filter_query=f"pat_id=eq.{pat_id}")

@app.get("/predict/ecg/by_patient/{pat_id}")
def predict_ecg_by_patient(pat_id: int):
    readings = supabase_request(f"tbl_reading?pat_id=eq.{pat_id}&select=*")
    return {"predictions": [predict_model_generic("ecg", [r["signal_value"]]) for r in readings if r.get("signal_value")]}
# -------------------------------
# مسارات التدريب لبقية النماذج
# -------------------------------
@app.get("/train/oxygen/by_patient/{pat_id}")
def train_oxygen_by_patient(pat_id: int):
    return train_model_generic("tbl_reading", ["oxygen_value"], "diagnosis_label", "oxygen_model.keras", 2, filter_query=f"pat_id=eq.{pat_id}")

@app.get("/train/temperature/by_patient/{pat_id}")
def train_temperature_by_patient(pat_id: int):
    return train_model_generic("tbl_reading", ["temp_value"], "diagnosis_label", "temperature_model.keras", 2, filter_query=f"pat_id=eq.{pat_id}")

@app.get("/train/fall/by_patient/{pat_id}")
def train_fall_by_patient(pat_id: int):
    return train_model_generic("tbl_reading", ["fall_value"], "diagnosis_label", "fall_model.keras", 2, filter_query=f"pat_id=eq.{pat_id}")

@app.get("/train/heart_attack/by_patient/{pat_id}")
def train_heart_attack_by_patient(pat_id: int):
    return train_model_generic("tbl_reading", ["attack_value"], "diagnosis_label", "heart_attack_model.keras", 2, filter_query=f"pat_id=eq.{pat_id}")

@app.get("/train/arrhythmia/by_patient/{pat_id}")
def train_arrhythmia_by_patient(pat_id: int):
    return train_model_generic("tbl_reading", ["arrhythmia_value"], "diagnosis_label", "arrhythmia_model.keras", 2, filter_query=f"pat_id=eq.{pat_id}")

@app.get("/train/gps/by_patient/{pat_id}")
def train_gps_by_patient(pat_id: int):
    return train_model_generic("tbl_reading", ["latitude","longitude"], "diagnosis_label", "gps_model.keras", 2, filter_query=f"pat_id=eq.{pat_id}")

@app.get("/train/maigghn/by_patient/{pat_id}")
def train_maigghn_by_patient(pat_id: int):
    return train_model_generic("tbl_reading", ["maigghn_value"], "diagnosis_label", "maigghn_model.keras", 2, filter_query=f"pat_id=eq.{pat_id}")

# -------------------------------
# مسارات التنبؤ لبقية النماذج
# -------------------------------
@app.get("/predict/oxygen/by_patient/{pat_id}")
def predict_oxygen_by_patient(pat_id: int):
    readings = supabase_request(f"tbl_reading?pat_id=eq.{pat_id}&select=*")
    return {"predictions": [predict_model_generic("oxygen", [r["oxygen_value"]]) for r in readings if r.get("oxygen_value")]}

@app.get("/predict/temperature/by_patient/{pat_id}")
def predict_temperature_by_patient(pat_id: int):
    readings = supabase_request(f"tbl_reading?pat_id=eq.{pat_id}&select=*")
    return {"predictions": [predict_model_generic("temperature", [r["temp_value"]]) for r in readings if r.get("temp_value")]}

@app.get("/predict/fall/by_patient/{pat_id}")
def predict_fall_by_patient(pat_id: int):
    readings = supabase_request(f"tbl_reading?pat_id=eq.{pat_id}&select=*")
    return {"predictions": [predict_model_generic("fall", [r["fall_value"]]) for r in readings if r.get("fall_value")]}

@app.get("/predict/heart_attack/by_patient/{pat_id}")
def predict_heart_attack_by_patient(pat_id: int):
    readings = supabase_request(f"tbl_reading?pat_id=eq.{pat_id}&select=*")
    return {"predictions": [predict_model_generic("heart_attack", [r["attack_value"]]) for r in readings if r.get("attack_value")]}

@app.get("/predict/arrhythmia/by_patient/{pat_id}")
def predict_arrhythmia_by_patient(pat_id: int):
    readings = supabase_request(f"tbl_reading?pat_id=eq.{pat_id}&select=*")
    return {"predictions": [predict_model_generic("arrhythmia", [r["arrhythmia_value"]]) for r in readings if r.get("arrhythmia_value")]}

@app.get("/predict/gps/by_patient/{pat_id}")
def predict_gps_by_patient(pat_id: int):
    readings = supabase_request(f"tbl_reading?pat_id=eq.{pat_id}&select=*")
    return {"predictions": [predict_model_generic("gps", [r["latitude"], r["longitude"]]) for r in readings if r.get("latitude") and r.get("longitude")]}

@app.get("/predict/maigghn/by_patient/{pat_id}")
def predict_maigghn_by_patient(pat_id: int):
    readings = supabase_request(f"tbl_reading?pat_id=eq.{pat_id}&select=*")
    return {"predictions": [predict_model_generic("maigghn", [r["maigghn_value"]]) for r in readings if r.get("maigghn_value")]}
