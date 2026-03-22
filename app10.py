from fastapi import FastAPI
import requests, os, numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from dotenv import load_dotenv

# تحميل القيم من ملف .env
load_dotenv()
SUPABASE_URL, SUPABASE_KEY = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")

app = FastAPI()

def supabase_request(endpoint):
    url = f"{SUPABASE_URL}/rest/v1/{endpoint}"
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
    return requests.get(url, headers=headers).json()

# -------------------------------
# دوال قراءة جميع الجداول
# -------------------------------
@app.get("/patients")       ; def get_all_patients(): return {"patients": supabase_request("tbl_patient?select=*")}
@app.get("/users")          ; def get_all_users(): return {"users": supabase_request("tbl_user?select=*")}
@app.get("/reports")        ; def get_all_reports(): return {"reports": supabase_request("tbl_report?select=*")}
@app.get("/readings")       ; def get_all_readings(): return {"readings": supabase_request("tbl_reading?select=*")}
@app.get("/ecg")            ; def get_all_ecg(): return {"ecg": supabase_request("tbl_ecg?select=*")}
@app.get("/oxygen")         ; def get_all_oxygen(): return {"oxygen": supabase_request("tbl_oxygen?select=*")}
@app.get("/temperature")    ; def get_all_temperature(): return {"temperature": supabase_request("tbl_temperature?select=*")}
@app.get("/fall")           ; def get_all_fall(): return {"fall": supabase_request("tbl_fall?select=*")}
@app.get("/heart_attack")   ; def get_all_heart_attack(): return {"heart_attack": supabase_request("tbl_heart_attack?select=*")}
@app.get("/arrhythmia")     ; def get_all_arrhythmia(): return {"arrhythmia": supabase_request("tbl_arrhythmia?select=*")}
@app.get("/gps")            ; def get_all_gps(): return {"gps": supabase_request("tbl_gps?select=*")}
@app.get("/maigghn")        ; def get_all_maigghn(): return {"maigghn": supabase_request("tbl_maigghn?select=*")}

# -------------------------------
# دوال قراءة حسب رقم المريض
# -------------------------------
@app.get("/ecg/by_patient/{pat_id}")        ; def get_ecg_by_patient(pat_id: int): return {"ecg": supabase_request(f"tbl_ecg?pat_id=eq.{pat_id}&select=*")}
@app.get("/oxygen/by_patient/{pat_id}")     ; def get_oxygen_by_patient(pat_id: int): return {"oxygen": supabase_request(f"tbl_oxygen?pat_id=eq.{pat_id}&select=*")}
@app.get("/temperature/by_patient/{pat_id}"); def get_temperature_by_patient(pat_id: int): return {"temperature": supabase_request(f"tbl_temperature?pat_id=eq.{pat_id}&select=*")}
@app.get("/fall/by_patient/{pat_id}")       ; def get_fall_by_patient(pat_id: int): return {"fall": supabase_request(f"tbl_fall?pat_id=eq.{pat_id}&select=*")}
@app.get("/heart_attack/by_patient/{pat_id}"); def get_heart_attack_by_patient(pat_id: int): return {"heart_attack": supabase_request(f"tbl_heart_attack?pat_id=eq.{pat_id}&select=*")}
@app.get("/arrhythmia/by_patient/{pat_id}") ; def get_arrhythmia_by_patient(pat_id: int): return {"arrhythmia": supabase_request(f"tbl_arrhythmia?pat_id=eq.{pat_id}&select=*")}
@app.get("/gps/by_patient/{pat_id}")        ; def get_gps_by_patient(pat_id: int): return {"gps": supabase_request(f"tbl_gps?pat_id=eq.{pat_id}&select=*")}
@app.get("/maigghn/by_patient/{pat_id}")    ; def get_maigghn_by_patient(pat_id: int): return {"maigghn": supabase_request(f"tbl_maigghn?pat_id=eq.{pat_id}&select=*")}

# -------------------------------
# دوال قراءة حسب رقم القراءة
# -------------------------------
@app.get("/ecg/by_reading/{read_id}")        ; def get_ecg_by_reading(read_id: int): return {"ecg": supabase_request(f"tbl_ecg?read_id=eq.{read_id}&select=*")}
@app.get("/oxygen/by_reading/{read_id}")     ; def get_oxygen_by_reading(read_id: int): return {"oxygen": supabase_request(f"tbl_oxygen?read_id=eq.{read_id}&select=*")}
@app.get("/temperature/by_reading/{read_id}"); def get_temperature_by_reading(read_id: int): return {"temperature": supabase_request(f"tbl_temperature?read_id=eq.{read_id}&select=*")}
@app.get("/fall/by_reading/{read_id}")       ; def get_fall_by_reading(read_id: int): return {"fall": supabase_request(f"tbl_fall?read_id=eq.{read_id}&select=*")}
@app.get("/heart_attack/by_reading/{read_id}"); def get_heart_attack_by_reading(read_id: int): return {"heart_attack": supabase_request(f"tbl_heart_attack?read_id=eq.{read_id}&select=*")}
@app.get("/arrhythmia/by_reading/{read_id}") ; def get_arrhythmia_by_reading(read_id: int): return {"arrhythmia": supabase_request(f"tbl_arrhythmia?read_id=eq.{read_id}&select=*")}
@app.get("/gps/by_reading/{read_id}")        ; def get_gps_by_reading(read_id: int): return {"gps": supabase_request(f"tbl_gps?read_id=eq.{read_id}&select=*")}
@app.get("/maigghn/by_reading/{read_id}")    ; def get_maigghn_by_reading(read_id: int): return {"maigghn": supabase_request(f"tbl_maigghn?read_id=eq.{read_id}&select=*")}

# -------------------------------
# دوال التدريب والتنبؤ (عام + حسب المريض + حسب القراءة + جماعي)
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
    filename = f"{model_name}_model.h5"
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

# هنا تضاف جميع مسارات /train/... و /predict/... لكل النماذج كما كتبناها سابقًا
# بالإضافة إلى /predict/all/by_patient/{pat_id} و /predict/all/by_reading/{read_id}

# -------------------------------
# ECG
# -------------------------------
@app.get("/predict/ecg/by_patient/{pat_id}")
def predict_ecg_by_patient(pat_id: int):
    readings = supabase_request(f"tbl_ecg?pat_id=eq.{pat_id}&select=*")
    return {"predictions": [predict_model_generic("ecg", [r["signal_value"]]) for r in readings if r.get("signal_value")]}

@app.get("/predict/ecg/by_reading/{read_id}")
def predict_ecg_by_reading(read_id: int):
    readings = supabase_request(f"tbl_ecg?read_id=eq.{read_id}&select=*")
    if not readings: return {"error": "لا توجد قراءة بهذا الرقم"}
    return predict_model_generic("ecg", [readings[0]["signal_value"]])


# -------------------------------
# Oxygen
# -------------------------------
@app.get("/predict/oxygen/by_patient/{pat_id}")
def predict_oxygen_by_patient(pat_id: int):
    readings = supabase_request(f"tbl_oxygen?pat_id=eq.{pat_id}&select=*")
    return {"predictions": [predict_model_generic("oxygen", [r["oxygen_saturation"]]) for r in readings if r.get("oxygen_saturation")]}

@app.get("/predict/oxygen/by_reading/{read_id}")
def predict_oxygen_by_reading(read_id: int):
    readings = supabase_request(f"tbl_oxygen?read_id=eq.{read_id}&select=*")
    if not readings: return {"error": "لا توجد قراءة بهذا الرقم"}
    return predict_model_generic("oxygen", [readings[0]["oxygen_saturation"]])


# -------------------------------
# Temperature
# -------------------------------
@app.get("/predict/temperature/by_patient/{pat_id}")
def predict_temperature_by_patient(pat_id: int):
    readings = supabase_request(f"tbl_temperature?pat_id=eq.{pat_id}&select=*")
    return {"predictions": [predict_model_generic("temperature", [r["temperature"]]) for r in readings if r.get("temperature")]}

@app.get("/predict/temperature/by_reading/{read_id}")
def predict_temperature_by_reading(read_id: int):
    readings = supabase_request(f"tbl_temperature?read_id=eq.{read_id}&select=*")
    if not readings: return {"error": "لا توجد قراءة بهذا الرقم"}
    return predict_model_generic("temperature", [readings[0]["temperature"]])


# -------------------------------
# Fall
# -------------------------------
@app.get("/predict/fall/by_patient/{pat_id}")
def predict_fall_by_patient(pat_id: int):
    readings = supabase_request(f"tbl_fall?pat_id=eq.{pat_id}&select=*")
    return {"predictions": [predict_model_generic("fall", [r["acceleration_x"], r["acceleration_y"], r["acceleration_z"]]) 
                            for r in readings if all([r.get("acceleration_x"), r.get("acceleration_y"), r.get("acceleration_z")])]}

@app.get("/predict/fall/by_reading/{read_id}")
def predict_fall_by_reading(read_id: int):
    readings = supabase_request(f"tbl_fall?read_id=eq.{read_id}&select=*")
    if not readings: return {"error": "لا توجد قراءة بهذا الرقم"}
    r = readings[0]
    return predict_model_generic("fall", [r["acceleration_x"], r["acceleration_y"], r["acceleration_z"]])


# -------------------------------
# Heart Attack
# -------------------------------
@app.get("/predict/heart_attack/by_patient/{pat_id}")
def predict_heart_attack_by_patient(pat_id: int):
    readings = supabase_request(f"tbl_heart_attack?pat_id=eq.{pat_id}&select=*")
    return {"predictions": [predict_model_generic("heart_attack", [r["cholesterol"], r["blood_pressure"]]) 
                            for r in readings if r.get("cholesterol") and r.get("blood_pressure")]}

@app.get("/predict/heart_attack/by_reading/{read_id}")
def predict_heart_attack_by_reading(read_id: int):
    readings = supabase_request(f"tbl_heart_attack?read_id=eq.{read_id}&select=*")
    if not readings: return {"error": "لا توجد قراءة بهذا الرقم"}
    r = readings[0]
    return predict_model_generic("heart_attack", [r["cholesterol"], r["blood_pressure"]])


# -------------------------------
# Arrhythmia
# -------------------------------
@app.get("/predict/arrhythmia/by_patient/{pat_id}")
def predict_arrhythmia_by_patient(pat_id: int):
    readings = supabase_request(f"tbl_arrhythmia?pat_id=eq.{pat_id}&select=*")
    return {"predictions": [predict_model_generic("arrhythmia", [r["ecg_signal"]]) for r in readings if r.get("ecg_signal")]}

@app.get("/predict/arrhythmia/by_reading/{read_id}")
def predict_arrhythmia_by_reading(read_id: int):
    readings = supabase_request(f"tbl_arrhythmia?read_id=eq.{read_id}&select=*")
    if not readings: return {"error": "لا توجد قراءة بهذا الرقم"}
    return predict_model_generic("arrhythmia", [readings[0]["ecg_signal"]])


# -------------------------------
# GPS
# -------------------------------
@app.get("/predict/gps/by_patient/{pat_id}")
def predict_gps_by_patient(pat_id: int):
    readings = supabase_request(f"tbl_gps?pat_id=eq.{pat_id}&select=*")
    return {"predictions": [predict_model_generic("gps", [r["latitude"], r["longitude"]]) 
                            for r in readings if r.get("latitude") and r.get("longitude")]}

@app.get("/predict/gps/by_reading/{read_id}")
def predict_gps_by_reading(read_id: int):
    readings = supabase_request(f"tbl_gps?read_id=eq.{read_id}&select=*")
    if not readings: return {"error": "لا توجد قراءة بهذا الرقم"}
    r = readings[0]
    return predict_model_generic("gps", [r["latitude"], r["longitude"]])


# -------------------------------
# Maigghn
# -------------------------------
@app.get("/predict/maigghn/by_patient/{pat_id}")
def predict_maigghn_by_patient(pat_id: int):
    readings = supabase_request(f"tbl_maigghn?pat_id=eq.{pat_id}&select=*")
    return {"predictions": [predict_model_generic("maigghn", [r["feature1"], r["feature2"]]) 
                            for r in readings if r.get("feature1") and r.get("feature2")]}

@app.get("/predict/maigghn/by_reading/{read_id}")
def predict_maigghn_by_reading(read_id: int):
    readings = supabase_request(f"tbl_maigghn?read_id=eq.{read_id}&select=*")
    if not readings: return {"error": "لا توجد قراءة بهذا الرقم"}
    r = readings[0]
    return predict_model_generic("maigghn", [r["feature1"], r["feature2"]])

# -------------------------------
# التنبؤ الجماعي لكل النماذج حسب المريض
# -------------------------------
@app.get("/predict/all/by_patient/{pat_id}")
def predict_all_by_patient(pat_id: int):
    results = {}

    # ECG
    ecg_readings = supabase_request(f"tbl_ecg?pat_id=eq.{pat_id}&select=*")
    if ecg_readings:
        results["ecg"] = [predict_model_generic("ecg", [r["signal_value"]]) for r in ecg_readings if r.get("signal_value")]

    # Oxygen
    oxygen_readings = supabase_request(f"tbl_oxygen?pat_id=eq.{pat_id}&select=*")
    if oxygen_readings:
        results["oxygen"] = [predict_model_generic("oxygen", [r["oxygen_saturation"]]) for r in oxygen_readings if r.get("oxygen_saturation")]

    # Temperature
    temp_readings = supabase_request(f"tbl_temperature?pat_id=eq.{pat_id}&select=*")
    if temp_readings:
        results["temperature"] = [predict_model_generic("temperature", [r["temperature"]]) for r in temp_readings if r.get("temperature")]

    # Fall
    fall_readings = supabase_request(f"tbl_fall?pat_id=eq.{pat_id}&select=*")
    if fall_readings:
        results["fall"] = [predict_model_generic("fall", [r["acceleration_x"], r["acceleration_y"], r["acceleration_z"]]) 
                           for r in fall_readings if all([r.get("acceleration_x"), r.get("acceleration_y"), r.get("acceleration_z")])]

    # Heart Attack
    ha_readings = supabase_request(f"tbl_heart_attack?pat_id=eq.{pat_id}&select=*")
    if ha_readings:
        results["heart_attack"] = [predict_model_generic("heart_attack", [r["cholesterol"], r["blood_pressure"]]) 
                                   for r in ha_readings if r.get("cholesterol") and r.get("blood_pressure")]

    # Arrhythmia
    arr_readings = supabase_request(f"tbl_arrhythmia?pat_id=eq.{pat_id}&select=*")
    if arr_readings:
        results["arrhythmia"] = [predict_model_generic("arrhythmia", [r["ecg_signal"]]) for r in arr_readings if r.get("ecg_signal")]

    # GPS
    gps_readings = supabase_request(f"tbl_gps?pat_id=eq.{pat_id}&select=*")
    if gps_readings:
        results["gps"] = [predict_model_generic("gps", [r["latitude"], r["longitude"]]) for r in gps_readings if r.get("latitude") and r.get("longitude")]

    # Maigghn
    mg_readings = supabase_request(f"tbl_maigghn?pat_id=eq.{pat_id}&select=*")
    if mg_readings:
        results["maigghn"] = [predict_model_generic("maigghn", [r["feature1"], r["feature2"]]) for r in mg_readings if r.get("feature1") and r.get("feature2")]

    return {"pat_id": pat_id, "predictions": results}


# -------------------------------
# التنبؤ الجماعي لكل النماذج حسب القراءة
# -------------------------------
@app.get("/predict/all/by_reading/{read_id}")
def predict_all_by_reading(read_id: int):
    results = {}

    # ECG
    ecg_readings = supabase_request(f"tbl_ecg?read_id=eq.{read_id}&select=*")
    if ecg_readings:
        results["ecg"] = predict_model_generic("ecg", [ecg_readings[0]["signal_value"]])

    # Oxygen
    oxygen_readings = supabase_request(f"tbl_oxygen?read_id=eq.{read_id}&select=*")
    if oxygen_readings:
        results["oxygen"] = predict_model_generic("oxygen", [oxygen_readings[0]["oxygen_saturation"]])

    # Temperature
    temp_readings = supabase_request(f"tbl_temperature?read_id=eq.{read_id}&select=*")
    if temp_readings:
        results["temperature"] = predict_model_generic("temperature", [temp_readings[0]["temperature"]])

    # Fall
    fall_readings = supabase_request(f"tbl_fall?read_id=eq.{read_id}&select=*")
    if fall_readings:
        r = fall_readings[0]
        results["fall"] = predict_model_generic("fall", [r["acceleration_x"], r["acceleration_y"], r["acceleration_z"]])

    # Heart Attack
    ha_readings = supabase_request(f"tbl_heart_attack?read_id=eq.{read_id}&select=*")
    if ha_readings:
        r = ha_readings[0]
        results["heart_attack"] = predict_model_generic("heart_attack", [r["cholesterol"], r["blood_pressure"]])

    # Arrhythmia
    arr_readings = supabase_request(f"tbl_arrhythmia?read_id=eq.{read_id}&select=*")
    if arr_readings:
        results["arrhythmia"] = predict_model_generic("arrhythmia", [arr_readings[0]["ecg_signal"]])

    # GPS
    gps_readings = supabase_request(f"tbl_gps?read_id=eq.{read_id}&select=*")
    if gps_readings:
        results["gps"] = predict_model_generic("gps", [gps_readings[0]["latitude"], gps_readings[0]["longitude"]])

    # Maigghn
    mg_readings = supabase_request(f"tbl_maigghn?read_id=eq.{read_id}&select=*")
    if mg_readings:
        results["maigghn"] = predict_model_generic("maigghn", [mg_readings[0]["feature1"], mg_readings[0]["feature2"]])

    return {"read_id": read_id, "predictions": results}


