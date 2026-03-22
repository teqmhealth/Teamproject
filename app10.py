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
# أمثلة لمسارات التدريب
# -------------------------------
@app.get("/train/ecg/by_patient/{pat_id}")
def train_ecg_by_patient(pat_id: int):
    return train_model_generic("tbl_ecg", ["signal_value"], "diagnosis_label", "ecg_model.keras", 2, filter_query=f"pat_id=eq.{pat_id}")

@app.get("/train/ecg/by_reading/{read_id}")
def train_ecg_by_reading(read_id: int):
    return train_model_generic("tbl_ecg", ["signal_value"], "diagnosis_label", "ecg_model.keras", 2, filter_query=f"read_id=eq.{read_id}")

# -------------------------------
# أمثلة لمسارات التنبؤ
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
# مسارات التنبؤ الجماعي
# -------------------------------
@app.get("/predict/all/by_patient/{pat_id}")
def predict_all_by_patient(pat_id: int):
    results = {}
    ecg_readings = supabase_request(f"tbl_ecg?pat_id=eq.{pat_id}&select=*")
    if ecg_readings: results["ecg"] = [predict_model_generic("ecg", [r["signal_value"]]) for r in ecg_readings if r.get("signal_value")]
    return {"pat_id": pat_id, "predictions": results}

@app.get("/predict/all/by_reading/{read_id}")
def predict_all_by_reading(read_id: int):
    results = {}
    ecg_readings = supabase_request(f"tbl_ecg?read_id=eq.{read_id}&select=*")
    if ecg_readings: results["ecg"] = predict_model_generic("ecg", [ecg_readings[0]["signal_value"]])
    return {"read_id": read_id, "predictions": results}
