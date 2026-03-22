import os, requests, numpy as np
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from dotenv import load_dotenv

load_dotenv()
SUPABASE_URL, SUPABASE_KEY = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")

def supabase_request(endpoint):
    url = f"{SUPABASE_URL}/rest/v1/{endpoint}"
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
    return requests.get(url, headers=headers).json()

def train_ecg_model():
    readings = supabase_request("tbl_ecg?select=*")
    X, y = [], []
    for r in readings:
        if r.get("signal_value") and r.get("diagnosis_label") is not None:
            X.append([r["signal_value"]])
            y.append(r["diagnosis_label"])
    if not X: return {"error": "لا توجد بيانات مكتملة"}
    X, y = np.array(X), np.array(y)
    y_cat = to_categorical(y, num_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.3)
    model = Sequential([
        Dense(16, activation='relu', input_shape=(1,)),
        Dense(8, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)
    model.save("ecg_model.keras")
    return {"accuracy": float(model.evaluate(X_test, y_test, verbose=0)[1])}
