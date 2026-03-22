def train_gps_model():
    readings = supabase_request("tbl_gps?select=*")
    X, y = [], []
    for r in readings:
        if r.get("latitude") is not None and r.get("longitude") is not None and r.get("diagnosis_label") is not None:
            X.append([r["latitude"], r["longitude"]])
            y.append(r["diagnosis_label"])
    if not X:
        return {"error": "لا توجد بيانات مكتملة"}
    X, y = np.array(X), np.array(y)
    y_cat = to_categorical(y, num_classes=2)
    X_train, X_test, y_train, y
