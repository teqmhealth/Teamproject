def train_gps_model():
    readings = supabase_request("tbl_gps?select=*")
    X, y = [], []
    for r in readings:
        if r.get("latitude") and r.get("longitude") and r.get("diagnosis_label") is not None:
            X.append([r["latitude"], r["longitude"]])
            y.append(r["diagnosis_label"])
    if not X: return {"error": "لا توجد بيانات مكتملة"}
    X, y = np.array(X), np.array(y)
    y_cat = to_categorical(y, num_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.3)
    model = Sequential([
        Dense(16, activation='relu', input_shape=(2,)),
        Dense(8, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
