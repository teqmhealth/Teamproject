
def train_heart_attack_model():
    readings = supabase_request("tbl_heart_attack?select=*")
    X, y = [], []
    for r in readings:
        if r.get("cholesterol") is not None and r.get("blood_pressure") is not None and r.get("diagnosis_label") is not None:
            X.append([r["cholesterol"], r["blood_pressure"]])
            y.append(r["diagnosis_label"])
    if not X:
        return {"error": "لا توجد بيانات مكتملة"}
    X, y = np.array(X), np.array(y)
    y_cat = to_categorical(y, num_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.3)
    model = Sequential([Dense(16, activation='relu', input_shape=(2,)),
                        Dense(8, activation='relu'),
                        Dense(2, activation='softmax')])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)
    model.save("heart_attack_model.h5")
    return {"accuracy": float(model.evaluate(X_test, y_test, verbose=0)[1])}
