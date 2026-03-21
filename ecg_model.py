import tensorflow as tf
import numpy as np

# تحميل النموذج المحفوظ
model = tf.keras.models.load_model("ecg_model.h5")

def predict_ecg(pulse_rate, oxygen, temp):
    X = np.array([[pulse_rate, oxygen, temp]])
    prediction = model.predict(X)
    pred_class = int(np.argmax(prediction[0]))
    return "بخير" if pred_class == 0 else "غير طبيعي"
