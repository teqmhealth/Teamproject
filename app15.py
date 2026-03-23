from fastapi import FastAPI
import os, pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv

app = FastAPI()

# تحميل القيم من .env
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 🟢 توليد تقرير بنفس حقول tb_report
def generate_report(rep_id, pat_id, diagnosis, recommendation):
    return {
        "rep_id": rep_id,
        "pat_id": pat_id,
        "rep_date": pd.Timestamp.now(tz="UTC").isoformat(),
        "rep_diagnosis": diagnosis,
        "rep_recommendation": recommendation
    }

# 🟢 توليد تنبيه ديناميكي مبسط
def generate_alert(alert_id, pat_id, emergency_flag, feature_name=None, value=None):
    if emergency_flag == 1:
        if feature_name == "oxygen_saturation" and value is not None and value < 85:
            alert_type = "Severe Oxygen Drop"
            alert_message = "تشبع الأكسجين منخفض جدًا، الحالة حرجة للغاية."
        elif feature_name == "temperature" and value is not None and value > 40:
            alert_type = "High Fever"
            alert_message = "درجة الحرارة مرتفعة جدًا، قد تكون عدوى خطيرة."
        elif feature_name == "pulse_rate" and value is not None and value > 120:
            alert_type = "Arrhythmia Risk"
            alert_message = "معدل النبض غير طبيعي، هناك خطر اضطراب نظم القلب."
        else:
            alert_type = "Critical Condition"
            alert_message = "الحالة حرجة، يجب مراجعة الطبيب فورًا."
        is_seen = False
    else:
        alert_type = "Healthy"
        alert_message = "الوضع مستقر، لا ترهق نفسك ولا تهمل أدويتك."
        is_seen = True

    return {
        "alert_id": alert_id,
        "pat_id": pat_id,
        "alert_type": alert_type,
        "alert_message": alert_message,
        "alert_timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "is_seen": is_seen
    }

# 🟢 حفظ التقرير في قاعدة البيانات مع رسالة
def save_report_to_db(report):
    try:
        response = supabase.table("tb_report").insert(report).execute()
        if response.data:
            return {"status": "success", "message": "تم حفظ التقرير بنجاح"}
        else:
            return {"status": "error", "message": "تعذر حفظ التقرير", "details": response}
    except Exception as e:
        return {"status": "error", "message": "خطأ أثناء حفظ التقرير", "details": str(e)}

# 🟢 حفظ التنبيه في قاعدة البيانات مع رسالة
def save_alert_to_db(alert):
    try:
        response = supabase.table("tb_alert").insert(alert).execute()
        if response.data:
            return {"status": "success", "message": "تم حفظ التنبيه بنجاح"}
        else:
            return {"status": "error", "message": "تعذر حفظ التنبيه", "details": response}
    except Exception as e:
        return {"status": "error", "message": "خطأ أثناء حفظ التنبيه", "details": str(e)}

# 🟢 التنبؤ بحسب رقم القراءة
@app.get("/predict/reading/{read_id}")
def predict_by_reading(read_id: int):
    reading = supabase.table("tbl_reading").select("*").eq("read_id", read_id).execute()
    if not reading.data:
        return {"error": f"لا توجد قراءة بالرقم {read_id}"}

    selected_reading = reading.data[0]
    pat_id = selected_reading.get("pat_id", 0)

    emergency_flag = int((selected_reading["temperature"] > 38) or (selected_reading["oxygen_saturation"] < 90))

    diagnosis = "خطر عالي" if emergency_flag == 1 else "الوضع مستقر"
    recommendation = "يجب مراجعة الطبيب فورًا" if emergency_flag == 1 else "استمر في المتابعة الدورية فقط"

    report = generate_report(100 + read_id, pat_id, diagnosis, recommendation)
    alert = generate_alert(200 + read_id, pat_id, emergency_flag, feature_name="temperature", value=selected_reading["temperature"])

    report_status = save_report_to_db(report)
    alert_status = save_alert_to_db(alert)

    return {
        "read_id": read_id,
        "reading": selected_reading,
        "report": report,
        "report_status": report_status,
        "alert": alert,
        "alert_status": alert_status
    }

# 🟢 التنبؤ بحسب المريض (آخر قراءة له)
@app.get("/predict/patient/{pat_id}")
def predict_by_patient(pat_id: int):
    reading = supabase.table("tbl_reading").select("*").eq("pat_id", pat_id).order("created_at", desc=True).limit(1).execute()
    if not reading.data:
        return {"error": f"لا توجد قراءات للمريض {pat_id}"}

    selected_reading = reading.data[0]

    emergency_flag = int((selected_reading["temperature"] > 38) or (selected_reading["oxygen_saturation"] < 90))

    diagnosis = "خطر عالي" if emergency_flag == 1 else "الوضع مستقر"
    recommendation = "يجب مراجعة الطبيب فورًا" if emergency_flag == 1 else "استمر في المتابعة الدورية فقط"

    report = generate_report(1000 + pat_id, pat_id, diagnosis, recommendation)
    alert = generate_alert(2000 + pat_id, pat_id, emergency_flag, feature_name="oxygen_saturation", value=selected_reading["oxygen_saturation"])

    report_status = save_report_to_db(report)
    alert_status = save_alert_to_db(alert)

    return {
        "pat_id": pat_id,
        "reading": selected_reading,
        "report": report,
        "report_status": report_status,
        "alert": alert,
        "alert_status": alert_status
    }

# 🟢 إرجاع آخر تنبيهين من جدول tb_alert
@app.get("/alerts/latest")
def get_latest_alerts():
    try:
        response = supabase.table("tb_alert").select("*").order("alert_timestamp", desc=True).limit(2).execute()
        if response.data:
            return {"status": "success", "latest_alerts": response.data}
        else:
            return {"status": "error", "message": "لا توجد تنبيهات"}
    except Exception as e:
        return {"status": "error", "message": "خطأ أثناء جلب التنبيهات", "details": str(e)}
