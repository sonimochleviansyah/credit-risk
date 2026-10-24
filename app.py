import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Memuat Artefak Model, Scaler, dan Kolom Fitur
model = joblib.load("model_credit_risk.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # 1. Ambil input dari Form HTML
        age = float(request.form["age"])
        income = float(request.form["income"])
        emp_length = float(request.form["emp_length"])
        amount = float(request.form["amount"])
        int_rate = float(request.form["int_rate"])
        cred_hist = float(request.form["cred_hist"])

        home_ownership = request.form["home_ownership"]
        loan_intent = request.form["loan_intent"]
        loan_grade = request.form["loan_grade"]
        cb_default = request.form["cb_default"]

        # Hitung rasio pinjaman terhadap pendapatan
        percent_income = amount / income if income > 0 else 0

        # 2. Inisialisasi dictionary fitur sesuai kolom hasil One-Hot Encoding
        input_data = {col: 0 for col in feature_columns}

        # Isi fitur numerik
        input_data["person_age"] = age
        input_data["person_income"] = income
        input_data["person_emp_length"] = emp_length
        input_data["loan_amnt"] = amount
        input_data["loan_int_rate"] = int_rate
        input_data["loan_percent_income"] = percent_income
        input_data["cb_person_cred_hist_length"] = cred_hist

        # Set nilai 1 pada variabel kategorikal yang sesuai
        if f"person_home_ownership_{home_ownership}" in input_data:
            input_data[f"person_home_ownership_{home_ownership}"] = 1
        if f"loan_intent_{loan_intent}" in input_data:
            input_data[f"loan_intent_{loan_intent}"] = 1
        if f"loan_grade_{loan_grade}" in input_data:
            input_data[f"loan_grade_{loan_grade}"] = 1
        if f"cb_person_default_on_file_{cb_default}" in input_data:
            input_data[f"cb_person_default_on_file_{cb_default}"] = 1

        # 3. Konversi ke DataFrame & Normalisasi dengan Scaler
        df_input = pd.DataFrame([input_data])
        scaled_input = scaler.transform(df_input)

        # 4. Melakukan Prediksi
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1] * 100

        if prediction == 1:
            status = "BERISIKO GAGAL BAYAR (HIGH RISK)"
            badge_class = "danger"
        else:
            status = "LANCAR / KELAYAKAN DISETUJUI (LOW RISK)"
            badge_class = "success"

        return render_template(
            "index.html",
            prediction_text=status,
            probability_text=f"{probability:.2f}%",
            badge_class=badge_class,
        )


if __name__ == "__main__":
    app.run(debug=True)
