import joblib
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("[1/5] Memuat dataset...")
df = pd.read_csv("credit_risk_dataset.csv")

print("[2/5] Cleaning & Preprocessing data...")
# Filter outlier usia dan lama kerja
df = df[df["person_age"] <= 100]
df = df[df["person_emp_length"] <= 60]

# Imputasi missing values dengan median
df["person_emp_length"] = df["person_emp_length"].fillna(
    df["person_emp_length"].median()
)
df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_int_rate"].median())

# Encoding variabel kategorikal
categorical_cols = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]
df_encoded = pd.get_dummies(
    df, columns=categorical_cols, drop_first=True, dtype=int
)

# Pemisahan Fitur dan Target
X = df_encoded.drop(columns=["loan_status"])
y = df_encoded["loan_status"]

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling Fitur Numerik
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("[3/5] Handling Class Imbalance menggunakan SMOTE...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

print("[4/5] Melatih model Logistic Regression...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_res, y_train_res)

print("[5/5] Evaluasi Model:")
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# Simpan Artefak Model
joblib.dump(model, "model_credit_risk.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")

print("\n[SUKSES] Model, Scaler, dan Feature Columns berhasil disimpan!")
