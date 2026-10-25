# 📊 Credit Risk Analytics 

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Flask-green.svg)
![Machine Learning](https://img.shields.io/badge/ML-Logistic%20Regression-orange.svg)
![UI Design](https://img.shields.io/badge/UI-Google%20Studio%20Theme-sky.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

Aplikasi web end-to-end berbasis Machine Learning untuk **Prediksi Kelayakan & Risiko Gagal Bayar (Default) Pinjaman P2P Lending** menggunakan algoritma **Logistic Regression** dan teknik penyeimbangan data **SMOTE**.

Proyek ini disusun untuk memenuhi **Tugas Akhir Modul 03 (TA-03) - Algoritma Logistic Regression** pada mata kuliah Pembelajaran Mesin.

---

## 🎯 Latar Belakang & Tujuan
Pada industri keuangan digital (*Peer-to-Peer Lending* / *FinTech*), risiko kerugian terbesar berasal dari nasabah yang mengalami gagal bayar (*credit default*). Proyek ini bertujuan untuk:
1. Membangun model pemfaktoran risiko kualitatif & kuantitatif berdasarkan riwayat finansial pemohon.
2. Mengatasi permasalahan *class imbalance* pada data transaksi keuangan.
3. Mengembangkan alat bantu keputusan (*decision support tool*) interaktif berbasis web untuk analis kredit secara real-time.

---

## ⚙️ Ringkasan Performa Model ML

* **Algoritma Utama**: Logistic Regression (`sklearn.linear_model.LogisticRegression`)
* **Handling Class Imbalance**: SMOTE (*Synthetic Minority Over-sampling Technique*)
* **Scaling Fitur**: `StandardScaler`
* **Skor ROC-AUC**: **0.8766 (87.66%)** — *Excellent Classification Performance*
* **Recall Kelas Default (1)**: **78%** — *Berhasil meminimalkan False Negative risiko gagal bayar*

### 📊 Confusion Matrix & Metrics Summary
| Metrik Evaluasi | Kelas 0 (Lancar) | Kelas 1 (Default / Riskan) | Overall |
| :--- | :---: | :---: | :---: |
| **Precision** | 0.93 | 0.54 | - |
| **Recall** | 0.82 | **0.78** | - |
| **F1-Score** | 0.87 | 0.64 | - |
| **Accuracy** | - | - | **81%** |

---

## 💻 Fitur Utama Aplikasi Web
* **Google Studio Theme**: Antarmuka responsif Dark Mode yang bersih, modern, dan intuitif menggunakan Tailwind CSS & Lucide Icons.
* **Real-time Scoring**: Menghasilkan status kelayakan kredit ("LANCAR" vs "BERISIKO GAGAL BAYAR") beserta persentase probabilitas default.
* **Comprehensive Metrics Input**: Memproses 11 variabel demografis & finansial meliputi Usia, Pendapatan, Status Kepemilikan Rumah, Grade Kredit, Suku Bunga, hingga Histori Wanprestasi.

---

## 🗂️ Struktur Direktori Proyek

```text
ta03-credit-risk/
│
├── credit_risk_dataset.csv     # Dataset Historis Kredit (Kaggle/UCI)
├── train_model.py              # Script EDA, Cleaning, SMOTE & Training Model
├── app.py                      # Server Aplikasi Web Flask
├── model_credit_risk.pkl       # Artefak Model Terlatih (Logistic Regression)
├── scaler.pkl                  # Artefak StandardScaler
├── feature_columns.pkl         # Daftar Kolom Fitur Encoded
├── requirements.txt            # Dependensi Package Python
├── README.md                   # Dokumentasi Resmi Proyek
│
└── templates/
    └── index.html              # Interface Dashboard UI (Google Studio Styling)
```

---

## 🚀 Panduan Instalasi & Jalankan Secara Lokal

### 1. Clone Repositori
```bash
git clone https://github.com/USERNAME_KAMU/ta03-credit-risk.git
cd ta03-credit-risk
```

### 2. Buat & Aktifkan Virtual Environment
* **Windows (PowerShell)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **Mac/Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Instal Dependensi
```bash
pip install -r requirements.txt
```

### 4. Pelatihan Model (Opsional)
Untuk melatih ulang model dan menghasilkan artefak `.pkl` baru:
```bash
python train_model.py
```

### 5. Jalankan Aplikasi Flask
```bash
python app.py
```
Buka browser favoritmu dan akses: **`http://127.0.0.1:5000`**

---

## 🛠️ Tools & Teknologi
* **Bahasa Pemrograman**: Python 3.11+
* **Data Processing & ML**: Pandas, NumPy, Scikit-Learn, Imbalanced-Learn (SMOTE)
* **Web Framework**: Flask
* **Frontend**: HTML5, Tailwind CSS, Lucide Icons
* **Model Persistence**: Joblib

---

## 📝 Lisensi
Proyek ini dibuat untuk keperluan akademik praktikum Pembelajaran Mesin. Bebas dikembangkan kembali untuk tujuan edukasi.
