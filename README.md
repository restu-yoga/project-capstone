# Prediksi Risiko Diabetes Berbasis Machine Learning

Aplikasi ini menggunakan Streamlit untuk memprediksi tingkat risiko diabetes pada young adults berdasarkan faktor klinis, gaya hidup, dan model machine learning XGBoost yang sudah dilatih.

## Fitur Aplikasi

- Pengisian data bertahap dengan 13 pertanyaan.
- Fitur klinis memiliki nilai default jika pengguna tidak mengetahui nilai pastinya.
- Perhitungan skor risiko manual berbasis sistem scoring.
- Prediksi tingkat risiko diabetes menggunakan model `.pkl` yang dilatih untuk mempelajari kategori scoring.
- Perhitungan BMI otomatis dari berat badan dan tinggi badan.
- Gauge estimasi tingkat risiko diabetes dari probabilitas model untuk kelas berisiko.
- Skor risiko dan keyakinan model disimpan di detail teknis.
- Detail probabilitas per kelas disimpan di expander teknis.
- Rekomendasi personal berdasarkan hasil prediksi dan faktor risiko pengguna.
- Tampilan gambar rekomendasi dari folder `assets/`.
- Validasi path model, dataset, dan asset dari root project.

## Sistem Scoring Risiko

Kategori risiko utama ditentukan dari skor manual berbasis faktor genetik, klinis, dan gaya hidup. Threshold yang digunakan:

| Skor Total | Kategori |
|---:|---|
| < 8 | Tidak Berisiko |
| 8-13 | Sedang |
| >= 14 | Tinggi |

Model machine learning dilatih menggunakan label `Risk_Level` yang dibuat dari sistem scoring tersebut. Kolom `ID`, `Prediabetes`, `Diabetes_Type`, `Risk_Score`, dan `Risk_Level` tidak digunakan sebagai fitur input model untuk menghindari data leakage.

Gauge **Estimasi Tingkat Risiko Diabetes** dihitung dari probabilitas model untuk kelas `Sedang` dan `Tinggi`. Nilai ini adalah estimasi model berdasarkan data yang dimasukkan, bukan diagnosis medis.

Skor risiko, kategori scoring, keyakinan model, dan probabilitas per kelas tetap tersedia di bagian `Detail teknis model` agar tidak mengganggu tampilan utama untuk pengguna awam.

## Struktur Folder

```text
PROJECT-CAPSTONE/
|-- app.py
|-- data/
|   `-- diabetes_young_adults_india.csv
|-- models/
|   `-- model_xgboost_risk_level.pkl
|-- notebooks/
|   |-- Inferensi_Diabetes_Rekomendasi.ipynb
|   `-- Prediksi_Resiko_Diabetes_XGBoost.ipynb
|-- src/
|   |-- __init__.py
|   |-- inference.py
|   |-- recommendations.py
|   `-- ui_helpers.py
|-- assets/
|   |-- exercise.png
|   |-- healthy_food.png
|   |-- medical_checkup.png
|   |-- no_smoking.png
|   |-- sleep.png
|   |-- stress.png
|   `-- weight_control.png
|-- requirements.txt
|-- README.md
`-- .gitignore
```

## Prasyarat

- Python 3.10 atau 3.11 disarankan.
- VS Code opsional.
- Terminal dijalankan dari root project, yaitu folder `PROJECT-CAPSTONE/`.

## Cara Clone atau Download Project

Jika menggunakan Git:

```bash
git clone <url-repository>
cd PROJECT-CAPSTONE
```

Jika project didapat dari ZIP, ekstrak file ZIP lalu buka terminal di folder hasil ekstrak.

## Membuat Virtual Environment di Windows

```powershell
python -m venv .venv
```

## Aktivasi Virtual Environment di Windows

Command Prompt:

```cmd
.venv\Scripts\activate
```

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Jika PowerShell menolak aktivasi script, jalankan:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Lalu aktifkan kembali virtual environment.

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Pastikan File Penting Ada

Sebelum menjalankan aplikasi, pastikan file berikut berada di lokasi yang benar:

- `data/diabetes_young_adults_india.csv`
- `models/model_xgboost_risk_level.pkl`
- `assets/exercise.png`
- `assets/healthy_food.png`
- `assets/medical_checkup.png`
- `assets/no_smoking.png`
- `assets/sleep.png`
- `assets/stress.png`
- `assets/weight_control.png`

Jika file model terlalu besar untuk diupload ke GitHub, letakkan file model secara manual di folder `models/` setelah clone/download project.

## Menjalankan Aplikasi Streamlit

Jalankan dari root project:

```bash
streamlit run app.py
```

Aplikasi akan membuka browser lokal. Isi pertanyaan satu per satu, cek ringkasan input, lalu klik tombol `Prediksi`.

Jika pengguna tidak mengetahui nilai klinis seperti HbA1c atau gula darah puasa, gunakan nilai default yang sudah disediakan aplikasi. Nilai tersebut hanya membantu proses estimasi dan bukan pengganti pemeriksaan medis.

BMI tidak diisi manual oleh pengguna. BMI dihitung otomatis menggunakan:

```text
BMI = berat_kg / ((tinggi_cm / 100) ** 2)
```

## Menjalankan Notebook Training

Notebook training tersedia di:

```text
notebooks/Prediksi_Resiko_Diabetes_XGBoost.ipynb
```

Jalankan notebook ini jika ingin melatih ulang model. Pastikan dataset tersedia di folder `data/`, lalu simpan model hasil training ke:

```text
models/model_xgboost_risk_level.pkl
```

Notebook training membuat `Risk_Score` dan `Risk_Level` dari aturan scoring revisi, lalu melatih XGBoost dengan 12 fitur final tanpa kolom leakage.

## Menjalankan Notebook Inferensi

Notebook inferensi tersedia di:

```text
notebooks/Inferensi_Diabetes_Rekomendasi.ipynb
```

Notebook tersebut menjadi referensi logic inferensi dan rekomendasi. Versi modular untuk aplikasi Streamlit berada di folder `src/`.

## Menjalankan Test Scoring

Test ringan untuk validasi scoring dapat dijalankan dari root project:

```bash
python -m unittest discover -s tests
```

Test mencakup contoh skor 7 yang harus menghasilkan kategori `Tidak Berisiko`.

## Troubleshooting

`ModuleNotFoundError`
: Pastikan virtual environment aktif dan dependencies sudah diinstall dengan `pip install -r requirements.txt`.

`FileNotFoundError` model
: Pastikan file `models/model_xgboost_risk_level.pkl` tersedia dan terminal berada di root project.

`FileNotFoundError` dataset
: Pastikan file `data/diabetes_young_adults_india.csv` tersedia.

`FileNotFoundError` asset
: Aplikasi tetap berjalan tanpa gambar, tetapi rekomendasi hanya tampil sebagai teks. Pastikan semua PNG ada di folder `assets/`.

Error XGBoost belum terinstall
: Jalankan `pip install xgboost` atau ulangi `pip install -r requirements.txt`.

Terminal dari folder yang salah
: Pindah ke root project sebelum menjalankan `streamlit run app.py`.

PowerShell tidak bisa activate venv
: Jalankan `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, lalu aktifkan ulang `.venv`.

## Disclaimer Medis

Aplikasi ini hanya memberikan estimasi risiko berbasis data dan sistem scoring. Persentase pada gauge adalah estimasi model, bukan diagnosis medis. Pemeriksaan dan diagnosis diabetes tetap harus dilakukan oleh tenaga kesehatan.
