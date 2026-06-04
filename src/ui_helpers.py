from __future__ import annotations

from collections import OrderedDict
from typing import Any

import pandas as pd


GENETIC_OPTIONS = [
    "Tidak ada",
    "Ada (ayah/ibu/saudara kandung)",
]

FAMILY_HISTORY_OPTIONS = OrderedDict(
    [
        ("Tidak", 0),
        ("Ya", 1),
    ]
)

PHYSICAL_ACTIVITY_OPTIONS = OrderedDict(
    [
        ("Sedentary / jarang aktivitas fisik", 0),
        ("Moderate / aktivitas sedang", 1),
        ("Active / aktif berolahraga", 2),
    ]
)

DIETARY_HABITS_OPTIONS = OrderedDict(
    [
        ("Unhealthy / kurang sehat", 0),
        ("Moderate / cukup", 1),
        ("Healthy / sehat", 2),
    ]
)

YES_NO_OPTIONS = OrderedDict(
    [
        ("Tidak", 0),
        ("Ya", 1),
    ]
)

SLEEP_OPTIONS = [
    "Kurang dari 5 jam",
    "5-6 jam",
    "7-8 jam (ideal)",
    "Lebih dari 8 jam",
]

STRESS_OPTIONS = [
    "Jarang stres",
    "Kadang-kadang stres",
    "Sering stres",
    "Sangat sering stres",
]

DISPLAY_LABELS = {
    "hba1c": "HbA1c (%)",
    "fasting_blood_sugar": "Gula Darah Puasa (mg/dL)",
    "berat_kg": "Berat Badan (kg)",
    "tinggi_cm": "Tinggi Badan (cm)",
    "usia": "Usia (tahun)",
    "pilihan_genetic": "Risiko Genetik",
    "family_history": "Riwayat Diabetes Keluarga",
    "physical_activity": "Aktivitas Fisik",
    "dietary_habits": "Pola Makan",
    "smoking": "Merokok",
    "alcohol": "Konsumsi Alkohol",
    "pilihan_sleep": "Durasi Tidur",
    "pilihan_stress": "Tingkat Stres",
}

SUMMARY_LABELS = {
    **DISPLAY_LABELS,
    "bmi": "BMI",
}

QUESTION_TITLES = {
    "hba1c": "HbA1c",
    "fasting_blood_sugar": "Gula Darah Puasa",
    "berat_kg": "Berat Badan",
    "tinggi_cm": "Tinggi Badan",
    "usia": "Usia",
    "pilihan_genetic": "Risiko Genetik",
    "family_history": "Riwayat Diabetes Keluarga",
    "physical_activity": "Aktivitas Fisik",
    "dietary_habits": "Pola Makan",
    "smoking": "Merokok",
    "alcohol": "Konsumsi Alkohol",
    "pilihan_sleep": "Durasi Tidur",
    "pilihan_stress": "Tingkat Stres",
}

INPUT_LABELS = {
    "hba1c": "Pilih nilai HbA1c (%)",
    "fasting_blood_sugar": "Pilih nilai gula darah puasa (mg/dL)",
    "berat_kg": "Pilih berat badan (kg)",
    "tinggi_cm": "Pilih tinggi badan (cm)",
    "usia": "Pilih usia (tahun)",
    "pilihan_genetic": "Pilih risiko genetik",
    "family_history": "Pilih riwayat diabetes keluarga",
    "physical_activity": "Pilih aktivitas fisik",
    "dietary_habits": "Pilih pola makan",
    "smoking": "Pilih status merokok",
    "alcohol": "Pilih konsumsi alkohol",
    "pilihan_sleep": "Pilih durasi tidur",
    "pilihan_stress": "Pilih tingkat stres",
}

CAPTIONS = {
    "hba1c": "HbA1c dihitung dalam satuan persen (%). Nilai ini menunjukkan rata-rata kadar gula darah dalam 2–3 bulan terakhir.",
    "fasting_blood_sugar": "Gula darah puasa dihitung dalam satuan mg/dL. Nilai ini menunjukkan kadar gula darah setelah tidak mengonsumsi kalori selama beberapa jam.",
    "berat_kg": "Berat badan diisi dalam satuan kilogram (kg) dan digunakan untuk menghitung BMI.",
    "tinggi_cm": "Tinggi badan diisi dalam satuan centimeter (cm) dan digunakan untuk menghitung BMI.",
    "usia": "Usia diisi dalam satuan tahun.",
    "pilihan_genetic": "Pilih apakah terdapat faktor keturunan atau riwayat diabetes pada keluarga dekat.",
    "family_history": "Pilih apakah terdapat riwayat diabetes dalam keluarga.",
    "physical_activity": "Pilih tingkat aktivitas fisik harian, dari jarang bergerak sampai aktif berolahraga.",
    "dietary_habits": "Pilih gambaran kebiasaan pola makan sehari-hari.",
    "smoking": "Pilih apakah pengguna memiliki kebiasaan merokok.",
    "alcohol": "Pilih apakah pengguna mengonsumsi alkohol.",
    "pilihan_sleep": "Pilih durasi tidur rata-rata per hari.",
    "pilihan_stress": "Pilih frekuensi atau tingkat stres yang dirasakan.",
    "bmi": "BMI dihitung otomatis dalam satuan kg/m² dari berat badan dan tinggi badan.",
}

CAPTIONS.update(
    {
        "hba1c": "HbA1c menggambarkan rata-rata kadar gula darah dalam beberapa bulan terakhir.",
        "fasting_blood_sugar": "Gula darah puasa adalah kadar gula darah setelah tidak mengonsumsi makanan atau minuman berkalori selama beberapa jam.",
        "berat_kg": "Berat badan digunakan bersama tinggi badan untuk menghitung BMI.",
        "tinggi_cm": "Tinggi badan digunakan bersama berat badan untuk menghitung BMI.",
        "bmi": "BMI adalah indeks massa tubuh yang dihitung otomatis dari berat badan dan tinggi badan.",
    }
)

CATEGORY_EXPLANATIONS = {
    "physical_activity": [
        {
            "Kategori": "Sedentary / jarang aktivitas fisik",
            "Deskripsi": "Lebih banyak duduk atau jarang melakukan aktivitas fisik dalam kegiatan sehari-hari.",
        },
        {
            "Kategori": "Moderate / aktivitas sedang",
            "Deskripsi": "Cukup aktif dalam kegiatan sehari-hari dan sesekali melakukan olahraga atau aktivitas fisik.",
        },
        {
            "Kategori": "Active / aktif berolahraga",
            "Deskripsi": "Aktif bergerak dan melakukan olahraga atau aktivitas fisik secara rutin.",
        },
    ],
    "dietary_habits": [
        {
            "Kategori": "Unhealthy / kurang sehat",
            "Deskripsi": "Sering mengonsumsi makanan tinggi gula, tinggi lemak, gorengan, atau makanan cepat saji.",
        },
        {
            "Kategori": "Moderate / cukup",
            "Deskripsi": "Pola makan cukup beragam, tetapi masih sesekali mengonsumsi makanan kurang sehat.",
        },
        {
            "Kategori": "Healthy / sehat",
            "Deskripsi": "Lebih sering mengonsumsi makanan bergizi seimbang seperti sayur, buah, protein, dan membatasi makanan tinggi gula atau lemak.",
        },
    ],
    "pilihan_sleep": [
        {
            "Kategori": "Kurang dari 5 jam",
            "Deskripsi": "Durasi tidur sangat pendek dan tubuh mungkin tidak mendapatkan waktu istirahat yang cukup.",
        },
        {
            "Kategori": "5-6 jam",
            "Deskripsi": "Durasi tidur masih relatif pendek bagi sebagian orang.",
        },
        {
            "Kategori": "7-8 jam (ideal)",
            "Deskripsi": "Durasi tidur yang umumnya cukup untuk mendukung pemulihan tubuh.",
        },
        {
            "Kategori": "Lebih dari 8 jam",
            "Deskripsi": "Durasi tidur lebih panjang dari rata-rata.",
        },
    ],
    "pilihan_stress": [
        {
            "Kategori": "Jarang stres",
            "Deskripsi": "Hampir tidak pernah merasa tertekan atau cemas dalam aktivitas sehari-hari.",
        },
        {
            "Kategori": "Kadang-kadang stres",
            "Deskripsi": "Sesekali merasa stres saat menghadapi tugas, pekerjaan, atau masalah tertentu, tetapi masih dapat mengatasinya dengan baik.",
        },
        {
            "Kategori": "Sering stres",
            "Deskripsi": "Cukup sering merasa tertekan, cemas, atau kewalahan dalam menjalani aktivitas sehari-hari.",
        },
        {
            "Kategori": "Sangat sering stres",
            "Deskripsi": "Hampir setiap hari merasa stres atau tekanan yang cukup berat sehingga memengaruhi aktivitas dan konsentrasi.",
        },
    ],
    "pilihan_genetic": [
        {
            "Kategori": "Tidak ada",
            "Deskripsi": "Tidak diketahui adanya diabetes pada ayah, ibu, atau saudara kandung.",
        },
        {
            "Kategori": "Ada (ayah/ibu/saudara kandung)",
            "Deskripsi": "Terdapat anggota keluarga dekat yang memiliki riwayat diabetes.",
        },
    ],
    "family_history": [
        {
            "Kategori": "Tidak",
            "Deskripsi": "Tidak terdapat riwayat diabetes yang diketahui dalam keluarga.",
        },
        {
            "Kategori": "Ya",
            "Deskripsi": "Terdapat anggota keluarga yang pernah didiagnosis diabetes.",
        },
    ],
    "smoking": [
        {
            "Kategori": "Tidak",
            "Deskripsi": "Tidak memiliki kebiasaan merokok.",
        },
        {
            "Kategori": "Ya",
            "Deskripsi": "Memiliki kebiasaan merokok, baik secara rutin maupun sesekali.",
        },
    ],
    "alcohol": [
        {
            "Kategori": "Tidak",
            "Deskripsi": "Tidak mengonsumsi minuman beralkohol.",
        },
        {
            "Kategori": "Ya",
            "Deskripsi": "Mengonsumsi minuman beralkohol, baik secara rutin maupun sesekali.",
        },
    ],
}

QUESTIONS = [
    {
        "key": "hba1c",
        "title": QUESTION_TITLES["hba1c"],
        "label": INPUT_LABELS["hba1c"],
        "kind": "slider",
        "min": 4.0,
        "max": 10.0,
        "default": 5.7,
        "step": 0.1,
        "format": "%.1f",
    },
    {
        "key": "fasting_blood_sugar",
        "title": QUESTION_TITLES["fasting_blood_sugar"],
        "label": INPUT_LABELS["fasting_blood_sugar"],
        "kind": "slider",
        "min": 70,
        "max": 200,
        "default": 100,
        "step": 1,
    },
    {
        "key": "berat_kg",
        "title": QUESTION_TITLES["berat_kg"],
        "label": INPUT_LABELS["berat_kg"],
        "kind": "slider",
        "min": 30,
        "max": 150,
        "default": 60,
        "step": 1,
    },
    {
        "key": "tinggi_cm",
        "title": QUESTION_TITLES["tinggi_cm"],
        "label": INPUT_LABELS["tinggi_cm"],
        "kind": "slider",
        "min": 120,
        "max": 220,
        "default": 165,
        "step": 1,
    },
    {
        "key": "usia",
        "title": QUESTION_TITLES["usia"],
        "label": INPUT_LABELS["usia"],
        "kind": "slider",
        "min": 15,
        "max": 25,
        "default": 23,
        "step": 1,
    },
    {
        "key": "pilihan_genetic",
        "title": QUESTION_TITLES["pilihan_genetic"],
        "label": INPUT_LABELS["pilihan_genetic"],
        "kind": "select",
        "options": GENETIC_OPTIONS,
        "default": "Tidak ada",
    },
    {
        "key": "family_history",
        "title": QUESTION_TITLES["family_history"],
        "label": INPUT_LABELS["family_history"],
        "kind": "select_map",
        "options": FAMILY_HISTORY_OPTIONS,
        "default": "Tidak",
    },
    {
        "key": "physical_activity",
        "title": QUESTION_TITLES["physical_activity"],
        "label": INPUT_LABELS["physical_activity"],
        "kind": "select_map",
        "options": PHYSICAL_ACTIVITY_OPTIONS,
        "default": "Moderate / aktivitas sedang",
    },
    {
        "key": "dietary_habits",
        "title": QUESTION_TITLES["dietary_habits"],
        "label": INPUT_LABELS["dietary_habits"],
        "kind": "select_map",
        "options": DIETARY_HABITS_OPTIONS,
        "default": "Moderate / cukup",
    },
    {
        "key": "smoking",
        "title": QUESTION_TITLES["smoking"],
        "label": INPUT_LABELS["smoking"],
        "kind": "select_map",
        "options": YES_NO_OPTIONS,
        "default": "Tidak",
    },
    {
        "key": "alcohol",
        "title": QUESTION_TITLES["alcohol"],
        "label": INPUT_LABELS["alcohol"],
        "kind": "select_map",
        "options": YES_NO_OPTIONS,
        "default": "Tidak",
    },
    {
        "key": "pilihan_sleep",
        "title": QUESTION_TITLES["pilihan_sleep"],
        "label": INPUT_LABELS["pilihan_sleep"],
        "kind": "select",
        "options": SLEEP_OPTIONS,
        "default": "7-8 jam (ideal)",
    },
    {
        "key": "pilihan_stress",
        "title": QUESTION_TITLES["pilihan_stress"],
        "label": INPUT_LABELS["pilihan_stress"],
        "kind": "select",
        "options": STRESS_OPTIONS,
        "default": "Kadang-kadang stres",
    },
]


def hitung_bmi(berat_kg: float, tinggi_cm: float) -> tuple[float, str]:
    tinggi_m = tinggi_cm / 100
    bmi = berat_kg / (tinggi_m**2)

    if bmi < 18.5:
        kategori = "Underweight"
    elif bmi < 25:
        kategori = "Normal"
    elif bmi < 30:
        kategori = "Overweight"
    else:
        kategori = "Obesitas"

    return round(bmi, 2), kategori


def get_default_answers() -> dict[str, Any]:
    return {question["key"]: question["default"] for question in QUESTIONS}


def normalize_answer_value(question: dict[str, Any], answer: Any) -> Any:
    if question["kind"] != "select_map":
        return answer
    if answer in question["options"]:
        return question["options"][answer]
    if answer in question["options"].values():
        return answer
    return question["options"][question["default"]]


def make_prediction_payload(answers: dict[str, Any]) -> dict[str, Any]:
    payload = {}
    question_by_key = {question["key"]: question for question in QUESTIONS}
    for key, answer in answers.items():
        payload[key] = normalize_answer_value(question_by_key[key], answer)
    return payload


def label_from_options(options: OrderedDict[str, int], value: Any) -> str:
    if value in options:
        return str(value)

    for label, mapped_value in options.items():
        if value == mapped_value:
            return label

    return str(value)


def make_summary_dataframe(answers: dict[str, Any]) -> pd.DataFrame:
    payload = make_prediction_payload(answers)
    bmi, kategori_bmi = hitung_bmi(payload["berat_kg"], payload["tinggi_cm"])

    summary_rows = [
        {"Fitur": SUMMARY_LABELS["hba1c"], "Nilai": f"{payload['hba1c']:.1f}%"},
        {"Fitur": SUMMARY_LABELS["fasting_blood_sugar"], "Nilai": f"{payload['fasting_blood_sugar']} mg/dL"},
        {"Fitur": SUMMARY_LABELS["berat_kg"], "Nilai": f"{payload['berat_kg']} kg"},
        {"Fitur": SUMMARY_LABELS["tinggi_cm"], "Nilai": f"{payload['tinggi_cm']} cm"},
        {"Fitur": SUMMARY_LABELS["bmi"], "Nilai": f"{bmi:.2f} kg/m² ({kategori_bmi})"},
        {"Fitur": SUMMARY_LABELS["usia"], "Nilai": f"{payload['usia']} tahun"},
        {"Fitur": SUMMARY_LABELS["pilihan_genetic"], "Nilai": answers["pilihan_genetic"]},
        {
            "Fitur": SUMMARY_LABELS["family_history"],
            "Nilai": label_from_options(FAMILY_HISTORY_OPTIONS, answers["family_history"]),
        },
        {
            "Fitur": SUMMARY_LABELS["physical_activity"],
            "Nilai": label_from_options(PHYSICAL_ACTIVITY_OPTIONS, answers["physical_activity"]),
        },
        {
            "Fitur": SUMMARY_LABELS["dietary_habits"],
            "Nilai": label_from_options(DIETARY_HABITS_OPTIONS, answers["dietary_habits"]),
        },
        {"Fitur": SUMMARY_LABELS["smoking"], "Nilai": label_from_options(YES_NO_OPTIONS, answers["smoking"])},
        {"Fitur": SUMMARY_LABELS["alcohol"], "Nilai": label_from_options(YES_NO_OPTIONS, answers["alcohol"])},
        {"Fitur": SUMMARY_LABELS["pilihan_sleep"], "Nilai": answers["pilihan_sleep"]},
        {"Fitur": SUMMARY_LABELS["pilihan_stress"], "Nilai": answers["pilihan_stress"]},
    ]
    return pd.DataFrame(summary_rows, columns=["Fitur", "Nilai"])
