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

HELP_TEXT = {
    "hba1c": "HbA1c menunjukkan rata-rata kadar gula darah dalam 2-3 bulan terakhir.",
    "fasting_blood_sugar": "Gula darah puasa adalah kadar gula darah setelah tidak mengonsumsi kalori selama beberapa jam.",
    "berat_kg": "Berat badan digunakan bersama tinggi badan untuk menghitung BMI.",
    "tinggi_cm": "Tinggi badan digunakan bersama berat badan untuk menghitung BMI.",
    "usia": "Usia pengguna dalam tahun.",
    "pilihan_genetic": "Riwayat genetik atau keturunan dapat berkaitan dengan risiko diabetes.",
    "family_history": "Apakah ada riwayat diabetes dalam keluarga?",
    "physical_activity": "Tingkat aktivitas fisik harian.",
    "dietary_habits": "Gambaran kebiasaan pola makan sehari-hari.",
    "smoking": "Status kebiasaan merokok.",
    "alcohol": "Status konsumsi alkohol.",
    "pilihan_sleep": "Durasi tidur rata-rata per hari.",
    "pilihan_stress": "Frekuensi atau tingkat stres yang dirasakan.",
}

QUESTIONS = [
    {
        "key": "hba1c",
        "label": "HbA1c",
        "kind": "slider",
        "min": 4.0,
        "max": 10.0,
        "default": 5.7,
        "step": 0.1,
        "format": "%.1f",
    },
    {
        "key": "fasting_blood_sugar",
        "label": "Fasting Blood Sugar",
        "kind": "slider",
        "min": 70,
        "max": 200,
        "default": 100,
        "step": 1,
    },
    {
        "key": "berat_kg",
        "label": "Berat Badan",
        "kind": "slider",
        "min": 30,
        "max": 150,
        "default": 60,
        "step": 1,
    },
    {
        "key": "tinggi_cm",
        "label": "Tinggi Badan",
        "kind": "slider",
        "min": 120,
        "max": 220,
        "default": 165,
        "step": 1,
    },
    {
        "key": "usia",
        "label": "Usia",
        "kind": "slider",
        "min": 15,
        "max": 25,
        "default": 23,
        "step": 1,
    },
    {
        "key": "pilihan_genetic",
        "label": "Genetic Risk",
        "kind": "select",
        "options": GENETIC_OPTIONS,
        "default": "Tidak ada",
    },
    {
        "key": "family_history",
        "label": "Family History",
        "kind": "select_map",
        "options": FAMILY_HISTORY_OPTIONS,
        "default": "Tidak",
    },
    {
        "key": "physical_activity",
        "label": "Physical Activity",
        "kind": "select_map",
        "options": PHYSICAL_ACTIVITY_OPTIONS,
        "default": "Moderate / aktivitas sedang",
    },
    {
        "key": "dietary_habits",
        "label": "Dietary Habits",
        "kind": "select_map",
        "options": DIETARY_HABITS_OPTIONS,
        "default": "Moderate / cukup",
    },
    {
        "key": "smoking",
        "label": "Smoking",
        "kind": "select_map",
        "options": YES_NO_OPTIONS,
        "default": "Tidak",
    },
    {
        "key": "alcohol",
        "label": "Alcohol",
        "kind": "select_map",
        "options": YES_NO_OPTIONS,
        "default": "Tidak",
    },
    {
        "key": "pilihan_sleep",
        "label": "Sleep",
        "kind": "select",
        "options": SLEEP_OPTIONS,
        "default": "7-8 jam (ideal)",
    },
    {
        "key": "pilihan_stress",
        "label": "Stress",
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
    return question["options"][answer]


def make_prediction_payload(answers: dict[str, Any]) -> dict[str, Any]:
    payload = {}
    question_by_key = {question["key"]: question for question in QUESTIONS}
    for key, answer in answers.items():
        payload[key] = normalize_answer_value(question_by_key[key], answer)
    return payload


def make_summary_dataframe(answers: dict[str, Any]) -> pd.DataFrame:
    payload = make_prediction_payload(answers)
    bmi, kategori_bmi = hitung_bmi(payload["berat_kg"], payload["tinggi_cm"])

    rows = [
        ("HbA1c", f"{payload['hba1c']:.1f}%"),
        ("Fasting Blood Sugar", f"{payload['fasting_blood_sugar']} mg/dL"),
        ("Berat Badan", f"{payload['berat_kg']} kg"),
        ("Tinggi Badan", f"{payload['tinggi_cm']} cm"),
        ("BMI", f"{bmi} ({kategori_bmi})"),
        ("Usia", f"{payload['usia']} tahun"),
        ("Genetic Risk", answers["pilihan_genetic"]),
        ("Family History", answers["family_history"]),
        ("Physical Activity", answers["physical_activity"]),
        ("Dietary Habits", answers["dietary_habits"]),
        ("Smoking", answers["smoking"]),
        ("Alcohol", answers["alcohol"]),
        ("Sleep", answers["pilihan_sleep"]),
        ("Stress", answers["pilihan_stress"]),
    ]
    return pd.DataFrame(rows, columns=["Input", "Nilai"])
