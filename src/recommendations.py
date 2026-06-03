from __future__ import annotations

from pathlib import Path
from typing import Any


ASSET_MAP = {
    "weight_control": "weight_control.png",
    "exercise": "exercise.png",
    "healthy_food": "healthy_food.png",
    "medical_checkup": "medical_checkup.png",
    "no_smoking": "no_smoking.png",
    "sleep": "sleep.png",
    "stress": "stress.png",
}

RECOMMENDATION_RULES = [
    {
        "id": "hba1c_diabetes",
        "asset": "medical_checkup",
        "feature": "HbA1c",
        "condition": lambda data: data["HbA1c"] >= 6.5,
        "text": "HbA1c Anda berada pada rentang indikasi diabetes. Segera konsultasikan hasil ini dengan tenaga kesehatan untuk evaluasi lebih lanjut.",
    },
    {
        "id": "hba1c_prediabetes",
        "asset": "medical_checkup",
        "feature": "HbA1c",
        "condition": lambda data: 5.7 <= data["HbA1c"] < 6.5,
        "text": "HbA1c Anda berada pada rentang prediabetes. Kurangi gula tambahan dan karbohidrat sederhana, lalu pantau HbA1c secara berkala.",
    },
    {
        "id": "bmi_obese",
        "asset": "weight_control",
        "feature": "BMI",
        "condition": lambda data: data["BMI"] >= 30,
        "text": "BMI Anda masuk kategori obesitas. Targetkan penurunan berat badan bertahap dengan pola makan seimbang dan aktivitas fisik rutin.",
    },
    {
        "id": "bmi_overweight",
        "asset": "weight_control",
        "feature": "BMI",
        "condition": lambda data: 25 <= data["BMI"] < 30,
        "text": "BMI Anda masuk kategori overweight. Jaga porsi makan dan mulai rutinitas olahraga ringan untuk mendekati BMI ideal.",
    },
    {
        "id": "fbs_diabetes",
        "asset": "medical_checkup",
        "feature": "Fasting_Blood_Sugar",
        "condition": lambda data: data["Fasting_Blood_Sugar"] >= 126,
        "text": "Gula darah puasa Anda berada pada rentang indikasi diabetes. Lakukan pemeriksaan ulang dan konsultasi medis.",
    },
    {
        "id": "fbs_prediabetes",
        "asset": "medical_checkup",
        "feature": "Fasting_Blood_Sugar",
        "condition": lambda data: 100 <= data["Fasting_Blood_Sugar"] < 126,
        "text": "Gula darah puasa Anda berada pada rentang prediabetes. Batasi minuman manis dan karbohidrat olahan.",
    },
    {
        "id": "family_history",
        "asset": "medical_checkup",
        "feature": "Family_History_Diabetes",
        "condition": lambda data: data["Family_History_Diabetes"] == 1,
        "text": "Ada riwayat diabetes dalam keluarga. Lakukan skrining gula darah rutin setidaknya satu kali per tahun.",
    },
    {
        "id": "activity_sedentary",
        "asset": "exercise",
        "feature": "Physical_Activity_Level",
        "condition": lambda data: data["Physical_Activity_Level"] == 0,
        "text": "Aktivitas fisik Anda masih rendah. Mulai dari jalan kaki 20-30 menit dan tingkatkan perlahan hingga minimal 150 menit per minggu.",
    },
    {
        "id": "activity_moderate",
        "asset": "exercise",
        "feature": "Physical_Activity_Level",
        "condition": lambda data: data["Physical_Activity_Level"] == 1,
        "text": "Aktivitas fisik Anda cukup, tetapi masih bisa ditingkatkan dengan latihan aerobik dan latihan kekuatan secara teratur.",
    },
    {
        "id": "genetic_high",
        "asset": "medical_checkup",
        "feature": "Genetic_Risk_Score",
        "condition": lambda data: data["Genetic_Risk_Score"] >= 8,
        "text": "Risiko genetik Anda tinggi. Perkuat pencegahan dengan pola makan sehat, aktivitas fisik, dan pemeriksaan berkala.",
    },
    {
        "id": "genetic_medium",
        "asset": "medical_checkup",
        "feature": "Genetic_Risk_Score",
        "condition": lambda data: 5 <= data["Genetic_Risk_Score"] < 8,
        "text": "Risiko genetik Anda sedang. Pertahankan gaya hidup sehat untuk menekan risiko di masa depan.",
    },
    {
        "id": "diet_unhealthy",
        "asset": "healthy_food",
        "feature": "Dietary_Habits",
        "condition": lambda data: data["Dietary_Habits"] == 0,
        "text": "Pola makan Anda kurang sehat. Perbanyak sayur, buah, protein tanpa lemak, dan kurangi makanan tinggi gula atau lemak jenuh.",
    },
    {
        "id": "diet_moderate",
        "asset": "healthy_food",
        "feature": "Dietary_Habits",
        "condition": lambda data: data["Dietary_Habits"] == 1,
        "text": "Pola makan Anda cukup. Tingkatkan konsistensi menu bergizi dan batasi camilan tinggi gula.",
    },
    {
        "id": "stress_high",
        "asset": "stress",
        "feature": "Stress_Level",
        "condition": lambda data: data["Stress_Level"] >= 8,
        "text": "Tingkat stres Anda tinggi. Coba jadwalkan relaksasi harian, olahraga ringan, tidur cukup, atau konseling bila diperlukan.",
    },
    {
        "id": "stress_medium",
        "asset": "stress",
        "feature": "Stress_Level",
        "condition": lambda data: 4 <= data["Stress_Level"] < 8,
        "text": "Stres Anda berada pada level sedang. Kelola dengan jeda istirahat, hobi, pernapasan dalam, atau jalan santai.",
    },
    {
        "id": "sleep_low",
        "asset": "sleep",
        "feature": "Sleep_Hours",
        "condition": lambda data: data["Sleep_Hours"] < 6,
        "text": "Durasi tidur Anda kurang. Targetkan 7-9 jam per malam dan kurangi layar sebelum tidur.",
    },
    {
        "id": "smoking",
        "asset": "no_smoking",
        "feature": "Smoking",
        "condition": lambda data: data["Smoking"] == 1,
        "text": "Anda merokok. Mengurangi hingga berhenti merokok dapat membantu menurunkan risiko diabetes tipe 2 dan penyakit kardiovaskular.",
    },
    {
        "id": "alcohol",
        "asset": "medical_checkup",
        "feature": "Alcohol_Consumption",
        "condition": lambda data: data["Alcohol_Consumption"] == 1,
        "text": "Anda mengonsumsi alkohol. Batasi konsumsi alkohol karena dapat mengganggu metabolisme glukosa dan kesehatan hati.",
    },
]


def resolve_asset_path(asset_key: str, assets_dir: str | Path = "assets") -> Path | None:
    filename = ASSET_MAP.get(asset_key)
    if not filename:
        return None
    path = Path(assets_dir) / filename
    return path if path.exists() else None


def generate_recommendations(
    data: dict[str, Any],
    risk_label: str,
    importance_dict: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    importance_dict = importance_dict or {}
    recommendations = []

    for rule in RECOMMENDATION_RULES:
        if rule["condition"](data):
            recommendations.append(
                {
                    "id": rule["id"],
                    "text": rule["text"],
                    "asset": rule["asset"],
                    "feature": rule["feature"],
                    "importance": float(importance_dict.get(rule["feature"], 0)),
                }
            )

    recommendations.sort(key=lambda item: item["importance"], reverse=True)

    if risk_label == "Tinggi":
        text = "Prioritas utama: segera konsultasikan kondisi Anda ke dokter atau ahli gizi untuk evaluasi menyeluruh."
    elif risk_label == "Sedang":
        text = "Lakukan pemeriksaan kesehatan berkala setiap 6 bulan untuk memantau perkembangan kondisi Anda."
    else:
        text = "Pertahankan gaya hidup sehat dan lakukan pemeriksaan tahunan sebagai deteksi dini."

    recommendations.append(
        {
            "id": "general",
            "text": text,
            "asset": "medical_checkup",
            "feature": "Risk_Level",
            "importance": 0.0,
        }
    )

    return recommendations
