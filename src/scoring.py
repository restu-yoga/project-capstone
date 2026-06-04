from __future__ import annotations

from collections.abc import Mapping
from typing import Any


RISK_LABELS = {
    0: "Tidak Berisiko",
    1: "Sedang",
    2: "Tinggi",
}

RISK_LABEL_TO_CODE = {label: code for code, label in RISK_LABELS.items()}

SCORING_THRESHOLDS = {
    "tidak_berisiko_max": 7,
    "sedang_min": 8,
    "sedang_max": 13,
    "tinggi_min": 14,
}


def _get_value(data: Mapping[str, Any], *keys: str, default: Any = 0) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return default


def _is_yes(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"yes", "ya", "true", "1"}
    return value == 1 or value is True


def _activity_level(value: Any) -> int:
    if isinstance(value, str):
        mapping = {"sedentary": 0, "moderate": 1, "active": 2}
        return mapping.get(value.strip().split("/")[0].strip().lower(), 2)
    return int(value)


def _dietary_level(value: Any) -> int:
    if isinstance(value, str):
        mapping = {"unhealthy": 0, "moderate": 1, "healthy": 2}
        return mapping.get(value.strip().split("/")[0].strip().lower(), 2)
    return int(value)


def create_risk_level(score: int | float) -> str:
    if score >= 14:
        return "Tinggi"
    if score >= 8:
        return "Sedang"
    return "Tidak Berisiko"


def calculate_risk_score(data: Mapping[str, Any] | None = None, **kwargs: Any) -> int:
    values: Mapping[str, Any] = data if data is not None else kwargs

    genetic_risk_score = float(_get_value(values, "Genetic_Risk_Score", "genetic_risk_score"))
    family_history = _get_value(values, "Family_History_Diabetes", "family_history")
    bmi = float(_get_value(values, "BMI", "bmi"))
    hba1c = float(_get_value(values, "HbA1c", "hba1c"))
    fasting_blood_sugar = float(_get_value(values, "Fasting_Blood_Sugar", "fasting_blood_sugar"))
    physical_activity = _activity_level(_get_value(values, "Physical_Activity_Level", "physical_activity"))
    dietary_habits = _dietary_level(_get_value(values, "Dietary_Habits", "dietary_habits"))
    smoking = _get_value(values, "Smoking", "smoking")
    alcohol = _get_value(values, "Alcohol_Consumption", "alcohol")
    stress_level = float(_get_value(values, "Stress_Level", "stress_level"))
    sleep_hours = float(_get_value(values, "Sleep_Hours", "sleep_hours"))
    age = int(_get_value(values, "Age", "usia", "age"))

    score = 0

    if genetic_risk_score >= 8:
        score += 3
    elif genetic_risk_score >= 5:
        score += 2

    if _is_yes(family_history):
        score += 2

    if bmi >= 30:
        score += 3
    elif bmi >= 25:
        score += 2

    if hba1c >= 6.5:
        score += 3
    elif hba1c >= 5.7:
        score += 2

    if fasting_blood_sugar >= 126:
        score += 3
    elif fasting_blood_sugar >= 100:
        score += 2

    if physical_activity == 0:
        score += 2
    elif physical_activity == 1:
        score += 1

    if dietary_habits == 0:
        score += 2
    elif dietary_habits == 1:
        score += 1

    if _is_yes(smoking):
        score += 1
    if _is_yes(alcohol):
        score += 1
    if stress_level >= 8:
        score += 1
    if sleep_hours <= 5:
        score += 1
    if age >= 22:
        score += 1

    return score
