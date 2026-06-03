from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.recommendations import generate_recommendations
from src.ui_helpers import hitung_bmi


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "model_xgboost_risk_level.pkl"
DEFAULT_DATASET_PATH = ROOT_DIR / "data" / "diabetes_young_adults_india.csv"

DEFAULT_FEATURE_COLUMNS = [
    "Age",
    "BMI",
    "HbA1c",
    "Fasting_Blood_Sugar",
    "Genetic_Risk_Score",
    "Family_History_Diabetes",
    "Physical_Activity_Level",
    "Dietary_Habits",
    "Smoking",
    "Alcohol_Consumption",
    "Sleep_Hours",
    "Stress_Level",
]

DEFAULT_LABEL_MAP = {
    0: "Tidak Berisiko",
    1: "Sedang",
    2: "Tinggi",
}


def load_dataset(dataset_path: str | Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {dataset_path}")
    return pd.read_csv(dataset_path)


def load_model_artifact(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

    artifact = joblib.load(model_path)
    if isinstance(artifact, dict):
        model = artifact.get("model")
        if model is None:
            raise ValueError("Artifact model berbentuk dictionary tetapi key 'model' tidak tersedia.")
        metadata = dict(artifact)
    else:
        model = artifact
        metadata = {}

    feature_columns = metadata.get("feature_columns")
    if feature_columns is None:
        feature_columns = list(getattr(model, "feature_names_in_", DEFAULT_FEATURE_COLUMNS))

    target_classes = metadata.get("target_classes") or metadata.get("class_names")
    label_encoder = metadata.get("label_encoder")

    return {
        "raw_artifact": artifact,
        "model": model,
        "feature_columns": list(feature_columns),
        "target_classes": target_classes,
        "class_names": metadata.get("class_names"),
        "label_encoder": label_encoder,
        "final_model_name": metadata.get("final_model_name", model.__class__.__name__),
        "feature_importance": metadata.get("feature_importance"),
        "feature_importances": metadata.get("feature_importances"),
    }


def mapping_genetic(pilihan: str) -> int:
    mapping = {
        "Tidak ada": 2,
        "Tidak ada keluarga diabetes": 2,
        "Ada (paman/bibi/kakek/nenek)": 5,
        "Ada (ayah/ibu/saudara kandung)": 9,
    }
    return mapping[pilihan]


def mapping_sleep(pilihan: str) -> float:
    mapping = {
        "Kurang dari 5 jam": 4.5,
        "5-6 jam": 5.5,
        "7-8 jam (ideal)": 7.5,
        "Lebih dari 8 jam": 9.0,
    }
    return mapping[pilihan]


def mapping_stress(pilihan: str) -> int:
    mapping = {
        "Jarang stres": 2,
        "Hampir tidak pernah stres": 2,
        "Kadang-kadang stres": 5,
        "Sering stres": 8,
        "Sering stres / sulit dikendalikan": 8,
        "Sangat sering stres": 10,
    }
    return mapping[pilihan]


def build_model_input(
    usia: int,
    berat_kg: float,
    tinggi_cm: float,
    hba1c: float,
    fasting_blood_sugar: float,
    pilihan_genetic: str,
    family_history: int,
    physical_activity: int,
    dietary_habits: int,
    smoking: int,
    alcohol: int,
    pilihan_sleep: str,
    pilihan_stress: str,
) -> tuple[dict[str, Any], float, str]:
    bmi, kategori_bmi = hitung_bmi(berat_kg, tinggi_cm)
    data = {
        "Age": usia,
        "BMI": bmi,
        "HbA1c": hba1c,
        "Fasting_Blood_Sugar": fasting_blood_sugar,
        "Genetic_Risk_Score": mapping_genetic(pilihan_genetic),
        "Family_History_Diabetes": family_history,
        "Physical_Activity_Level": physical_activity,
        "Dietary_Habits": dietary_habits,
        "Smoking": smoking,
        "Alcohol_Consumption": alcohol,
        "Sleep_Hours": mapping_sleep(pilihan_sleep),
        "Stress_Level": mapping_stress(pilihan_stress),
    }
    return data, bmi, kategori_bmi


def get_feature_importance(artifact: dict[str, Any]) -> dict[str, float]:
    feature_columns = artifact["feature_columns"]
    metadata_importance = artifact.get("feature_importance") or artifact.get("feature_importances")

    if isinstance(metadata_importance, dict):
        return {str(key): float(value) for key, value in metadata_importance.items()}

    if metadata_importance is not None:
        return {
            feature: float(score)
            for feature, score in zip(feature_columns, metadata_importance)
        }

    model = artifact["model"]
    model_importance = getattr(model, "feature_importances_", None)
    if model_importance is None:
        return {feature: 0.0 for feature in feature_columns}

    return {
        feature: float(score)
        for feature, score in zip(feature_columns, model_importance)
    }


def decode_prediction(prediction: Any, artifact: dict[str, Any]) -> str:
    label_encoder = artifact.get("label_encoder")
    if label_encoder is not None and hasattr(label_encoder, "inverse_transform"):
        return str(label_encoder.inverse_transform([prediction])[0])

    target_classes = artifact.get("target_classes") or artifact.get("class_names")
    if target_classes is not None:
        try:
            if isinstance(target_classes, dict):
                return str(target_classes[prediction])
            return str(target_classes[int(prediction)])
        except (KeyError, IndexError, ValueError, TypeError):
            pass

    if prediction in DEFAULT_LABEL_MAP:
        return DEFAULT_LABEL_MAP[prediction]

    try:
        return DEFAULT_LABEL_MAP.get(int(prediction), str(prediction))
    except (TypeError, ValueError):
        return str(prediction)


def format_probabilities(probabilities: Any, artifact: dict[str, Any]) -> dict[str, float]:
    if probabilities is None:
        return {}

    classes = list(getattr(artifact["model"], "classes_", range(len(probabilities))))
    formatted = {}
    for class_value, probability in zip(classes, probabilities):
        label = decode_prediction(class_value, artifact)
        formatted[label] = round(float(probability), 4)
    return formatted


def calculate_risk_pct(probability_map: dict[str, float]) -> float | None:
    if not probability_map:
        return None
    sedang = probability_map.get("Sedang", 0.0)
    tinggi = probability_map.get("Tinggi", 0.0)
    return round((sedang * 0.5 + tinggi) * 100, 1)


def prediksi_diabetes(
    usia: int,
    berat_kg: float,
    tinggi_cm: float,
    hba1c: float,
    fasting_blood_sugar: float,
    pilihan_genetic: str,
    family_history: int,
    physical_activity: int,
    dietary_habits: int,
    smoking: int,
    alcohol: int,
    pilihan_sleep: str,
    pilihan_stress: str,
    model_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact = model_artifact or load_model_artifact()
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    data, bmi, kategori_bmi = build_model_input(
        usia=usia,
        berat_kg=berat_kg,
        tinggi_cm=tinggi_cm,
        hba1c=hba1c,
        fasting_blood_sugar=fasting_blood_sugar,
        pilihan_genetic=pilihan_genetic,
        family_history=family_history,
        physical_activity=physical_activity,
        dietary_habits=dietary_habits,
        smoking=smoking,
        alcohol=alcohol,
        pilihan_sleep=pilihan_sleep,
        pilihan_stress=pilihan_stress,
    )

    df_input = pd.DataFrame([data])
    missing_features = [feature for feature in feature_columns if feature not in df_input.columns]
    if missing_features:
        raise ValueError(f"Fitur model belum tersedia di input: {', '.join(missing_features)}")
    df_input = df_input[feature_columns]

    prediction = model.predict(df_input)[0]
    risk_label = decode_prediction(prediction, artifact)

    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(df_input)[0]
    probability_map = format_probabilities(probabilities, artifact)
    risk_pct = calculate_risk_pct(probability_map)
    importance_dict = get_feature_importance(artifact)
    recommendations = generate_recommendations(data, risk_label, importance_dict)

    return {
        "risk_label": risk_label,
        "risk_pct": risk_pct,
        "bmi": bmi,
        "kategori_bmi": kategori_bmi,
        "proba": probability_map,
        "recommendations": recommendations,
        "faktor_risiko": data,
        "feature_importance": importance_dict,
        "model_name": artifact.get("final_model_name", model.__class__.__name__),
    }
