from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, load_dataset, load_model_artifact, prediksi_diabetes
from src.recommendations import resolve_asset_path
from src.ui_helpers import (
    HELP_TEXT,
    QUESTIONS,
    get_default_answers,
    make_prediction_payload,
    make_summary_dataframe,
)


ASSETS_DIR = Path("assets")
DISCLAIMER = (
    "Aplikasi ini hanya memberikan estimasi risiko berbasis data dan sistem scoring. "
    "Hasil prediksi bukan diagnosis medis. Pemeriksaan dan diagnosis diabetes tetap harus dilakukan oleh tenaga kesehatan."
)


st.set_page_config(
    page_title="Prediksi Risiko Diabetes",
    page_icon="D",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def get_model_artifact():
    return load_model_artifact(DEFAULT_MODEL_PATH)


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    return load_dataset(DEFAULT_DATASET_PATH)


def init_session_state() -> None:
    if "step" not in st.session_state:
        st.session_state.step = 0
    if "answers" not in st.session_state:
        st.session_state.answers = get_default_answers()
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None


def reset_wizard() -> None:
    st.session_state.step = 0
    st.session_state.answers = get_default_answers()
    st.session_state.prediction_result = None


def render_question(question: dict) -> None:
    key = question["key"]
    widget_key = f"input_{key}"
    current_value = st.session_state.answers.get(key, question["default"])

    if question["kind"] == "slider":
        value = st.slider(
            question["label"],
            min_value=question["min"],
            max_value=question["max"],
            value=current_value,
            step=question["step"],
            format=question.get("format"),
            help=HELP_TEXT[key],
            key=widget_key,
        )
    elif question["kind"] == "select_map":
        options = list(question["options"].keys())
        value = st.selectbox(
            question["label"],
            options=options,
            index=options.index(current_value) if current_value in options else 0,
            help=HELP_TEXT[key],
            key=widget_key,
        )
    else:
        options = question["options"]
        value = st.selectbox(
            question["label"],
            options=options,
            index=options.index(current_value) if current_value in options else 0,
            help=HELP_TEXT[key],
            key=widget_key,
        )

    st.session_state.answers[key] = value


def render_navigation(is_summary: bool = False) -> None:
    back_col, reset_col, next_col = st.columns([1, 1, 1])

    with back_col:
        if st.button("Kembali", disabled=st.session_state.step == 0, use_container_width=True):
            st.session_state.prediction_result = None
            st.session_state.step = max(0, st.session_state.step - 1)
            st.rerun()

    with reset_col:
        if st.button("Reset", use_container_width=True):
            reset_wizard()
            st.rerun()

    with next_col:
        if not is_summary and st.button("Lanjut", use_container_width=True):
            st.session_state.prediction_result = None
            st.session_state.step = min(len(QUESTIONS), st.session_state.step + 1)
            st.rerun()


def render_probabilities(probabilities: dict[str, float]) -> None:
    if not probabilities:
        st.info("Model tidak menyediakan `predict_proba()`, sehingga probabilitas per kelas tidak ditampilkan.")
        return

    probability_df = pd.DataFrame(
        [{"Kelas": label, "Probabilitas": probability, "Persentase": f"{probability * 100:.2f}%"} for label, probability in probabilities.items()]
    )
    st.dataframe(probability_df, hide_index=True, use_container_width=True)


def render_recommendations(recommendations: list[dict]) -> None:
    st.subheader("Rekomendasi Personal")
    for recommendation in recommendations:
        image_path = resolve_asset_path(recommendation.get("asset", ""), ASSETS_DIR)
        text_col, image_col = st.columns([2, 1])
        with text_col:
            st.write(recommendation["text"])
            feature = recommendation.get("feature")
            importance = recommendation.get("importance", 0)
            if feature and importance:
                st.caption(f"Faktor terkait: {feature} | importance: {importance:.4f}")
        with image_col:
            if image_path is not None:
                st.image(str(image_path), width=160)
        st.divider()


def render_summary(model_artifact: dict) -> None:
    st.subheader("Ringkasan Input")
    summary_df = make_summary_dataframe(st.session_state.answers)
    st.table(summary_df.set_index("Input"))

    if st.button("Prediksi", type="primary", use_container_width=True):
        payload = make_prediction_payload(st.session_state.answers)
        st.session_state.prediction_result = prediksi_diabetes(
            usia=payload["usia"],
            berat_kg=payload["berat_kg"],
            tinggi_cm=payload["tinggi_cm"],
            hba1c=payload["hba1c"],
            fasting_blood_sugar=payload["fasting_blood_sugar"],
            pilihan_genetic=payload["pilihan_genetic"],
            family_history=payload["family_history"],
            physical_activity=payload["physical_activity"],
            dietary_habits=payload["dietary_habits"],
            smoking=payload["smoking"],
            alcohol=payload["alcohol"],
            pilihan_sleep=payload["pilihan_sleep"],
            pilihan_stress=payload["pilihan_stress"],
            model_artifact=model_artifact,
        )

    result = st.session_state.prediction_result
    if result:
        st.subheader("Hasil Prediksi")
        metric_cols = st.columns(3)
        metric_cols[0].metric("Risk Level", result.get("risk_label", "-"))
        risk_pct = result.get("risk_pct")
        metric_cols[1].metric("Diabetes Risk", "-" if risk_pct is None else f"{risk_pct:.1f}%")
        metric_cols[2].metric("BMI", f"{result.get('bmi')} ({result.get('kategori_bmi')})")

        st.caption(f"Model: {result.get('model_name', 'Model')}")
        st.subheader("Probabilitas per Kelas")
        render_probabilities(result.get("proba", {}))
        render_recommendations(result.get("recommendations", []))


def main() -> None:
    init_session_state()

    st.title("Prediksi Risiko Diabetes")
    st.caption("Wizard input klinis dan gaya hidup untuk estimasi tingkat risiko diabetes.")

    try:
        model_artifact = get_model_artifact()
    except Exception as exc:
        st.error(f"Gagal memuat model dari `{DEFAULT_MODEL_PATH}`. Detail: {exc}")
        st.stop()

    try:
        dataset = get_dataset()
    except Exception as exc:
        st.error(f"Gagal memuat dataset dari `{DEFAULT_DATASET_PATH}`. Detail: {exc}")
        st.stop()

    total_steps = len(QUESTIONS)
    current_step = st.session_state.step
    progress_value = min(current_step + 1, total_steps) / total_steps
    st.write(f"Pertanyaan {min(current_step + 1, total_steps)} dari {total_steps}" if current_step < total_steps else "Ringkasan input")
    st.progress(progress_value)

    with st.expander("Informasi dataset", expanded=False):
        st.write(f"Dataset: `{DEFAULT_DATASET_PATH}`")
        st.write(f"Jumlah data: {dataset.shape[0]:,} baris, {dataset.shape[1]} kolom")
        st.write("Kolom dataset:")
        st.write(", ".join(dataset.columns))

    if current_step < total_steps:
        render_question(QUESTIONS[current_step])
        render_navigation(is_summary=False)
    else:
        render_summary(model_artifact)
        render_navigation(is_summary=True)

    st.info(DISCLAIMER)


if __name__ == "__main__":
    main()
