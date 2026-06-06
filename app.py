from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.inference import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, load_dataset, load_model_artifact, prediksi_diabetes
from src.recommendations import resolve_asset_path
from src.ui_helpers import (
    CAPTIONS,
    CATEGORY_EXPLANATIONS,
    QUESTIONS,
    get_default_answers,
    make_prediction_payload,
    make_summary_dataframe,
)


ASSETS_DIR = Path("assets")
DISCLAIMER = (
    "Aplikasi ini hanya memberikan estimasi risiko berbasis data dan sistem scoring. "
    "Persentase pada gauge bukan diagnosis medis. "
    "Pemeriksaan dan diagnosis diabetes tetap harus dilakukan oleh tenaga kesehatan."
)
RISK_EXPLANATIONS = {
    "Tidak Berisiko": "Berdasarkan data yang Anda isi, sistem memperkirakan risiko diabetes Anda berada pada kategori Tidak Berisiko.",
    "Rendah": "Berdasarkan data yang Anda isi, sistem memperkirakan risiko diabetes Anda berada pada kategori rendah.",
    "Sedang": "Berdasarkan data yang Anda isi, sistem memperkirakan risiko diabetes Anda berada pada kategori Sedang.",
    "Tinggi": "Berdasarkan data yang Anda isi, sistem memperkirakan risiko diabetes Anda berada pada kategori Tinggi.",
}
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


def reset_form() -> None:
    st.session_state.step = 0
    st.session_state.answers = get_default_answers()
    st.session_state.prediction_result = None


def show_category_explanation(feature_key: str) -> None:
    explanation = CATEGORY_EXPLANATIONS.get(feature_key)
    if not explanation:
        return

    with st.expander("Panduan Memilih Kategori", expanded=True):
        st.info("Gunakan panduan berikut untuk memilih kategori yang paling sesuai dengan kondisi Anda.")
        st.markdown(f"**{explanation['title']}**")
        if explanation.get("subtitle"):
            st.caption(explanation["subtitle"])
        st.table(pd.DataFrame(explanation["rows"]))

        if explanation.get("extra_rows"):
            st.markdown(f"**{explanation['extra_title']}**")
            if explanation.get("extra_subtitle"):
                st.caption(explanation["extra_subtitle"])
            st.table(pd.DataFrame(explanation["extra_rows"]))
            if explanation.get("extra_note"):
                st.caption(explanation["extra_note"])


def render_question(question: dict) -> None:
    key = question["key"]
    widget_key = f"input_{key}"
    current_value = st.session_state.answers.get(key, question["default"])

    st.subheader(question["title"])
    st.caption(CAPTIONS.get(key, ""))

    if question["kind"] == "slider":
        value = st.slider(
            question["label"],
            min_value=question["min"],
            max_value=question["max"],
            value=current_value,
            step=question["step"],
            format=question.get("format"),
            key=widget_key,
        )
    elif question["kind"] == "select_map":
        options = list(question["options"].keys())
        selected_value = current_value
        if selected_value not in options:
            selected_value = next(
                (label for label, mapped_value in question["options"].items() if mapped_value == selected_value),
                question["default"],
            )
        value = st.selectbox(
            question["label"],
            options=options,
            index=options.index(selected_value) if selected_value in options else 0,
            key=widget_key,
        )
    else:
        options = question["options"]
        value = st.selectbox(
            question["label"],
            options=options,
            index=options.index(current_value) if current_value in options else 0,
            key=widget_key,
        )

    st.session_state.answers[key] = value
    show_category_explanation(key)


def render_navigation(is_summary: bool = False) -> None:
    back_col, reset_col, next_col = st.columns([1, 1, 1])

    with back_col:
        if st.button("Kembali", disabled=st.session_state.step == 0, width="stretch"):
            st.session_state.prediction_result = None
            st.session_state.step = max(0, st.session_state.step - 1)
            st.rerun()

    with reset_col:
        if st.button("Reset", width="stretch"):
            reset_form()
            st.rerun()

    with next_col:
        if not is_summary and st.button("Lanjut", width="stretch"):
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
    st.dataframe(probability_df, hide_index=True, width="stretch")


def get_risk_explanation(risk_label: str) -> str:
    return RISK_EXPLANATIONS.get(
        risk_label,
        "Hasil ini menunjukkan kategori risiko berdasarkan data yang Anda isi.",
    )


def calculate_diabetes_risk_estimate(probabilities: dict[str, float]) -> float | None:
    if not probabilities:
        return None

    moderate_probability = probabilities.get("Sedang")
    high_probability = probabilities.get("Tinggi")
    if moderate_probability is not None or high_probability is not None:
        return round(((moderate_probability or 0) + (high_probability or 0)) * 100, 2)

    return round(max(probabilities.values()) * 100, 2)


def render_risk_estimate_gauge(estimate_pct: float | None) -> None:
    value = 0 if estimate_pct is None else estimate_pct
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%", "valueformat": ".2f"},
            title={
                "text": "Estimasi Tingkat Risiko Diabetes",
                "font": {"size": 24},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickvals": [0, 33, 66, 100],
                    "ticktext": ["0%", "33%", "66%", "100%"],
                },
                "bar": {"color": "#2563eb", "thickness": 0.25},
                "steps": [
                    {"range": [0, 33], "color": "#D1FAE5"},
                    {"range": [33, 66], "color": "#FEF3C7"},
                    {"range": [66, 100], "color": "#FEE2E2"},
                ],
                "threshold": {
                    "line": {"color": "#111827", "width": 5},
                    "thickness": 0.8,
                    "value": value,
                },
            },
        )
    )
    fig.update_layout(height=350, margin={"l": 40, "r": 40, "t": 80, "b": 30})
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_recommendations(recommendations: list[dict]) -> None:
    st.subheader("Rekomendasi Personal")
    for recommendation in recommendations:
        image_path = resolve_asset_path(recommendation.get("asset", ""), ASSETS_DIR)
        text_col, image_col = st.columns([2, 1])
        with text_col:
            st.write(recommendation["text"])
        with image_col:
            if image_path is not None:
                st.image(str(image_path), width=160)
        st.divider()


def render_technical_details(result: dict) -> None:
    with st.expander("Detail teknis model"):
        st.write(f"Skor Risiko: {result.get('risk_score', '-')}")
        st.write(f"Kategori Scoring: {result.get('scoring_label', '-')}")
        st.write(f"Kategori Prediksi Model: {result.get('model_risk_label', result.get('risk_label', '-'))}")
        confidence = result.get("model_confidence_pct")
        st.write("Keyakinan Model: -" if confidence is None else f"Keyakinan Model: {confidence:.2f}%")
        st.caption(f"Model: {result.get('model_name', 'Model')}")
        st.write(f"BMI: {result.get('bmi')} ({result.get('kategori_bmi')})")
        st.subheader("Probabilitas per Kelas")
        render_probabilities(result.get("proba", {}))


def render_summary(model_artifact: dict) -> None:
    st.subheader("Ringkasan Input")
    summary_df = make_summary_dataframe(st.session_state.answers)
    if "Fitur" in summary_df.columns:
        st.table(summary_df.set_index("Fitur"))
    else:
        st.table(summary_df)

    if st.button("Prediksi", type="primary", width="stretch"):
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
        scoring_label = result.get("scoring_label", "-")
        model_risk_label = result.get("model_risk_label", result.get("risk_label", "-"))
        risk_estimate_pct = calculate_diabetes_risk_estimate(result.get("proba", {}))
        st.markdown("**Kategori Prediksi Model:**")
        st.write(model_risk_label)
        st.markdown("**Estimasi Tingkat Risiko Diabetes:**")
        st.write("-" if risk_estimate_pct is None else f"{risk_estimate_pct:.2f}%")
        render_risk_estimate_gauge(risk_estimate_pct)
        st.caption("Persentase ini merupakan estimasi model berdasarkan data yang dimasukkan, bukan diagnosis medis.")

        st.subheader("Keterangan Singkat")
        st.write(get_risk_explanation(model_risk_label))
        render_recommendations(result.get("recommendations", []))
        render_technical_details(result)


def main() -> None:
    init_session_state()

    st.title("Prediksi Risiko Diabetes")
    st.caption("Form prediksi risiko diabetes dengan pengisian data bertahap.")

    try:
        model_artifact = get_model_artifact()
    except Exception as exc:
        st.error(f"Gagal memuat model dari `{DEFAULT_MODEL_PATH}`. Detail: {exc}")
        st.stop()

    try:
        get_dataset()
    except Exception as exc:
        st.error(f"Gagal memuat dataset dari `{DEFAULT_DATASET_PATH}`. Detail: {exc}")
        st.stop()

    total_steps = len(QUESTIONS)
    current_step = st.session_state.step
    progress_value = min(current_step + 1, total_steps) / total_steps
    st.write(f"Pertanyaan {min(current_step + 1, total_steps)} dari {total_steps}" if current_step < total_steps else "Ringkasan input")
    st.progress(progress_value)

    if current_step < total_steps:
        render_question(QUESTIONS[current_step])
        render_navigation(is_summary=False)
    else:
        render_summary(model_artifact)
        render_navigation(is_summary=True)

    st.info(DISCLAIMER)


if __name__ == "__main__":
    main()
