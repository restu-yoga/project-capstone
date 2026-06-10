from __future__ import annotations

import base64
import html
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
    "Hasil prediksi bukan diagnosis medis. Pemeriksaan dan diagnosis diabetes tetap harus dilakukan oleh tenaga kesehatan."
)
RISK_EXPLANATIONS = {
    "Tidak Berisiko": "Berdasarkan data yang Anda isi, risiko diabetes Anda saat ini tergolong rendah. Tetap pertahankan pola hidup sehat.",
    "Rendah": "Berdasarkan data yang Anda isi, risiko diabetes Anda saat ini tergolong rendah. Tetap pertahankan pola hidup sehat.",
    "Sedang": "Berdasarkan data yang Anda isi, sistem memperkirakan risiko diabetes Anda berada pada kategori sedang. Beberapa faktor perlu diperhatikan dan diperbaiki.",
    "Tinggi": "Berdasarkan data yang Anda isi, sistem memperkirakan risiko diabetes Anda berada pada kategori tinggi. Anda disarankan untuk lebih waspada dan mempertimbangkan pemeriksaan lebih lanjut.",
}
st.set_page_config(
    page_title="Prediksi Risiko Diabetes",
    page_icon="D",
    layout="centered",
)


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 2.5rem;
        }
        .app-subtitle {
            color: #475569;
            font-size: 1.02rem;
            line-height: 1.55;
            margin: -0.35rem 0 1.25rem 0;
        }
        .result-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
        }
        .result-card {
            padding: 1rem 1.05rem;
            min-height: 120px;
        }
        .card-label {
            color: #64748b;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .card-value {
            color: #0f172a;
            font-size: 1.5rem;
            font-weight: 800;
            line-height: 1.2;
            overflow-wrap: anywhere;
        }
        .card-caption {
            color: #64748b;
            font-size: 0.82rem;
            font-weight: 600;
            margin-top: 0.45rem;
        }
        .recommendation-card {
            background: rgba(17, 24, 39, 0.96);
            border: 1px solid rgba(148, 163, 184, 0.35);
            border-radius: 16px;
            padding: 20px;
            min-height: 170px;
            display: flex;
            align-items: center;
            gap: 18px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
            margin-bottom: 1rem;
        }
        .recommendation-card img {
            width: 86px;
            height: 86px;
            object-fit: contain;
            flex-shrink: 0;
        }
        .recommendation-title {
            color: #F8FAFC !important;
            font-size: 1.08rem;
            font-weight: 800;
            margin-bottom: 8px;
        }
        .recommendation-text {
            color: #CBD5E1 !important;
            font-size: 0.98rem;
            line-height: 1.6;
            font-weight: 500;
        }
        .section-note {
            color: #64748b;
            font-size: 0.93rem;
            margin-top: -0.45rem;
            margin-bottom: 0.8rem;
        }
        .disclaimer-box {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            color: #475569;
            font-size: 0.9rem;
            line-height: 1.45;
            padding: 0.85rem 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
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


def render_metric_card(label: str, value: str, caption: str | None = None) -> None:
    caption_html = f'<div class="card-caption">{html.escape(caption)}</div>' if caption else ""
    st.markdown(
        f"""
        <div class="result-card">
            <div class="card-label">{html.escape(label)}</div>
            <div class="card-value">{html.escape(value)}</div>
            {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_recommendation_title(recommendation: dict) -> str:
    recommendation_id = recommendation.get("id", "")
    feature = recommendation.get("feature", "")
    title_map = {
        "hba1c_diabetes": "Pemeriksaan Medis",
        "hba1c_prediabetes": "Pemeriksaan Medis",
        "fbs_diabetes": "Pemeriksaan Medis",
        "fbs_prediabetes": "Pemeriksaan Medis",
        "family_history": "Pemeriksaan Berkala",
        "activity_sedentary": "Aktivitas Fisik",
        "activity_moderate": "Aktivitas Fisik",
        "genetic_high": "Pencegahan Dini",
        "genetic_medium": "Pencegahan Dini",
        "diet_unhealthy": "Pola Makan",
        "diet_moderate": "Pola Makan",
        "stress_high": "Manajemen Stres",
        "stress_medium": "Manajemen Stres",
        "sleep_low": "Tidur",
        "smoking": "Berhenti Merokok",
        "alcohol": "Konsumsi Alkohol",
        "general": "Langkah Berikutnya",
    }
    feature_map = {
        "BMI": "Berat Badan",
        "Physical_Activity_Level": "Aktivitas Fisik",
        "Dietary_Habits": "Pola Makan",
        "Sleep_Hours": "Tidur",
        "Stress_Level": "Manajemen Stres",
    }
    return title_map.get(recommendation_id, feature_map.get(feature, "Rekomendasi"))


def image_to_base64_data_uri(image_path: Path | None) -> str | None:
    if image_path is None:
        return None
    return f"data:image/png;base64,{base64.b64encode(image_path.read_bytes()).decode('ascii')}"


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
    st.markdown(
        """
        <div style="text-align:center; margin-top: 12px; margin-bottom: 34px;">
            <h3 style="margin:0; font-size: 1.65rem; font-weight: 800;">
                Estimasi Tingkat Risiko Diabetes
            </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%", "valueformat": ".2f", "font": {"size": 58}},
            title={"text": ""},
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
    fig.update_layout(height=390, margin={"l": 25, "r": 25, "t": 10, "b": 20})
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        """
        <p style="text-align:center; color:#A1A1AA; margin-top: 8px;">
            Persentase ini merupakan estimasi model berdasarkan data yang dimasukkan, bukan diagnosis medis.
        </p>
        """,
        unsafe_allow_html=True,
    )


def render_recommendations(recommendations: list[dict]) -> None:
    st.subheader("Rekomendasi Personal")
    if not recommendations:
        st.info("Belum ada rekomendasi khusus untuk ditampilkan.")
        return

    for index, recommendation in enumerate(recommendations):
        image_path = resolve_asset_path(recommendation.get("asset", ""), ASSETS_DIR)
        image_data_uri = image_to_base64_data_uri(image_path)
        image_html = f'<img src="{image_data_uri}" alt="">' if image_data_uri else ""
        title = html.escape(get_recommendation_title(recommendation))
        text = html.escape(recommendation["text"])
        if index % 2 == 0:
            card_cols = st.columns(2, gap="medium")

        with card_cols[index % 2]:
            st.markdown(
                f"""
                <div class="recommendation-card">
                    {image_html}
                    <div>
                        <div class="recommendation-title">{title}</div>
                        <div class="recommendation-text">{text}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


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
    st.markdown('<div class="section-note">Periksa kembali data yang Anda masukkan sebelum melakukan prediksi.</div>', unsafe_allow_html=True)
    with st.container(border=True):
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
        st.divider()
        st.subheader("Hasil Prediksi")
        model_risk_label = result.get("model_risk_label", result.get("risk_label", "-"))
        risk_estimate_pct = calculate_diabetes_risk_estimate(result.get("proba", {}))
        risk_estimate_text = "-" if risk_estimate_pct is None else f"{risk_estimate_pct:.2f}%"
        bmi_text = f"{result.get('bmi', '-')} ({result.get('kategori_bmi', '-')})"

        card_cols = st.columns(3, gap="medium")
        with card_cols[0]:
            render_metric_card("Kategori Prediksi Model", model_risk_label)
        with card_cols[1]:
            render_metric_card("Estimasi Risiko", risk_estimate_text, "Berdasarkan input pengguna")
        with card_cols[2]:
            render_metric_card("BMI", bmi_text)

        st.divider()
        render_risk_estimate_gauge(risk_estimate_pct)

        st.divider()
        st.subheader("Keterangan Singkat")
        explanation = get_risk_explanation(model_risk_label)
        if model_risk_label in {"Tidak Berisiko", "Rendah"}:
            st.success(explanation)
        elif model_risk_label == "Tinggi":
            st.error(explanation)
        else:
            st.warning(explanation)

        st.divider()
        render_recommendations(result.get("recommendations", []))
        st.divider()
        render_technical_details(result)


def main() -> None:
    inject_custom_css()
    init_session_state()

    st.title("Prediksi Risiko Diabetes")
    st.markdown(
        '<div class="app-subtitle">Aplikasi ini membantu memperkirakan tingkat risiko diabetes berdasarkan data kesehatan dan gaya hidup yang Anda masukkan.</div>',
        unsafe_allow_html=True,
    )

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

    st.divider()
    st.markdown(f'<div class="disclaimer-box">{DISCLAIMER}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
