import unittest

from src.inference import prediksi_diabetes
from src.scoring import calculate_risk_score, create_risk_level
from src.ui_helpers import (
    FAMILY_GENETIC_MAPPING,
    QUESTIONS,
    get_default_answers,
    make_prediction_payload,
    make_summary_dataframe,
)


class RiskScoringTest(unittest.TestCase):
    def test_score_7_is_tidak_berisiko(self):
        data = {
            "Age": 23,
            "BMI": 22.04,
            "HbA1c": 5.7,
            "Fasting_Blood_Sugar": 100,
            "Genetic_Risk_Score": 2,
            "Family_History_Diabetes": 0,
            "Physical_Activity_Level": 1,
            "Dietary_Habits": 1,
            "Smoking": 0,
            "Alcohol_Consumption": 0,
            "Sleep_Hours": 7.5,
            "Stress_Level": 5,
        }

        score = calculate_risk_score(data)

        self.assertEqual(score, 7)
        self.assertEqual(create_risk_level(score), "Tidak Berisiko")

    def test_moderate_case_is_at_least_sedang(self):
        data = {
            "Age": 23,
            "BMI": 27,
            "HbA1c": 6.0,
            "Fasting_Blood_Sugar": 110,
            "Genetic_Risk_Score": 2,
            "Family_History_Diabetes": 1,
            "Physical_Activity_Level": 0,
            "Dietary_Habits": 1,
            "Smoking": 0,
            "Alcohol_Consumption": 0,
            "Sleep_Hours": 7.5,
            "Stress_Level": 5,
        }

        score = calculate_risk_score(data)

        self.assertGreaterEqual(score, 8)
        self.assertIn(create_risk_level(score), {"Sedang", "Tinggi"})

    def test_high_case_is_tinggi(self):
        data = {
            "Age": 25,
            "BMI": 31,
            "HbA1c": 6.5,
            "Fasting_Blood_Sugar": 126,
            "Genetic_Risk_Score": 9,
            "Family_History_Diabetes": 1,
            "Physical_Activity_Level": 0,
            "Dietary_Habits": 0,
            "Smoking": 1,
            "Alcohol_Consumption": 1,
            "Sleep_Hours": 4.5,
            "Stress_Level": 8,
        }

        score = calculate_risk_score(data)

        self.assertGreaterEqual(score, 14)
        self.assertEqual(create_risk_level(score), "Tinggi")

    def test_streamlit_default_case_scoring_and_model(self):
        result = prediksi_diabetes(
            usia=23,
            berat_kg=60,
            tinggi_cm=165,
            hba1c=5.7,
            fasting_blood_sugar=100,
            pilihan_genetic="Tidak ada",
            family_history=0,
            physical_activity=1,
            dietary_habits=1,
            smoking=0,
            alcohol=0,
            pilihan_sleep="7-8 jam (ideal)",
            pilihan_stress="Kadang-kadang stres",
        )

        self.assertEqual(result["risk_score"], 7)
        self.assertEqual(result["scoring_label"], "Tidak Berisiko")
        self.assertEqual(result["model_risk_label"], "Tidak Berisiko")
        self.assertIsNotNone(result["model_confidence_pct"])

    def test_family_genetic_input_is_combined_in_payload(self):
        answers = get_default_answers()
        answers["riwayat_diabetes_keluarga"] = "Ada pada keluarga besar"

        payload = make_prediction_payload(answers)

        self.assertEqual(payload["family_history"], 1)
        self.assertEqual(payload["genetic_risk_score"], 6)
        self.assertEqual(payload["pilihan_genetic"], "Ada pada keluarga besar")

    def test_family_genetic_question_replaces_old_separate_questions(self):
        question_keys = [question["key"] for question in QUESTIONS]

        self.assertIn("riwayat_diabetes_keluarga", question_keys)
        self.assertNotIn("pilihan_genetic", question_keys)
        self.assertNotIn("family_history", question_keys)

    def test_summary_shows_single_family_history_row(self):
        answers = get_default_answers()
        answers["riwayat_diabetes_keluarga"] = "Ada pada ayah/ibu/saudara kandung"

        summary_df = make_summary_dataframe(answers)
        fitur_values = summary_df["Fitur"].tolist()

        self.assertIn("Riwayat diabetes keluarga", fitur_values)
        self.assertNotIn("Risiko Genetik", fitur_values)
        self.assertNotIn("Riwayat Diabetes Keluarga Inti", fitur_values)

    def test_required_family_genetic_mapping_is_available(self):
        self.assertEqual(
            FAMILY_GENETIC_MAPPING["Tidak tahu / gunakan nilai default"],
            {
                "family_history": 0,
                "genetic_risk_score": 5,
                "pilihan_genetic": "Tidak tahu / gunakan nilai default",
            },
        )


if __name__ == "__main__":
    unittest.main()
