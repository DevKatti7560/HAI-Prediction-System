import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="HAI Risk Prediction",
    page_icon="🏥",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = "models/hai_logistic_model.pkl"

try:
    model = joblib.load(MODEL_PATH)

    # =========================================================
    # SHAP EXPLAINER
    # =========================================================

    preprocessor = model.named_steps["preprocessor"]
    logistic_model = model.named_steps["model"]

    shap_background = (
        pd.read_csv(
            "dataset/hai_temporal_dataset.csv"
        )
        .dropna(subset=["HAI"])
        .drop(columns=["HAI", "Patient_ID"])
    )

    shap_background = shap_background.sample(
        min(100, len(shap_background)),
        random_state=42
    )

    background_transformed = preprocessor.transform(
        shap_background
    )

    explainer = shap.LinearExplainer(
        logistic_model,
        background_transformed
    )

    feature_names = (
        preprocessor.get_feature_names_out()
    )

except FileNotFoundError:

    st.error(
        "Model or dataset file not found. "
        "Please check your model and dataset paths."
    )

    st.stop()


# =========================================================
# CONSTANTS
# =========================================================

THRESHOLD = 0.45


# =========================================================
# HEADER
# =========================================================

st.title("🏥 Hospital-Acquired Infection Risk Prediction")

st.markdown(
    """
    **AI-based screening prototype for Hospital-Acquired Infection (HAI) risk**

    This system uses patient history, clinical measurements,
    hospital-environment information and temporal changes
    to estimate HAI risk.
    """
)

st.divider()


# =========================================================
# PATIENT INFORMATION
# =========================================================

st.header("👤 Patient Information")

col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=45
    )


with col2:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )


with col3:

    ward = st.selectbox(
        "Ward Type",
        [
            "General",
            "ICU",
            "Surgical",
            "Emergency"
        ]
    )


# =========================================================
# MEDICAL HISTORY
# =========================================================

st.header("🩺 Medical History")

col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    previous_infection = st.selectbox(
        "Previous Infection",
        [0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )


with col2:

    diabetes = st.selectbox(
        "Diabetes",
        [0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )


with col3:

    immunocompromised = st.selectbox(
        "Immunocompromised",
        [0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )


with col4:

    icu_admission = st.selectbox(
        "ICU Admission",
        [0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )


with col5:

    surgery = st.selectbox(
        "Surgery",
        [0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )


# =========================================================
# HOSPITAL FACTORS
# =========================================================

st.header("🏥 Hospital Environment")

col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    length_of_stay = st.number_input(
        "Length of Stay (days)",
        min_value=1,
        max_value=100,
        value=5
    )


with col2:

    catheter = st.selectbox(
        "Catheter Use",
        [0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )


with col3:

    ventilator = st.selectbox(
        "Ventilator Use",
        [0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )


with col4:

    central_line = st.selectbox(
        "Central Line",
        [0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )


with col5:

    antibiotic = st.selectbox(
        "Antibiotic Exposure",
        [0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )


# =========================================================
# DAY 1 CLINICAL MEASUREMENTS
# =========================================================

st.header("📊 Day 1 Clinical Measurements")

col1, col2, col3 = st.columns(3)


with col1:

    temperature_day1 = st.number_input(
        "Temperature Day 1 (°C)",
        min_value=30.0,
        max_value=45.0,
        value=36.8,
        step=0.1
    )


with col2:

    heart_rate_day1 = st.number_input(
        "Heart Rate Day 1",
        min_value=30,
        max_value=200,
        value=80
    )


with col3:

    wbc_day1 = st.number_input(
        "WBC Count Day 1",
        min_value=1.0,
        max_value=30.0,
        value=7.5,
        step=0.1
    )


# =========================================================
# DAY 3 CLINICAL MEASUREMENTS
# =========================================================

st.header("📈 Day 3 Clinical Measurements")

col1, col2, col3 = st.columns(3)


with col1:

    temperature_day3 = st.number_input(
        "Temperature Day 3 (°C)",
        min_value=30.0,
        max_value=45.0,
        value=37.5,
        step=0.1
    )


with col2:

    heart_rate_day3 = st.number_input(
        "Heart Rate Day 3",
        min_value=30,
        max_value=200,
        value=85
    )


with col3:

    wbc_day3 = st.number_input(
        "WBC Count Day 3",
        min_value=1.0,
        max_value=30.0,
        value=8.5,
        step=0.1
    )


# =========================================================
# TEMPORAL FEATURE CALCULATION
# =========================================================

temperature_change = round(
    temperature_day3 - temperature_day1,
    2
)

heart_rate_change = (
    heart_rate_day3 - heart_rate_day1
)

wbc_change = round(
    wbc_day3 - wbc_day1,
    2
)


st.info(
    f"Temporal changes → "
    f"Temperature: {temperature_change:+.2f} °C | "
    f"Heart Rate: {heart_rate_change:+d} | "
    f"WBC: {wbc_change:+.2f}"
)


# =========================================================
# PREDICTION
# =========================================================

st.divider()

predict_button = st.button(
    "🔍 Predict HAI Risk",
    width="stretch"
)


if predict_button:

    # -----------------------------------------------------
    # CREATE INPUT DATA
    # -----------------------------------------------------

    input_data = pd.DataFrame([{

        "Age": age,

        "Gender": gender,

        "Previous_Infection":
            previous_infection,

        "Diabetes":
            diabetes,

        "Immunocompromised":
            immunocompromised,

        "ICU_Admission":
            icu_admission,

        "Length_of_Stay":
            length_of_stay,

        "Catheter_Use":
            catheter,

        "Ventilator_Use":
            ventilator,

        "Central_Line":
            central_line,

        "Surgery":
            surgery,

        "Antibiotic_Exposure":
            antibiotic,

        "Ward_Type":
            ward,

        "Temperature_Day1":
            temperature_day1,

        "Heart_Rate_Day1":
            heart_rate_day1,

        "WBC_Day1":
            wbc_day1,

        "Temperature_Day3":
            temperature_day3,

        "Heart_Rate_Day3":
            heart_rate_day3,

        "WBC_Day3":
            wbc_day3,

        "Temperature_Change":
            temperature_change,

        "Heart_Rate_Change":
            heart_rate_change,

        "WBC_Change":
            wbc_change
    }])


    # -----------------------------------------------------
    # PREDICT PROBABILITY
    # -----------------------------------------------------

    probability = model.predict_proba(
        input_data
    )[0][1]


    # =====================================================
    # INDIVIDUAL SHAP EXPLANATION
    # =====================================================

    input_transformed = preprocessor.transform(
        input_data
    )

    shap_explanation = explainer(
        input_transformed
    )

    shap_values = shap_explanation.values[0]

    explanation_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": shap_values
    })

    # Sort by absolute impact
    explanation_df["Absolute_Impact"] = (
        explanation_df["Impact"].abs()
    )

    explanation_df = explanation_df.sort_values(
        "Absolute_Impact",
        ascending=False
    )

    top_features = explanation_df.head(8)


    st.subheader(
        "🧠 Why did the model make this prediction?"
    )


    for _, row in top_features.iterrows():

        feature = (
            row["Feature"]
            .replace("numerical__", "")
            .replace("categorical__", "")
        )

        impact = row["Impact"]

        if impact > 0:

            st.write(
                f"🔴 **{feature}** → "
                f"increases HAI risk"
            )

        else:

            st.write(
                f"🟢 **{feature}** → "
                f"reduces HAI risk"
            )


    # -----------------------------------------------------
    # APPLY OPTIMIZED THRESHOLD
    # -----------------------------------------------------

    if probability >= THRESHOLD:

        risk_level = "HIGH RISK"

    elif probability >= 0.30:

        risk_level = "MEDIUM RISK"

    else:

        risk_level = "LOW RISK"


    # -----------------------------------------------------
    # DISPLAY RESULT
    # -----------------------------------------------------

    st.divider()

    st.header("📋 Prediction Result")


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "HAI Probability",
            f"{probability * 100:.2f}%"
        )


    with col2:

        st.metric(
            "Screening Threshold",
            f"{THRESHOLD:.2f}"
        )


    if risk_level == "HIGH RISK":

        st.error(
            "🔴 HIGH RISK — "
            "Patient should be flagged for professional review."
        )

    elif risk_level == "MEDIUM RISK":

        st.warning(
            "🟠 MEDIUM RISK — "
            "Consider closer monitoring."
        )

    else:

        st.success(
            "🟢 LOW RISK"
        )


    # -----------------------------------------------------
    # TEMPORAL ANALYSIS
    # -----------------------------------------------------

    st.subheader(
        "📈 Temporal Clinical Changes"
    )


    temporal_df = pd.DataFrame({

        "Measurement": [
            "Temperature",
            "Heart Rate",
            "WBC Count"
        ],

        "Day 1": [
            temperature_day1,
            heart_rate_day1,
            wbc_day1
        ],

        "Day 3": [
            temperature_day3,
            heart_rate_day3,
            wbc_day3
        ],

        "Change": [
            temperature_change,
            heart_rate_change,
            wbc_change
        ]
    })


    st.dataframe(
        temporal_df,
        width="stretch",
        hide_index=True
    )


    # -----------------------------------------------------
    # RISK FACTOR SUMMARY
    # -----------------------------------------------------

    st.subheader(
        "⚠️ Important Risk Factors"
    )


    risk_factors = []


    if length_of_stay >= 10:

        risk_factors.append(
            "Long hospital stay"
        )


    if icu_admission == 1:

        risk_factors.append(
            "ICU admission"
        )


    if surgery == 1:

        risk_factors.append(
            "Recent surgery"
        )


    if previous_infection == 1:

        risk_factors.append(
            "Previous infection"
        )


    if catheter == 1:

        risk_factors.append(
            "Catheter use"
        )


    if ventilator == 1:

        risk_factors.append(
            "Ventilator use"
        )


    if immunocompromised == 1:

        risk_factors.append(
            "Immunocompromised status"
        )


    if temperature_change >= 0.5:

        risk_factors.append(
            "Increasing temperature"
        )


    if wbc_change >= 2:

        risk_factors.append(
            "Increasing WBC count"
        )


    if heart_rate_change >= 15:

        risk_factors.append(
            "Increasing heart rate"
        )


    if risk_factors:

        for factor in risk_factors:

            st.write(
                f"• {factor}"
            )

    else:

        st.write(
            "No major predefined risk indicators detected."
        )


# =========================================================
# DISCLAIMER
# =========================================================

st.divider()

st.caption(
    """
    ⚠️ Educational / research prototype only.
    This system is not a medical diagnostic tool and
    should not replace assessment by qualified healthcare
    professionals.
    """
)