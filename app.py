
import streamlit as st
import joblib
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Student Result Prediction",
    page_icon="🎓"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("Christ_college_Study_Hrs_Result_prediction_model.pkl")

# -----------------------------
# Title
# -----------------------------
st.title("🎓 Student Result Prediction")
st.write("Predict whether a student will pass or fail using Logistic Regression.")

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0,
    step=1.0
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Result"):

    # Input must be in the same order
    # as used during model training
    input_data = np.array([[study_hours, attendance]])

    prediction = model.predict(input_data)[0]

    # -------------------------
    # Display Result
    # -------------------------
    if prediction == 1:
        st.success("✅ Student is likely to PASS")
    else:
        st.error("❌ Student is likely to FAIL")

    # Probability
    probability = model.predict_proba(input_data)[0]

    st.write(
        f"Pass Probability: **{probability[1] * 100:.2f}%**"
    )

