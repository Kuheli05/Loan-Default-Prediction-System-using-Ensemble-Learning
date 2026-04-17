import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import io

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Loan Default Predictor", layout="wide")

# -------------------------------
# Load models
# -------------------------------
xgb_model = joblib.load('../model/xgb_model.pkl')
rf_model = joblib.load('../model/rf_model.pkl')
explainer = shap.TreeExplainer(xgb_model)
scaler = joblib.load('../model/scaler.pkl')
le_dict = joblib.load('../model/label_encoders.pkl')

# Feature names
try:
    feature_names = joblib.load('../model/features.pkl')
except:
    feature_names = [
        'LOAN', 'MORTDUE', 'VALUE', 'REASON', 'JOB',
        'YOJ', 'DEROG', 'DELINQ', 'CLAGE',
        'NINQ', 'CLNO', 'DEBTINC'
    ]

# -------------------------------
# Header
# -------------------------------
st.markdown("## 🏦 Loan Default Prediction System")
st.markdown("### Smart AI-based Risk Analysis Dashboard")

# -------------------------------
# Sidebar (Model Selection)
# -------------------------------
st.sidebar.header("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["XGBoost", "Random Forest", "Ensemble (Average)"]
)

threshold = st.sidebar.slider("Risk Threshold", 0.0, 1.0, 0.5)

# -------------------------------
# Input Section (Columns)
# -------------------------------
st.subheader("📋 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    loan = st.number_input("Loan Amount", 1000, 1000000, step=1000)
    mortdue = st.number_input("Mortgage Due", 0, 500000, step=1000)
    value = st.number_input("Property Value", 0, 1000000, step=1000)

with col2:
    reason = st.selectbox("Loan Purpose", ["HomeImp", "DebtCon"])
    job = st.selectbox("Job Type", ["Mgr", "Office", "Self", "ProfExe", "Other", "Sales"])

with col3:
    yoj = st.number_input("Years on Job", 0, 50)
    derog = st.number_input("Derogatory Reports", 0, 10)
    delinq = st.number_input("Delinquencies", 0, 10)

col4, col5, col6 = st.columns(3)

with col4:
    clage = st.number_input("Credit Age", 0.0, 500.0)

with col5:
    ninq = st.number_input("Recent Inquiries", 0, 10)

with col6:
    clno = st.number_input("Credit Lines", 0, 50)
    debtinc = st.number_input("Debt-to-Income Ratio", 0.0, 100.0)

# -------------------------------
# PDF Function
# -------------------------------
def generate_pdf(data, img_buffer):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Loan Default Prediction Report", styles['Title']))
    content.append(Spacer(1, 12))

    for key, value in data.items():
        content.append(Paragraph(f"<b>{key}:</b> {value}", styles['Normal']))
        content.append(Spacer(1, 10))

    content.append(Spacer(1, 20))
    content.append(Paragraph("Risk Probability Graph", styles['Heading2']))
    content.append(Spacer(1, 10))

    img = Image(img_buffer, width=400, height=250)
    content.append(img)

    doc.build(content)
    buffer.seek(0)
    return buffer

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("🚀 Predict"):

    try:
        reason_encoded = le_dict['REASON'].transform([reason])[0]
        job_encoded = le_dict['JOB'].transform([job])[0]

        input_dict = {
            'LOAN': loan,
            'MORTDUE': mortdue,
            'VALUE': value,
            'REASON': reason_encoded,
            'JOB': job_encoded,
            'YOJ': yoj,
            'DEROG': derog,
            'DELINQ': delinq,
            'CLAGE': clage,
            'NINQ': ninq,
            'CLNO': clno,
            'DEBTINC': debtinc
        }

        input_df = pd.DataFrame([input_dict])[feature_names]
        input_scaled = scaler.transform(input_df)

        # Predictions
        xgb_prob = xgb_model.predict_proba(input_scaled)[0][1]
        rf_prob = rf_model.predict_proba(input_scaled)[0][1]

        if model_choice == "XGBoost":
            probability = xgb_prob
        elif model_choice == "Random Forest":
            probability = rf_prob
        else:
            probability = (xgb_prob + rf_prob) / 2

        prediction = 1 if probability >= threshold else 0
        result_text = "High Risk" if prediction == 1 else "Low Risk"

        # -------------------------------
        # Metrics (Dashboard Style)
        # -------------------------------
        st.subheader("📊 Prediction Summary")

        m1, m2, m3 = st.columns(3)

        m1.metric("Risk Probability", f"{probability:.2f}")
        m2.metric("Model Used", model_choice)
        m3.metric("Decision", result_text)

        # -------------------------------
        # Graph
        # -------------------------------
        st.subheader("📈 Risk Visualization")

        fig, ax = plt.subplots()
        ax.bar(["Risk"], [probability])
        ax.set_ylim(0, 1)
        st.pyplot(fig)

        # Save graph for PDF
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png')
        img_buffer.seek(0)

        # -------------------------------
        # SHAP
        # -------------------------------
        if model_choice in ["XGBoost", "Ensemble (Average)"]:

            st.subheader("🧠 Explainability (SHAP)")

            try:
                shap_values = explainer(input_df)
                shap_values_array = shap_values.values[0]
            except:
                shap_values = explainer.shap_values(input_df)
                shap_values_array = shap_values[0]

            shap_df = pd.DataFrame({
                'Feature': feature_names,
                'Impact': shap_values_array
            }).sort_values(by='Impact', key=abs, ascending=False)

            fig3, ax3 = plt.subplots()
            ax3.barh(shap_df['Feature'], shap_df['Impact'])
            ax3.invert_yaxis()
            st.pyplot(fig3)

        # -------------------------------
        # PDF
        # -------------------------------
        report_data = {
            "Loan Amount": loan,
            "Mortgage Due": mortdue,
            "Property Value": value,
            "Job": job,
            "Loan Purpose": reason,
            "Prediction": result_text,
            "Probability": f"{probability:.2f}"
        }

        pdf = generate_pdf(report_data, img_buffer)

        st.download_button(
            label="📄 Download Report",
            data=pdf,
            file_name="loan_report.pdf",
            mime="application/pdf"
        )

    except Exception as e:
        st.error(f"Error: {str(e)}")