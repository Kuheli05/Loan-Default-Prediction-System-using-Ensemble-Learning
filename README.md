# Loan-Default-Prediction-System-using-Ensemble-Learning
A machine learning web app that predicts loan default risk using XGBoost and Random Forest. It provides probability scores, model comparison, SHAP-based explanations, and a downloadable PDF report through an interactive Streamlit interface.


🚀 Features
🔍 Loan default prediction (High / Low risk)
🤖 Ensemble learning (XGBoost + Random Forest)
📊 Model comparison with probability scores
🧠 SHAP explainability (feature impact)
🎯 Adjustable risk threshold
📈 Interactive graphs and visualization
📄 Downloadable PDF report with embedded graph
💻 User-friendly Streamlit dashboard


📊 Dataset
Name: HMEQ (Home Equity Loan Dataset)
Records: 5960
Features: 12 input + 1 target (BAD)
Type: Structured tabular dataset
Problem: Binary Classification (Default / No Default)


⚙️ Installation
Clone the repository:
git clone https://github.com/your-username/loan-default-prediction.git
cd loan-default-prediction
Install dependencies:
pip install -r requirements.txt
Run the app:
streamlit run app/app.py


📈 How It Works
User inputs customer details
Data is preprocessed (encoding + scaling)
Models (XGBoost & Random Forest) generate predictions
Ensemble averaging improves accuracy
Probability is calculated and classified
SHAP explains feature contributions
Results + graph + PDF report are generated


🧠 Model Details
XGBoost: Boosting algorithm for high accuracy
Random Forest: Bagging method for stability
Ensemble: Average of both models for better performance
📄 Output
Risk classification (High / Low)
Probability score
Model comparison graph
SHAP feature importance
Downloadable PDF report
xplainability
Add real-time data integration
