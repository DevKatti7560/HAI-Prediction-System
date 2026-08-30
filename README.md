🏥 HAI-Prediction

Hospital-Acquired Infection Risk Prediction System

A machine learning-based screening prototype that estimates the risk of Hospital-Acquired Infection (HAI) using patient history, clinical measurements, hospital-environment factors, and temporal changes in patient condition.

📌 Case Study

Case Study 25 – Predicting Hospital-Acquired Infections

Hospitals need methods to identify patients who may be at increased risk of developing hospital-acquired infections.

This project develops a machine learning system that estimates HAI risk and provides an interpretable explanation of the prediction.

🎯 Objectives

Predict the probability of Hospital-Acquired Infection.

Analyze patient history and hospital-related risk factors.

Incorporate temporal clinical measurements.

Handle missing information.

Address class imbalance.

Compare multiple machine learning algorithms.

Optimize the classification threshold.

Provide explainable predictions using SHAP.

Develop an interactive Streamlit dashboard.

🧠 Machine Learning Models

Three models were evaluated:

Logistic Regression

Random Forest

XGBoost

Model Comparison

Model

Accuracy

Precision

Recall

F1 Score

ROC-AUC

Logistic Regression

72.60%

49.60%

68.27%

57.45%

0.7978

Random Forest

75.80%

58.10%

38.38%

46.22%

0.7519

XGBoost

73.50%

50.99%

56.83%

53.75%

0.7619

Selected Model

Logistic Regression was selected based on its stronger HAI recall and ROC-AUC performance.

⏱️ Temporal Features

The system incorporates changes in clinical measurements between hospital days.

Measurements

Temperature

Heart Rate

WBC Count

Temporal Features

Temperature Change

Heart Rate Change

WBC Change

Example:

Day 1 Temperature → 36.8°C
Day 3 Temperature → 38.2°C

Temperature Change → +1.4°C

This allows the system to consider trends rather than only individual measurements.

⚖️ Class Imbalance

The temporal dataset contains:

No HAI: 2915

HAI: 1085

The imbalance was addressed using class weighting.

For Logistic Regression and Random Forest:

class_weight="balanced"

For XGBoost:

scale_pos_weight

🧹 Missing Data Handling

Missing feature values are handled using:

Numerical Features

Median imputation.

Categorical Features

Most-frequent-value imputation.

Rows with missing HAI target values are removed because the target cannot be reliably inferred.

🎯 Threshold Optimization

The default classification threshold of 0.50 was evaluated against several alternatives.

At threshold 0.50:

Recall: 68.27%

False Negatives: 86

At threshold 0.45:

Recall: 77.49%

False Negatives: 61

Precision: 46.56%

F1 Score: 58.17%

A threshold of 0.45 was selected for the screening prototype because reducing false negatives is important when identifying potentially high-risk HAI cases.

🧠 Explainable AI

SHAP was used to understand which features influence model predictions.

Top Global Features

Length of Stay

ICU Admission

Surgery

Previous Infection

Age

Temperature Change

Catheter Use

Ventilator Use

Immunocompromised Status

Antibiotic Exposure

SHAP explanations are also incorporated into the application to help interpret individual predictions.

🌐 Streamlit Application

The application provides:

Patient information input

Medical history input

Hospital-environment factors

Day 1 clinical measurements

Day 3 clinical measurements

Temporal change calculation

HAI probability

Low / Medium / High risk classification

SHAP-based explanation

Prediction dashboard

Prediction history

Risk distribution visualization

🏗️ Project Structure

HAI-Prediction/
│
├── dataset/
│   ├── hai_dataset.csv
│   └── hai_temporal_dataset.csv
│
├── models/
│   ├── hai_model.pkl
│   ├── hai_temporal_model.pkl
│   └── hai_logistic_model.pkl
│
├── outputs/
│   ├── shap_feature_importance.csv
│   ├── shap_feature_importance.png
│   └── shap_summary.png
│
├── src/
│   ├── generate_dataset.py
│   ├── generate_temporal_dataset.py
│   ├── train_model.py
│   ├── train_temporal_model.py
│   ├── model_comparison.py
│   ├── threshold_analysis.py
│   └── explain_model.py
│
├── app.py
├── README.md
└── requirements.txt

⚙️ Installation

Clone the repository and navigate into the project:

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd HAI-Prediction

Install dependencies:

pip install -r requirements.txt

▶️ Running the Project

Generate the basic dataset:

python src/generate_dataset.py

Generate the temporal dataset:

python src/generate_temporal_dataset.py

Train the original model:

python src/train_model.py

Train the temporal model:

python src/train_temporal_model.py

Compare models:

python src/model_comparison.py

Analyze classification thresholds:

python src/threshold_analysis.py

Generate SHAP explanations:

python src/explain_model.py

Launch the Streamlit application:

streamlit run app.py

📊 Evaluation

The project evaluates models using:

Accuracy

Precision

Recall

F1 Score

ROC-AUC

Confusion Matrix

Recall is particularly important because false-negative predictions may result in potentially high-risk patients not being flagged.

🔐 Important Limitation

This project is an educational/research prototype.

It does not provide medical diagnosis or treatment recommendations and should not replace assessment by qualified healthcare professionals.

The datasets used for this prototype are synthetic and are intended for demonstrating the machine learning workflow.

🚀 Future Scope

Use real-world hospital datasets.

Add continuous patient monitoring.

Incorporate additional laboratory measurements.

Use LSTM/GRU models for longer patient sequences.

Add model calibration.

Deploy using Docker.

Add secure hospital authentication.

Integrate with hospital information systems.

Perform external validation on unseen hospital data.

Improve fairness and bias evaluation.

🛠️ Technologies

Python

Pandas

NumPy

Scikit-learn

XGBoost

SHAP

Matplotlib

Plotly

Streamlit

Joblib

👨‍💻 Author

Devaraja Katti

AI & ML Engineering Student