🏥 HAI Prediction System

Machine Learning-Based Hospital-Acquired Infection Risk Prediction

An end-to-end machine learning application that estimates the risk of Hospital-Acquired Infection (HAI) using patient history, clinical measurements, hospital-environment factors, and temporal changes in patient condition.

The project includes data preprocessing, class-imbalance handling, temporal feature engineering, model comparison, threshold optimization, SHAP explainability, and an interactive Streamlit dashboard.

⚠️ Disclaimer: This is an educational/research prototype using synthetic data. It is not a medical diagnostic or treatment system.

📌 Case Study 25 – Predicting Hospital-Acquired Infections

Problem Statement

A hospital wants to identify patients who are at increased risk of developing hospital-acquired infections.

The system uses:

Patient history

Clinical measurements

Length of hospital stay

Hospital environment information

Temporal clinical changes

The project also considers:

Missing information

Class imbalance

False positives and false negatives

Model explainability

🎯 Objectives

Predict the probability of Hospital-Acquired Infection.

Identify patients at increased risk.

Incorporate temporal clinical measurements.

Handle missing patient information.

Address class imbalance.

Compare multiple machine learning algorithms.

Optimize the prediction threshold.

Explain predictions using SHAP.

Provide an interactive web-based dashboard.

🧠 System Architecture

                    PATIENT DATA
                         │
                         ▼
                DATA PREPROCESSING
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
       Missing Data   Encoding   Class Balance
            │            │            │
            └────────────┼────────────┘
                         ▼
                 FEATURE ENGINEERING
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       Patient/Hospital       Temporal Features
          Features            ├── Temperature Change
                              ├── Heart Rate Change
                              └── WBC Change
              │                     │
              └──────────┬──────────┘
                         ▼
                   MODEL COMPARISON
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
        Logistic       Random      XGBoost
       Regression      Forest
             │           │           │
             └───────────┼───────────┘
                         ▼
                  SELECTED MODEL
                 Logistic Regression
                         │
                         ▼
                  THRESHOLD ANALYSIS
                        0.45
                         │
                         ▼
                  HAI RISK PREDICTION
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
          LOW RISK   MEDIUM RISK   HIGH RISK
                                      │
                                      ▼
                               SHAP EXPLANATION
                                      │
                                      ▼
                              STREAMLIT DASHBOARD

📊 Dataset

The project uses synthetic datasets generated specifically for this educational case study.

Basic Dataset

File:

dataset/hai_dataset.csv

Features include:

Age

Gender

Previous Infection

Diabetes

Immunocompromised Status

ICU Admission

Length of Stay

Temperature

Heart Rate

WBC Count

Catheter Use

Ventilator Use

Central Line

Surgery

Antibiotic Exposure

Ward Type

HAI Target

⏱️ Temporal Dataset

File:

dataset/hai_temporal_dataset.csv

Additional temporal measurements include:

Temperature Day 1

Temperature Day 3

Heart Rate Day 1

Heart Rate Day 3

WBC Day 1

WBC Day 3

Temperature Change

Heart Rate Change

WBC Change

Example

Temperature Day 1 → 36.8°C
Temperature Day 3 → 38.2°C

Temperature Change → +1.4°C

This allows the system to analyze clinical trends over time rather than relying only on a single measurement.

🧹 Missing Data Handling

The system handles missing information using preprocessing pipelines.

Numerical Features

SimpleImputer(strategy="median")

Categorical Features

SimpleImputer(strategy="most_frequent")

Rows with missing HAI target values are removed because the actual target cannot reliably be inferred.

⚖️ Class Imbalance

The temporal dataset contains:

No HAI → 2915
HAI    → 1085

Therefore, HAI is the minority class.

Class imbalance is handled using:

Logistic Regression

class_weight="balanced"

Random Forest

class_weight="balanced"

XGBoost

scale_pos_weight

The temporal XGBoost model calculated:

Scale Pos Weight = 2.6866

🤖 Machine Learning Models

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

🏆 Selected Model

Logistic Regression

Although Random Forest achieved the highest overall accuracy, Logistic Regression achieved:

HAI Recall = 68.27%
ROC-AUC    = 0.7978

Therefore, Logistic Regression was selected because the project prioritizes identifying potentially high-risk HAI cases rather than maximizing accuracy alone.

🎯 Threshold Optimization

The default classification threshold of 0.50 was evaluated using multiple thresholds.

Threshold

Accuracy

Precision

Recall

F1 Score

False Negatives

0.30

56.90%

37.73%

90.77%

53.30%

25

0.35

61.30%

40.00%

85.61%

54.52%

39

0.40

66.40%

43.64%

82.29%

57.03%

48

0.45

69.80%

46.56%

77.49%

58.17%

61

0.50

72.60%

49.60%

68.27%

57.45%

86

0.55

75.40%

53.87%

64.21%

58.59%

97

0.60

77.20%

57.71%

59.41%

58.55%

110

Selected Screening Threshold: 0.45

At a threshold of 0.45:

HAI Recall      = 77.49%
Precision       = 46.56%
F1 Score        = 58.17%
False Negatives = 61
False Positives = 241

The threshold was selected for the screening prototype because increasing recall helps reduce missed HAI cases.

Lowering the threshold increases the number of patients flagged as potentially high risk, which also increases false positives. This demonstrates the trade-off between false negatives and false positives.

🧠 Explainable AI — SHAP

The project uses SHAP (SHapley Additive exPlanations) to understand which features influence model predictions.

Top Global Features

Rank

Feature

SHAP Importance

1

Length of Stay

0.5071

2

ICU Admission

0.4228

3

Surgery

0.3526

4

Previous Infection

0.3298

5

Age

0.3274

6

Temperature Change

0.3212

7

Catheter Use

0.3103

8

Ventilator Use

0.2547

9

Immunocompromised

0.2455

10

Antibiotic Exposure

0.2258

SHAP is also integrated into the Streamlit application to provide explanations for individual predictions.

SHAP Feature Importance

View SHAP Feature Importance

SHAP Summary

View SHAP Summary

🌐 Streamlit Application

The project provides an interactive web application built using Streamlit.

Features

👤 Patient Information

Age

Gender

Ward Type

🩺 Medical History

Previous Infection

Diabetes

Immunocompromised Status

ICU Admission

Surgery

🏥 Hospital Environment

Length of Stay

Catheter Use

Ventilator Use

Central Line

Antibiotic Exposure

📊 Clinical Measurements

Day 1 Temperature

Day 1 Heart Rate

Day 1 WBC

Day 3 Temperature

Day 3 Heart Rate

Day 3 WBC

📈 Prediction

The system calculates:

HAI Probability
      ↓
Threshold = 0.45
      ↓
Low / Medium / High Risk

🧠 Explainability

The application provides SHAP-based explanations showing which features influence the prediction.

📁 Project Structure

HAI-Prediction-System/
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
├── .gitignore
├── README.md
└── requirements.txt

🛠️ Technologies Used

Category

Technologies

Programming

Python

Data Processing

Pandas, NumPy

Machine Learning

Scikit-learn, XGBoost

Explainable AI

SHAP

Visualization

Matplotlib, Plotly

Web Application

Streamlit

Model Persistence

Joblib

Version Control

Git, GitHub

⚙️ Installation

1. Clone the Repository

git clone https://github.com/DevKatti7560/HAI-Prediction-System.git

2. Navigate to the Project

cd HAI-Prediction-System

3. Install Dependencies

pip install -r requirements.txt

▶️ Running the Project

Generate Basic Dataset

python src/generate_dataset.py

Generate Temporal Dataset

python src/generate_temporal_dataset.py

Train Original Model

python src/train_model.py

Train Temporal Model

python src/train_temporal_model.py

Compare Models

python src/model_comparison.py

Perform Threshold Analysis

python src/threshold_analysis.py

Generate SHAP Explanations

python src/explain_model.py

Launch Streamlit Application

streamlit run app.py

The application will be available at:

http://localhost:8501

📈 Evaluation Metrics

The following metrics are used:

Accuracy

Precision

Recall

F1 Score

ROC-AUC

Confusion Matrix

Why Recall?

For HAI screening, false negatives are particularly important.

A false negative means a patient who may be at increased infection risk is not flagged by the model.

Therefore, model selection considers:

HAI Recall
     +
F1 Score
     +
ROC-AUC

rather than accuracy alone.

🔐 Limitations

The datasets are synthetic and created for educational purposes.

The model has not been clinically validated.

The model has not been externally validated on independent hospital datasets.

Temporal data is currently represented using engineered changes between selected time points.

The model should not be used to diagnose or treat patients.

Real-world deployment would require clinical validation, security, privacy controls, monitoring, and regulatory review.

🚀 Future Scope

Train and validate the system using real-world hospital datasets.

Integrate continuous patient monitoring.

Add additional laboratory and clinical measurements.

Use LSTM/GRU models for longer patient sequences.

Perform external validation across multiple hospitals.

Add probability calibration.

Evaluate model fairness and potential bias.

Deploy using Docker and cloud infrastructure.

Integrate with hospital information systems.

Add secure authentication and role-based access.

Implement model monitoring and drift detection.

📊 Key Results

Metric

Result

Selected Model

Logistic Regression

ROC-AUC

0.7978

HAI Recall at 0.50

68.27%

Optimized Screening Threshold

0.45

HAI Recall at 0.45

77.49%

False Negatives at 0.45

61

💡 Key Project Highlights

             🏥 HAI Prediction
                    │
           ┌────────┴────────┐
           │                 │
       ML Models        Temporal Data
           │                 │
           ▼                 ▼
     Logistic LR       Clinical Trends
     Random Forest
     XGBoost
           │
           ▼
    Threshold Optimization
           │
           ▼
      Explainable AI
           │
           ▼
          SHAP
           │
           ▼
    Streamlit Dashboard

👨‍💻 Author

Devaraja Katti

AI & ML Engineering Student

GitHub:
https://github.com/DevKatti7560