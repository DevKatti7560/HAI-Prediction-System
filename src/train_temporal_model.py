import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from xgboost import XGBClassifier


# =========================================================
# 1. LOAD TEMPORAL DATASET
# =========================================================

df = pd.read_csv("dataset/hai_temporal_dataset.csv")

print("=" * 60)
print("TEMPORAL HAI PREDICTION MODEL")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)


# =========================================================
# 2. CHECK MISSING VALUES
# =========================================================

print("\nMissing values:")
print(df.isnull().sum())


# =========================================================
# 3. REMOVE ROWS WITH MISSING TARGET
# =========================================================

df = df.dropna(subset=["HAI"])

print("\nDataset shape after removing missing HAI:")
print(df.shape)


# =========================================================
# 4. SEPARATE FEATURES AND TARGET
# =========================================================

X = df.drop(
    columns=["HAI", "Patient_ID"]
)

y = df["HAI"]


# =========================================================
# 5. IDENTIFY CATEGORICAL AND NUMERICAL FEATURES
# =========================================================

categorical_columns = [
    "Gender",
    "Ward_Type"
]

numerical_columns = [
    column
    for column in X.columns
    if column not in categorical_columns
]


print("\nCategorical features:")
print(categorical_columns)

print("\nNumber of numerical features:")
print(len(numerical_columns))


# =========================================================
# 6. NUMERICAL PIPELINE
# =========================================================

numerical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])


# =========================================================
# 7. CATEGORICAL PIPELINE
# =========================================================

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),

    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])


# =========================================================
# 8. PREPROCESSOR
# =========================================================

preprocessor = ColumnTransformer([
    (
        "numerical",
        numerical_pipeline,
        numerical_columns
    ),

    (
        "categorical",
        categorical_pipeline,
        categorical_columns
    )
])


# =========================================================
# 9. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =========================================================
# 10. CLASS DISTRIBUTION
# =========================================================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

print("\nClass distribution:")
print("No HAI:", negative)
print("HAI:", positive)


# Avoid division by zero
if positive > 0:
    scale_pos_weight = negative / positive
else:
    scale_pos_weight = 1.0


print(
    "Scale Pos Weight:",
    scale_pos_weight
)


# =========================================================
# 11. XGBOOST MODEL
# =========================================================

model = XGBClassifier(

    n_estimators=250,

    max_depth=5,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    scale_pos_weight=scale_pos_weight,

    eval_metric="logloss",

    random_state=42
)


# =========================================================
# 12. COMPLETE PIPELINE
# =========================================================

pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),

    (
        "model",
        model
    )
])


# =========================================================
# 13. TRAIN
# =========================================================

print("\nTraining temporal model...")

pipeline.fit(
    X_train,
    y_train
)

print("Temporal model training completed!")


# =========================================================
# 14. PREDICTION
# =========================================================

y_pred = pipeline.predict(X_test)

y_probability = pipeline.predict_proba(
    X_test
)[:, 1]


# =========================================================
# 15. EVALUATION
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred
    )
)


print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


# =========================================================
# 16. SAVE TEMPORAL MODEL
# =========================================================

joblib.dump(
    pipeline,
    "models/hai_temporal_model.pkl"
)

print("\nTemporal model saved successfully!")

print(
    "\nSaved to:"
    " models/hai_temporal_model.pkl"
)