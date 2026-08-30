import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv(
    "dataset/hai_temporal_dataset.csv"
)

df = df.dropna(
    subset=["HAI"]
)


# =========================================================
# 2. FEATURES / TARGET
# =========================================================

X = df.drop(
    columns=["HAI", "Patient_ID"]
)

y = df["HAI"]


# =========================================================
# 3. COLUMNS
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


# =========================================================
# 4. PREPROCESSING
# =========================================================

numerical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])


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
# 5. LOGISTIC REGRESSION
# =========================================================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)


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
# 6. TRAIN / TEST
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


pipeline.fit(
    X_train,
    y_train
)


# =========================================================
# 7. GET PROBABILITIES
# =========================================================

probabilities = pipeline.predict_proba(
    X_test
)[:, 1]


# =========================================================
# 8. TEST DIFFERENT THRESHOLDS
# =========================================================

thresholds = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60
]


results = []


for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)


    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )


    cm = confusion_matrix(
        y_test,
        predictions
    )


    tn, fp, fn, tp = cm.ravel()


    results.append({

        "Threshold": threshold,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "False Positives": fp,

        "False Negatives": fn,

        "True Positives": tp,

        "True Negatives": tn
    })


# =========================================================
# 9. DISPLAY RESULTS
# =========================================================

results_df = pd.DataFrame(
    results
)


print("\n")
print("=" * 95)
print("THRESHOLD ANALYSIS")
print("=" * 95)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# =========================================================
# 10. BEST F1 THRESHOLD
# =========================================================

best = results_df.loc[
    results_df["F1 Score"].idxmax()
]


print("\n")
print("=" * 95)
print("BEST THRESHOLD BASED ON F1 SCORE")
print("=" * 95)

print(
    "Threshold:",
    best["Threshold"]
)

print(
    "Accuracy:",
    round(best["Accuracy"], 4)
)

print(
    "Precision:",
    round(best["Precision"], 4)
)

print(
    "Recall:",
    round(best["Recall"], 4)
)

print(
    "F1 Score:",
    round(best["F1 Score"], 4)
)

print(
    "False Negatives:",
    int(best["False Negatives"])
)

print(
    "False Positives:",
    int(best["False Positives"])
)