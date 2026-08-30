import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# =========================================================
# 1. LOAD TEMPORAL DATASET
# =========================================================

df = pd.read_csv(
    "dataset/hai_temporal_dataset.csv"
)

print("=" * 65)
print("HAI MODEL COMPARISON")
print("=" * 65)


# =========================================================
# 2. REMOVE MISSING TARGET
# =========================================================

df = df.dropna(
    subset=["HAI"]
)


# =========================================================
# 3. FEATURES AND TARGET
# =========================================================

X = df.drop(
    columns=["HAI", "Patient_ID"]
)

y = df["HAI"]


# =========================================================
# 4. COLUMNS
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
# 5. PREPROCESSING
# =========================================================

numerical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    )
])


categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),

    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore"
        )
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
# 6. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


# =========================================================
# 7. CLASS IMBALANCE
# =========================================================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print("\nClass Distribution")
print("-------------------")
print("No HAI:", negative)
print("HAI:", positive)
print(
    "Scale Pos Weight:",
    round(scale_pos_weight, 4)
)


# =========================================================
# 8. DEFINE MODELS
# =========================================================

models = {

    "Logistic Regression":

        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),


    "Random Forest":

        RandomForestClassifier(

            n_estimators=250,

            max_depth=10,

            class_weight="balanced",

            random_state=42,

            n_jobs=-1
        ),


    "XGBoost":

        XGBClassifier(

            n_estimators=250,

            max_depth=5,

            learning_rate=0.05,

            subsample=0.8,

            colsample_bytree=0.8,

            scale_pos_weight=scale_pos_weight,

            eval_metric="logloss",

            random_state=42
        )
}


# =========================================================
# 9. TRAIN AND EVALUATE
# =========================================================

results = []


for name, model in models.items():

    print("\n")
    print("=" * 65)
    print("Training:", name)
    print("=" * 65)


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


    pipeline.fit(
        X_train,
        y_train
    )


    # Predictions

    y_pred = pipeline.predict(
        X_test
    )

    y_probability = pipeline.predict_proba(
        X_test
    )[:, 1]


    # Metrics

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


    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC-AUC": roc_auc
    })


# =========================================================
# 10. RESULTS TABLE
# =========================================================

results_df = pd.DataFrame(
    results
)


print("\n\n")
print("=" * 65)
print("FINAL MODEL COMPARISON")
print("=" * 65)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# =========================================================
# 11. BEST MODEL
# =========================================================

best_model = results_df.loc[
    results_df["ROC-AUC"].idxmax()
]


print("\n")
print("=" * 65)
print("BEST MODEL")
print("=" * 65)

print(
    "Model:",
    best_model["Model"]
)

print(
    "ROC-AUC:",
    round(
        best_model["ROC-AUC"],
        4
    )
)

print(
    "HAI Recall:",
    round(
        best_model["Recall"],
        4
    )
)

print(
    "HAI F1:",
    round(
        best_model["F1 Score"],
        4
    )
)