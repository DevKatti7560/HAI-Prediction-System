import os
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib

from sklearn.model_selection import train_test_split


# =========================================================
# 1. CREATE OUTPUT DIRECTORY
# =========================================================

os.makedirs("outputs", exist_ok=True)


# =========================================================
# 2. LOAD DATASET
# =========================================================

df = pd.read_csv(
    "dataset/hai_temporal_dataset.csv"
)

# Remove rows where target is missing
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
# 4. SAME TRAIN/TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


# =========================================================
# 5. LOAD LOGISTIC REGRESSION MODEL
# =========================================================

# We need to recreate the preprocessing + model pipeline
# because model_comparison.py does not save the pipeline.

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


categorical_columns = [
    "Gender",
    "Ward_Type"
]

numerical_columns = [
    column
    for column in X.columns
    if column not in categorical_columns
]


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
# 6. TRAIN MODEL
# =========================================================

print("Training Logistic Regression model...")

pipeline.fit(
    X_train,
    y_train
)

print("Model trained successfully!")


# =========================================================
# 7. TRANSFORM DATA
# =========================================================

preprocessor_fitted = pipeline.named_steps[
    "preprocessor"
]

logistic_model = pipeline.named_steps[
    "model"
]


X_train_transformed = preprocessor_fitted.transform(
    X_train
)

X_test_transformed = preprocessor_fitted.transform(
    X_test
)


# =========================================================
# 8. GET FEATURE NAMES
# =========================================================

feature_names = (
    preprocessor_fitted
    .get_feature_names_out()
)


print("\nNumber of model features:")
print(len(feature_names))


# =========================================================
# 9. SHAP EXPLAINER
# =========================================================

print("\nCreating SHAP explainer...")

explainer = shap.LinearExplainer(
    logistic_model,
    X_train_transformed
)


# Explain a sample of test patients
sample_size = min(
    200,
    X_test_transformed.shape[0]
)

X_sample = X_test_transformed[
    :sample_size
]

shap_values = explainer(
    X_sample
)


print("SHAP explanation generated!")


# =========================================================
# 10. GLOBAL FEATURE IMPORTANCE
# =========================================================

print("\nCalculating feature importance...")

mean_abs_shap = (
    abs(shap_values.values)
    .mean(axis=0)
)


importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": mean_abs_shap
})


importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


print("\nTop 15 Important Features:")
print(
    importance_df.head(15).to_string(
        index=False
    )
)


# =========================================================
# 11. SAVE FEATURE IMPORTANCE
# =========================================================

importance_df.to_csv(
    "outputs/shap_feature_importance.csv",
    index=False
)


# =========================================================
# 12. SHAP BAR PLOT
# =========================================================

plt.figure()

shap.summary_plot(
    shap_values.values,
    X_sample,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)

plt.title(
    "Global Feature Importance - HAI Prediction"
)

plt.tight_layout()

plt.savefig(
    "outputs/shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "\nSaved:"
    " outputs/shap_feature_importance.png"
)


# =========================================================
# 13. SHAP SUMMARY PLOT
# =========================================================

plt.figure()

shap.summary_plot(
    shap_values.values,
    X_sample,
    feature_names=feature_names,
    show=False
)

plt.title(
    "SHAP Summary - HAI Prediction"
)

plt.tight_layout()

plt.savefig(
    "outputs/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:"
    " outputs/shap_summary.png"
)


# =========================================================
# 14. SAVE MODEL
# =========================================================

joblib.dump(
    pipeline,
    "models/hai_logistic_model.pkl"
)


print(
    "\nLogistic model saved:"
    " models/hai_logistic_model.pkl"
)


print("\nSHAP analysis completed successfully!")