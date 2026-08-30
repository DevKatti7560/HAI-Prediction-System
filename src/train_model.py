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
    roc_auc_score
)

from xgboost import XGBClassifier


# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("dataset/hai_dataset.csv")

print("Dataset shape:", df.shape)

print("\nMissing values before preprocessing:")
print(df.isnull().sum())


# ==========================================
# 2. Handle Missing Target Values
# ==========================================

# HAI is the target variable.
# We cannot impute the target because its true value is unknown.

df = df.dropna(subset=["HAI"])

print("\nDataset shape after removing missing HAI:")
print(df.shape)

print("\nMissing values after removing missing HAI:")
print(df.isnull().sum())


# ==========================================
# 3. Separate Features and Target
# ==========================================

X = df.drop("HAI", axis=1)
y = df["HAI"]


# ==========================================
# 4. Define Columns
# ==========================================

categorical_columns = [
    "Gender",
    "Ward_Type"
]

numerical_columns = [
    col for col in X.columns
    if col not in categorical_columns
]


# ==========================================
# 5. Numerical Pipeline
# ==========================================

numerical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])


# ==========================================
# 6. Categorical Pipeline
# ==========================================

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


# ==========================================
# 7. Preprocessor
# ==========================================

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


# ==========================================
# 8. Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 9. Handle Class Imbalance
# ==========================================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print("\nClass distribution:")
print("No HAI:", negative)
print("HAI:", positive)

print(
    "Scale Pos Weight:",
    scale_pos_weight
)


# ==========================================
# 10. XGBoost Model
# ==========================================

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)


# ==========================================
# 11. Complete Pipeline
# ==========================================

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


# ==========================================
# 12. Train Model
# ==========================================

print("\nTraining model...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed!")


# ==========================================
# 13. Predictions
# ==========================================

y_pred = pipeline.predict(X_test)

y_probability = pipeline.predict_proba(
    X_test
)[:, 1]


# ==========================================
# 14. Evaluation
# ==========================================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_test,
        y_pred
    )
)


print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


print("\n==============================")
print("ROC-AUC SCORE")
print("==============================")

print(
    roc_auc_score(
        y_test,
        y_probability
    )
)


# ==========================================
# 15. Save Model
# ==========================================

joblib.dump(
    pipeline,
    "models/hai_model.pkl"
)

print("\nModel saved successfully!")