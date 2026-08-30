import pandas as pd
import numpy as np

np.random.seed(42)

# Number of patients
n = 5000

# -----------------------------------------
# Patient-level information
# -----------------------------------------

data = pd.DataFrame({
    "Patient_ID": [
        f"P{str(i).zfill(4)}"
        for i in range(1, n + 1)
    ],

    "Age": np.random.randint(18, 90, n),

    "Gender": np.random.choice(
        ["Male", "Female"],
        n
    ),

    "Previous_Infection": np.random.choice(
        [0, 1],
        n,
        p=[0.75, 0.25]
    ),

    "Diabetes": np.random.choice(
        [0, 1],
        n,
        p=[0.70, 0.30]
    ),

    "Immunocompromised": np.random.choice(
        [0, 1],
        n,
        p=[0.85, 0.15]
    ),

    "ICU_Admission": np.random.choice(
        [0, 1],
        n,
        p=[0.75, 0.25]
    ),

    "Length_of_Stay": np.random.randint(
        2,
        31,
        n
    ),

    "Catheter_Use": np.random.choice(
        [0, 1],
        n,
        p=[0.65, 0.35]
    ),

    "Ventilator_Use": np.random.choice(
        [0, 1],
        n,
        p=[0.85, 0.15]
    ),

    "Central_Line": np.random.choice(
        [0, 1],
        n,
        p=[0.80, 0.20]
    ),

    "Surgery": np.random.choice(
        [0, 1],
        n,
        p=[0.65, 0.35]
    ),

    "Antibiotic_Exposure": np.random.choice(
        [0, 1],
        n,
        p=[0.60, 0.40]
    ),

    "Ward_Type": np.random.choice(
        [
            "General",
            "ICU",
            "Surgical",
            "Emergency"
        ],
        n
    )
})


# -----------------------------------------
# Initial clinical measurements
# -----------------------------------------

data["Temperature_Day1"] = np.round(
    np.random.normal(36.8, 0.4, n),
    1
)

data["Heart_Rate_Day1"] = np.random.randint(
    60,
    100,
    n
)

data["WBC_Day1"] = np.round(
    np.random.normal(7.5, 1.8, n),
    1
)


# -----------------------------------------
# Clinical measurements near end of stay
# -----------------------------------------

data["Temperature_Day3"] = np.round(
    data["Temperature_Day1"]
    + np.random.normal(0.3, 0.5, n),
    1
)

data["Heart_Rate_Day3"] = (
    data["Heart_Rate_Day1"]
    + np.random.randint(-5, 16, n)
)

data["WBC_Day3"] = np.round(
    data["WBC_Day1"]
    + np.random.normal(1.0, 1.5, n),
    1
)


# -----------------------------------------
# Temporal changes
# -----------------------------------------

data["Temperature_Change"] = np.round(
    data["Temperature_Day3"]
    - data["Temperature_Day1"],
    2
)

data["Heart_Rate_Change"] = (
    data["Heart_Rate_Day3"]
    - data["Heart_Rate_Day1"]
)

data["WBC_Change"] = np.round(
    data["WBC_Day3"]
    - data["WBC_Day1"],
    2
)


# -----------------------------------------
# Generate HAI target
# -----------------------------------------

risk = (
    0.02 * data["Age"]
    + 0.8 * data["Previous_Infection"]
    + 0.6 * data["Diabetes"]
    + 1.0 * data["Immunocompromised"]
    + 1.3 * data["ICU_Admission"]
    + 0.07 * data["Length_of_Stay"]
    + 0.8 * data["Catheter_Use"]
    + 1.0 * data["Ventilator_Use"]
    + 0.7 * data["Central_Line"]
    + 0.7 * data["Surgery"]
    + 0.5 * data["Antibiotic_Exposure"]

    # Temporal factors
    + 1.0 * data["Temperature_Change"]
    + 0.025 * data["Heart_Rate_Change"]
    + 0.15 * data["WBC_Change"]
)


# Convert risk to probability
probability = 1 / (
    1 + np.exp(-(risk - 6.0))
)


data["HAI"] = np.random.binomial(
    1,
    probability
)


# -----------------------------------------
# Save dataset
# -----------------------------------------

data.to_csv(
    "dataset/hai_temporal_dataset.csv",
    index=False
)


print("Temporal dataset generated successfully!")

print("\nDataset shape:")
print(data.shape)

print("\nColumns:")
print(data.columns.tolist())

print("\nHAI distribution:")
print(data["HAI"].value_counts())

print("\nHAI percentage:")
print(
    data["HAI"].value_counts(normalize=True) * 100
)