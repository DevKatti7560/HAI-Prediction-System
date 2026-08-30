import pandas as pd
import numpy as np

np.random.seed(42)

n = 5000

data = pd.DataFrame({
    "Age": np.random.randint(18, 90, n),
    "Gender": np.random.choice(["Male", "Female"], n),

    "Previous_Infection": np.random.choice([0, 1], n, p=[0.75, 0.25]),
    "Diabetes": np.random.choice([0, 1], n, p=[0.7, 0.3]),
    "Immunocompromised": np.random.choice([0, 1], n, p=[0.85, 0.15]),

    "ICU_Admission": np.random.choice([0, 1], n, p=[0.75, 0.25]),

    "Length_of_Stay": np.random.randint(1, 31, n),

    "Temperature": np.round(
        np.random.normal(37.0, 0.7, n), 1
    ),

    "Heart_Rate": np.random.randint(55, 130, n),

    "WBC_Count": np.round(
        np.random.normal(8, 3, n), 1
    ),

    "Catheter_Use": np.random.choice([0, 1], n, p=[0.65, 0.35]),
    "Ventilator_Use": np.random.choice([0, 1], n, p=[0.85, 0.15]),
    "Central_Line": np.random.choice([0, 1], n, p=[0.8, 0.2]),
    "Surgery": np.random.choice([0, 1], n, p=[0.65, 0.35]),
    "Antibiotic_Exposure": np.random.choice([0, 1], n, p=[0.6, 0.4]),

    "Ward_Type": np.random.choice(
        ["General", "ICU", "Surgical", "Emergency"],
        n
    )
})

# Create risk score for prototype target generation
risk = (
    0.03 * data["Age"]
    + 0.8 * data["Previous_Infection"]
    + 0.6 * data["Diabetes"]
    + 1.0 * data["Immunocompromised"]
    + 1.5 * data["ICU_Admission"]
    + 0.08 * data["Length_of_Stay"]
    + 0.8 * data["Catheter_Use"]
    + 1.0 * data["Ventilator_Use"]
    + 0.7 * data["Central_Line"]
    + 0.7 * data["Surgery"]
    + 0.5 * data["Antibiotic_Exposure"]
)

probability = 1 / (1 + np.exp(-(risk - 4.5)))

data["HAI"] = np.random.binomial(1, probability)

data.to_csv("../dataset/hai_dataset.csv", index=False)

print("Dataset generated successfully!")
print(data.head())
print("\nHAI distribution:")
print(data["HAI"].value_counts())