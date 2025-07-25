# EDA and Model Training Script

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

# Load Data
df = pd.read_csv("data/supply_chain_data.csv")

# Encode Target
le = LabelEncoder()
df["Inspection results"] = le.fit_transform(df["Inspection results"])  # Pass=1, Fail=0

# Encode categorical features
for col in df.select_dtypes(include='object').columns:
    if col != "Inspection results":
        df[col] = le.fit_transform(df[col])

# Feature Selection
features = [
    "Price", "Availability", "Stock levels", "Lead times", "Order quantities",
    "Shipping times", "Shipping costs", "Lead time", "Production volumes",
    "Manufacturing lead time", "Manufacturing costs", "Defect rates",
    "Transportation modes", "Routes"
]

X = df[features]
y = df["Inspection results"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluation
print(classification_report(y_test, model.predict(X_test)))

# Save model & scaler
joblib.dump(model, "models/inspection_model.pkl")
joblib.dump(scaler, "models/scaler.pkl") 