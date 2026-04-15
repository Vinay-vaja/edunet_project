"""
=============================================================
 AI Disease Prediction System - Model Training Script
 SDG 3 – Good Health and Well-being
=============================================================
 This script trains ML models on the Pima Indians Diabetes
 Dataset and saves the best performing model for prediction.
=============================================================
"""

# ── Step 1: Import Libraries ──────────────────────────────
import pandas as pd
import numpy as np
import pickle
import warnings
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore")

# ── Step 2: Load the Dataset ──────────────────────────────
print("=" * 55)
print("  AI Disease Prediction System — Training Pipeline")
print("=" * 55)

# Get the directory where train.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "diabetes.csv")

df = pd.read_csv(DATA_PATH)

print(f"\n Dataset loaded successfully!")
print(f"   → Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"   → Features: {list(df.columns[:-1])}")
print(f"   → Target: {df.columns[-1]}")

# ── Step 3: Handle Missing / Zero Values ──────────────────
# In this dataset, 0 values in certain columns are actually
# missing data. We replace them with the column median.

zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in zero_columns:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

print("\n Missing values (zeros) replaced with median values.")

# ── Step 4: Separate Features and Target ──────────────────
X = df.drop('Outcome', axis=1)
y = df['Outcome']

print(f"\n📊 Class Distribution:")
print(f"   → No Diabetes (0): {(y == 0).sum()} patients")
print(f"   → Diabetes    (1): {(y == 1).sum()} patients")

# ── Step 5: Feature Scaling ───────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Step 6: Train-Test Split (80-20) ─────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n🔀 Data Split (80/20):")
print(f"   → Training samples: {X_train.shape[0]}")
print(f"   → Testing  samples: {X_test.shape[0]}")

# ── Step 7: Train Models ─────────────────────────────────

# --- Model 1: Logistic Regression ---
print("\n" + "─" * 55)
print(" Training Model 1: Logistic Regression")
print("─" * 55)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_cm = confusion_matrix(y_test, lr_pred)

print(f"    Accuracy: {lr_accuracy * 100:.2f}%")
print(f"   Confusion Matrix:")
print(f"      {lr_cm[0]}")
print(f"      {lr_cm[1]}")
print(f"\n    Classification Report:")
print(classification_report(y_test, lr_pred, target_names=["No Diabetes", "Diabetes"]))

# --- Model 2: Decision Tree ---
print("─" * 55)
print("raining Model 2: Decision Tree")
print("─" * 55)

dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
dt_cm = confusion_matrix(y_test, dt_pred)

print(f"    Accuracy: {dt_accuracy * 100:.2f}%")
print(f"    Confusion Matrix:")
print(f"      {dt_cm[0]}")
print(f"      {dt_cm[1]}")
print(f"\n    Classification Report:")
print(classification_report(y_test, dt_pred, target_names=["No Diabetes", "Diabetes"]))

# ── Step 8: Compare Models & Save Best ───────────────────
print("=" * 55)
print("MODEL COMPARISON")
print("=" * 55)
print(f"   Logistic Regression : {lr_accuracy * 100:.2f}%")
print(f"   Decision Tree       : {dt_accuracy * 100:.2f}%")

# Select the best model
if lr_accuracy >= dt_accuracy:
    best_model = lr_model
    best_name = "Logistic Regression"
    best_accuracy = lr_accuracy
else:
    best_model = dt_model
    best_name = "Decision Tree"
    best_accuracy = dt_accuracy

print(f"\n🏆 Best Model: {best_name} ({best_accuracy * 100:.2f}%)")

# ── Step 9: Save Model and Scaler using Pickle ──────────
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(best_model, f)

with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)

print(f"\n💾 Model saved to: model.pkl")
print(f"💾 Scaler saved to: scaler.pkl")
print("\n" + "=" * 55)
print(" Training complete! Run 'streamlit run app.py' next.")
print("=" * 55)
