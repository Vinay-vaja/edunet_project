"""
=============================================================
 AI Disease Prediction System - Model Training Script
 SDG 3 – Good Health and Well-being
=============================================================
"""

import pandas as pd
import numpy as np
import pickle
import warnings
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

warnings.filterwarnings("ignore")

print("=" * 55)
print("  AI Disease Prediction System — Training Pipeline")
print("=" * 55)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "diabetes.csv")

df = pd.read_csv(DATA_PATH)
print(f"\n✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ── Replace zero values with median (they are missing data) ──
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    df[col] = df[col].replace(0, np.nan).fillna(df[col].median())

# ── Features & Target ────────────────────────────────────
X = df.drop('Outcome', axis=1)
y = df['Outcome']

print(f"\n📊 Class Distribution:")
print(f"   → No Diabetes (0): {(y == 0).sum()}")
print(f"   → Diabetes    (1): {(y == 1).sum()}")

# ── Scale — fit on numpy array so no feature name mismatch ──
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values)  # numpy array, no column names

# ── Train / Test split ───────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n🔀 Split → Train: {len(X_train)}, Test: {len(X_test)}")

# ── Train individual models ───────────────────────────────
print("\n" + "─" * 55)

# Logistic Regression — well calibrated by default
lr = LogisticRegression(C=0.5, max_iter=2000, random_state=42)
lr.fit(X_train, y_train)
lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:,1])
print(f"  Logistic Regression  — AUC: {lr_auc:.4f}  Acc: {accuracy_score(y_test, lr.predict(X_test))*100:.1f}%")

# Random Forest calibrated with isotonic regression
rf_base = RandomForestClassifier(n_estimators=300, max_depth=6,
                                  min_samples_leaf=8, class_weight='balanced',
                                  random_state=42)
rf = CalibratedClassifierCV(rf_base, cv=5, method='isotonic')
rf.fit(X_train, y_train)
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
print(f"  Random Forest (cal.) — AUC: {rf_auc:.4f}  Acc: {accuracy_score(y_test, rf.predict(X_test))*100:.1f}%")

# Gradient Boosting calibrated
gb_base = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                      learning_rate=0.05, subsample=0.8,
                                      random_state=42)
gb = CalibratedClassifierCV(gb_base, cv=5, method='isotonic')
gb.fit(X_train, y_train)
gb_auc = roc_auc_score(y_test, gb.predict_proba(X_test)[:,1])
print(f"  Gradient Boosting    — AUC: {gb_auc:.4f}  Acc: {accuracy_score(y_test, gb.predict(X_test))*100:.1f}%")

# ── Pick best model by AUC ────────────────────────────────
candidates = [
    ("Logistic Regression", lr,  lr_auc),
    ("Random Forest",       rf,  rf_auc),
    ("Gradient Boosting",   gb,  gb_auc),
]
best_name, best_model, best_auc = max(candidates, key=lambda x: x[2])
best_acc = accuracy_score(y_test, best_model.predict(X_test))

print("\n" + "=" * 55)
print(f"🏆 Best: {best_name}  AUC: {best_auc:.4f}  Acc: {best_acc*100:.1f}%")
print(classification_report(y_test, best_model.predict(X_test),
                             target_names=["No Diabetes", "Diabetes"]))
print("=" * 55)

MODEL_PATH  = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(best_model, f)
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)

print(f"\n💾 model.pkl  saved")
print(f"💾 scaler.pkl saved")
print("\n✅ Training complete! Run: streamlit run app.py")
print("=" * 55)
