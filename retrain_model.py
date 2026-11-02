"""
Retrain medisense_model.pkl on 4 features we actually track:
  1. no_show_rate       — past no-shows / past total
  2. completion_rate    — past completed / past total
  3. cancellation_rate  — past cancelled / past total
  4. total_appointments — past total

Source: the appointments table in medisense_users.db
Target: whether the patient has ever no-showed (1) or not (0)
"""
import os
import sqlite3
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# ---- Find the DB ----
BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, 'database', 'medisense_users.db')

if not os.path.exists(DB_PATH):
    print(f"❌ DB not found at {DB_PATH}")
    print("   Make sure you've run the app at least once to create it.")
    exit(1)

print(f"Using DB: {DB_PATH}")
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# ---- Get every patient and their aggregated appointment stats ----
cursor.execute('''
SELECT
    patient_email,
    COUNT(*) as total,
    SUM(CASE WHEN status = 'No-Show'   THEN 1 ELSE 0 END) as no_shows,
    SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) as completed,
    SUM(CASE WHEN status = 'Cancelled' THEN 1 ELSE 0 END) as cancelled
FROM appointments
GROUP BY patient_email
''')

rows = cursor.fetchall()
conn.close()

if not rows:
    print("❌ No appointment records in the DB — nothing to train on.")
    print("   Create some test appointments in the app first, then rerun.")
    exit(1)

# ---- Build the dataframe ----
df = pd.DataFrame([dict(r) for r in rows])

# Safety: avoid divide-by-zero
df['total'] = df['total'].replace(0, 1)

# ---- Build the 4 features ----
df['no_show_rate']      = df['no_shows']  / df['total']
df['completion_rate']   = df['completed'] / df['total']
df['cancellation_rate'] = df['cancelled'] / df['total']
df['total_appointments']= df['total']

# ---- Target: has this patient ever no-showed? ----
df['target'] = (df['no_shows'] > 0).astype(int)

print(f"\nTotal patients (samples): {len(df)}")
print(f"Target distribution:")
print(df['target'].value_counts().to_string())

# ---- Not enough data → we'll pad with synthetic samples ----
if len(df) < 20:
    print(f"\n⚠️  Only {len(df)} real samples. Padding with synthetic data for training stability.")
    rng = np.random.default_rng(42)
    synth = []
    for _ in range(200):
        total = int(rng.integers(3, 20))
        no_shows = int(rng.integers(0, total))
        completed = int(rng.integers(0, total - no_shows + 1))
        cancelled = total - no_shows - completed
        synth.append({
            'total': total,
            'no_shows': no_shows,
            'completed': completed,
            'cancelled': cancelled,
        })
    synth_df = pd.DataFrame(synth)
    synth_df['no_show_rate']      = synth_df['no_shows']  / synth_df['total']
    synth_df['completion_rate']   = synth_df['completed'] / synth_df['total']
    synth_df['cancellation_rate'] = synth_df['cancelled'] / synth_df['total']
    synth_df['total_appointments']= synth_df['total']
    synth_df['target'] = (synth_df['no_shows'] > 0).astype(int)
    df = pd.concat([df[['no_show_rate','completion_rate','cancellation_rate','total_appointments','target']],
                    synth_df[['no_show_rate','completion_rate','cancellation_rate','total_appointments','target']]],
                   ignore_index=True)

# ---- Features and target ----
FEATURES = ['no_show_rate', 'completion_rate', 'cancellation_rate', 'total_appointments']
X = df[FEATURES].values
y = df['target'].values

# ---- Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
)

# ---- Train ----
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=3,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ---- Evaluate ----
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n=== Model performance ===")
print(classification_report(y_test, y_pred, zero_division=0))
try:
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
except Exception:
    pass

# Feature importance
print("\n=== Feature importance ===")
for name, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name:22s} {imp:.4f}")

# ---- Save ----
MODELS_DIR = os.path.join(BASE, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

model_path = os.path.join(MODELS_DIR, 'medisense_model.pkl')
scaler_path = os.path.join(MODELS_DIR, 'medisense_scaler.pkl')

joblib.dump(model, model_path)
print(f"\n✅ Saved {model_path}")

# Save a dummy scaler so routes.py doesn't fail trying to load it
# (we don't scale, but the load code expects the file to exist)
from sklearn.preprocessing import StandardScaler
dummy_scaler = StandardScaler()
dummy_scaler.fit(X_train)
joblib.dump(dummy_scaler, scaler_path)
print(f"✅ Saved {scaler_path} (dummy, unused)")

# Save feature names so the app knows what order to send them
import json
with open(os.path.join(MODELS_DIR, 'medisense_model_features.json'), 'w') as f:
    json.dump(FEATURES, f, indent=2)
print(f"✅ Saved {os.path.join(MODELS_DIR, 'medisense_model_features.json')}")

print("\nDone. Restart your Flask app to use the new model.")