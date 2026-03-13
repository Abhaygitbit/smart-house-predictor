import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import warnings
warnings.filterwarnings("ignore")

print("=" * 55)
print("   HOUSE PRICE PREDICTION — MODEL TRAINING")
print("=" * 55)

# ── Load Dataset ──────────────────────────────────────────────
df = pd.read_csv("E:\PYTHON  M L\projects\house_price_prediction\housing_dataset_1000.csv")
print(f"\n✅  Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n{'Column':<22} {'Dtype':<15} {'Nulls'}")
print("-" * 45)
for col in df.columns:
    print(f"  {col:<20} {str(df[col].dtype):<15} {df[col].isnull().sum()}")

# ── Features & Target ─────────────────────────────────────────
FEATURES = [
    "bhk", "area_sqft", "furnishing_status",
    "location_type", "mainroad_access",
    "floor_number", "parking_spaces", "house_age"
]
TARGET = "price"

X = df[FEATURES].copy()
y = df[TARGET]

# ── Encode Categorical Variables ──────────────────────────────
CATEGORICAL = ["furnishing_status", "location_type", "mainroad_access"]
label_encoders = {}

print("\n📦  Encoding categorical features:")
for col in CATEGORICAL:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    mapping = dict(zip(le.classes_, le.transform(le.classes_).tolist()))
    print(f"  {col}: {mapping}")

# ── Train / Test Split (80/20) ────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n📊  Split → Train: {len(X_train)} | Test: {len(X_test)}")

# ── Train Linear Regression ───────────────────────────────────
print("\n🧠  Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)
print("    Done!")

# ── Evaluate ──────────────────────────────────────────────────
y_pred = model.predict(X_test)

r2   = r2_score(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = np.mean(np.abs(y_test - y_pred))

# Accuracy: % predictions within ±20% of actual
accuracy = np.mean(np.abs((y_test - y_pred) / y_test) <= 0.20) * 100

print("\n" + "=" * 55)
print("              MODEL EVALUATION RESULTS")
print("=" * 55)
print(f"  R² Score         : {r2:.4f}  ({r2*100:.2f}%)")
print(f"  Accuracy (±20%)  : {accuracy:.2f}%")
print(f"  MSE              : {mse:>15,.2f}")
print(f"  RMSE             : {rmse:>15,.2f}")
print(f"  MAE              : {mae:>15,.2f}")
print("=" * 55)

# Feature importance (coefficients)
print("\n📈  Feature Coefficients:")
for feat, coef in zip(FEATURES, model.coef_):
    print(f"  {feat:<25} {coef:>+15,.2f}")
print(f"  {'Intercept':<25} {model.intercept_:>+15,.2f}")

# ── Save Model ────────────────────────────────────────────────
joblib.dump({
    "model": model,
    "encoders": label_encoders,
    "features": FEATURES,
    "metrics": {
        "r2": round(r2, 4),
        "accuracy": round(accuracy, 2),
        "rmse": round(rmse, 2),
        "mae": round(mae, 2)
    }
}, "model.pkl")

print("\n✅  Model saved → model.pkl")
print("\n🚀  Run 'python app.py' to start the web application!")
print("=" * 55)
