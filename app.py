from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
from emi_calculator import calculate_emi

app = Flask(__name__)

# ── Load model on startup ──────────────────────────────────────
MODEL_PATH = "model.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "❌  model.pkl not found!\n"
            "    Please run: python train_model.py"
        )
    return joblib.load(MODEL_PATH)

try:
    data      = load_model()
    model     = data["model"]
    encoders  = data["encoders"]
    features  = data.get("features", [])
    metrics   = data.get("metrics", {})
    print("✅  Model loaded successfully")
except FileNotFoundError as e:
    print(e)
    model, encoders, features, metrics = None, {}, [], {}


# ── Routes ─────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", metrics=metrics)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template("index.html",
            error="Model not loaded. Run python train_model.py first.",
            metrics=metrics
        )
    try:
        # Collect form inputs
        bhk               = int(request.form["bhk"])
        area_sqft         = float(request.form["area_sqft"])
        furnishing_status = request.form["furnishing_status"]
        location_type     = request.form["location_type"]
        mainroad_access   = request.form["mainroad_access"]
        floor_number      = int(request.form["floor_number"])
        parking_spaces    = int(request.form["parking_spaces"])
        house_age         = int(request.form["house_age"])
        interest_rate     = float(request.form["interest_rate"])
        loan_tenure       = int(request.form["loan_tenure"])

        # Encode categoricals
        fs_enc = encoders["furnishing_status"].transform([furnishing_status])[0]
        lt_enc = encoders["location_type"].transform([location_type])[0]
        ma_enc = encoders["mainroad_access"].transform([mainroad_access])[0]

        # Build feature array & predict
        feat_arr = np.array([[
            bhk, area_sqft, fs_enc, lt_enc,
            ma_enc, floor_number, parking_spaces, house_age
        ]])
        predicted_price = float(model.predict(feat_arr)[0])
        predicted_price = max(predicted_price, 100000)

        # EMI
        monthly_emi, total_payment, total_interest = calculate_emi(
            predicted_price, interest_rate, loan_tenure
        )

        # Down payment estimate (20%)
        down_payment   = predicted_price * 0.20
        loan_amount    = predicted_price * 0.80

        def fmt(val):
            return f"₹{val:,.0f}"

        def fmt_cr(val):
            if val >= 1e7:
                return f"₹{val/1e7:.2f} Cr"
            elif val >= 1e5:
                return f"₹{val/1e5:.2f} L"
            return fmt(val)

        context = dict(
            # Price
            predicted_price     = fmt(predicted_price),
            predicted_price_fmt = fmt_cr(predicted_price),
            predicted_price_raw = int(predicted_price),
            # EMI
            monthly_emi         = fmt(monthly_emi),
            total_payment       = fmt(total_payment),
            total_interest      = fmt(total_interest),
            total_interest_raw  = int(total_interest),
            # Loan breakdown
            down_payment        = fmt(down_payment),
            loan_amount         = fmt(loan_amount),
            # Input echo
            interest_rate       = interest_rate,
            loan_tenure         = loan_tenure,
            bhk                 = bhk,
            area_sqft           = int(area_sqft),
            location_type       = location_type,
            furnishing_status   = furnishing_status,
            mainroad_access     = mainroad_access,
            floor_number        = floor_number,
            parking_spaces      = parking_spaces,
            house_age           = house_age,
            # Chart data (principal vs interest %)
            principal_pct       = round((predicted_price / total_payment) * 100, 1) if total_payment else 100,
            interest_pct        = round((total_interest / total_payment) * 100, 1) if total_payment else 0,
        )
        return render_template("result.html", **context)

    except Exception as e:
        return render_template("index.html",
            error=f"Prediction error: {str(e)}",
            metrics=metrics
        )


if __name__ == "__main__":
    print("\n🏠  House Price Predictor — Flask App")
    print("    http://127.0.0.1:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
