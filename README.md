# 🏠 PropValue AI — House Price Predictor + EMI Calculator

A complete Machine Learning web application built with Python, Flask, and Scikit-learn.

## ✨ Features
- Predict house prices using Linear Regression (trained on 1000 properties)
- Instant EMI calculation with principal/interest breakdown
- Beautiful dark UI with animated backgrounds
- Interactive donut chart showing loan composition
- Fully responsive design

## 📁 Project Structure
```
house_price_prediction/
├── templates/
│   ├── index.html          ← Input form page
│   └── result.html         ← Results + charts page
├── static/
│   └── style.css           ← Premium dark UI styles
├── housing_dataset_1000.csv ← Dataset (1000 properties)
├── train_model.py          ← Train & save the ML model
├── emi_calculator.py       ← EMI formula module
├── app.py                  ← Flask web application
├── model.pkl               ← Saved model (generated)
├── requirements.txt        ← Python dependencies
└── README.md
```

## 🚀 How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the model
```bash
python train_model.py
```
This creates `model.pkl` and prints evaluation metrics.

### Step 3 — Start the web app
```bash
python app.py
```

### Step 4 — Open browser
Visit: **http://127.0.0.1:5000**

## 🧠 ML Model Details
- **Algorithm**: Linear Regression
- **Features**: BHK, Area (sq ft), Furnishing Status, Location Type,
  Main Road Access, Floor Number, Parking Spaces, House Age
- **Split**: 80% train / 20% test
- **Metrics**: R² Score, Accuracy (±20%), MSE, RMSE, MAE

## 💰 EMI Formula
```
EMI = P × r × (1+r)^n / ((1+r)^n - 1)
```
Where P = Principal, r = Monthly rate, n = Total months

---
Built with ❤️ using Python · Flask · Scikit-learn
