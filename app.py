
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

FEATURES = [
    "Region",
    "Product_Category",
    "Quantity_Sold",
    "Unit_Price",
    "Discount"
]

@app.get("/")
def health():
    return jsonify({
        "status": "ok",
        "service": "sales-prediction"
    })

@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "JSON object required"}), 400

    missing = [f for f in FEATURES if f not in data]

    if missing:
        return jsonify({
            "error": "Missing fields",
            "missing_fields": missing
        }), 400

    try:
        sample = pd.DataFrame([data])[FEATURES]
        model = joblib.load("sales_model.pkl")
        prediction = float(model.predict(sample)[0])
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({
        "predicted_sales_amount": round(prediction, 2)
    })

if __name__ == "__main__":
    app.run(port=5000)
