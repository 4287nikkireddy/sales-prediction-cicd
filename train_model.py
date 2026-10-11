
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("sales_data.csv")

features = [
    "Region",
    "Product_Category",
    "Quantity_Sold",
    "Unit_Price",
    "Discount"
]

X = df[features]
y = df["Sales_Amount"]

preprocessor = ColumnTransformer([
    ("category", OneHotEncoder(handle_unknown="ignore"),
     ["Region", "Product_Category"])
], remainder="passthrough")

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

metrics = {
    "mae": float(mean_absolute_error(y_test, predictions)),
    "r2": float(r2_score(y_test, predictions))
}

joblib.dump(model, "sales_model.pkl")

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Model trained successfully")
print(metrics)
