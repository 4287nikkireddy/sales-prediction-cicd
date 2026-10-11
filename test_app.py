
import unittest
from app import app

class TestSalesApp(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_prediction(self):
        response = self.client.post("/predict", json={
            "Region": "North",
            "Product_Category": "Furniture",
            "Quantity_Sold": 18,
            "Unit_Price": 267.22,
            "Discount": 0.09
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            "predicted_sales_amount",
            response.get_json()
        )

    def test_missing_fields(self):
        response = self.client.post(
            "/predict", json={"Region": "North"}
        )
        self.assertEqual(response.status_code, 400)

if __name__ == "__main__":
    unittest.main()
