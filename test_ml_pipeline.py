
import unittest
from pathlib import Path
import joblib

class TestMLPipeline(unittest.TestCase):

    def test_model_exists(self):
        self.assertTrue(Path("sales_model.pkl").exists())

    def test_model_loads(self):
        model = joblib.load("sales_model.pkl")
        self.assertTrue(hasattr(model, "predict"))

if __name__ == "__main__":
    unittest.main()
