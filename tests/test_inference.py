"""Smoke tests for the published inference artifacts."""
import unittest
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]

class ClassifierSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load(ROOT / "complaint_product_classifier.pkl")
        cls.vectorizer = joblib.load(ROOT / "hashing_vectorizer.pkl")

    def test_predict_category(self):
        x = self.vectorizer.transform(["I disputed an error on my credit report."])
        prediction = self.model.predict(x)
        self.assertEqual(len(prediction), 1)
        self.assertIn(prediction[0], self.model.classes_)

    def test_nine_categories(self):
        self.assertEqual(len(self.model.classes_), 9)

    def test_decision_function_shape(self):
        x = self.vectorizer.transform(["My mortgage payment was incorrectly recorded."])
        self.assertEqual(self.model.decision_function(x).shape, (1, 9))

if __name__ == "__main__":
    unittest.main()
