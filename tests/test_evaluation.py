"""
Unit tests for the Evaluation harness.
"""

import unittest
from evaluation import EvaluationHarness


class TestEvaluation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.harness = EvaluationHarness()

    def test_load_dataset(self):
        df = self.harness.load_dataset()
        self.assertFalse(df.empty)
        self.assertIn("query_text", df.columns)
        self.assertIn("relevant_article_ids", df.columns)
        self.assertIn("expected_category", df.columns)

    def test_evaluate_retrieval(self):
        ret = self.harness.evaluate_retrieval(top_k=3)
        self.assertIn("mean_precision_at_k", ret)
        self.assertIn("mean_recall_at_k", ret)
        self.assertGreater(ret["mean_recall_at_k"], 0.5)

    def test_evaluate_classification_and_latency(self):
        clf = self.harness.evaluate_classification_and_latency()
        self.assertIn("category_accuracy", clf)
        self.assertIn("macro_f1", clf)
        self.assertIn("average_latency_ms", clf)
        self.assertGreater(clf["category_accuracy"], 0.7)


if __name__ == "__main__":
    unittest.main()
