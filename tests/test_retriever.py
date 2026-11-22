"""
Unit tests for KnowledgeRetriever and threshold filtering.
"""

import unittest
from retriever import KnowledgeRetriever


class TestKnowledgeRetriever(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.retriever = KnowledgeRetriever()

    def test_kb_loaded_successfully(self):
        self.assertGreater(len(self.retriever.articles), 0)
        self.assertIsNotNone(self.retriever.vectorizer)

    def test_relevant_query_retrieval(self):
        query = "Users cannot connect to corporate VPN AnyConnect gateway"
        results, has_relevant = self.retriever.retrieve(query, top_k=2, threshold=0.15)

        self.assertTrue(has_relevant)
        self.assertGreater(len(results), 0)
        top_article = results[0]
        self.assertEqual(top_article["category"], "Network")
        self.assertIn("VPN", top_article["title"])
        self.assertGreaterEqual(top_article["score"], 0.15)
        self.assertIn("troubleshooting_steps", top_article)

    def test_irrelevant_query_threshold_blocking(self):
        # Gibberish or irrelevant query must return empty list and has_relevant = False
        query = "baking sourdough bread with chocolate chips and strawberries"
        results, has_relevant = self.retriever.retrieve(query, top_k=3, threshold=0.20)

        self.assertFalse(has_relevant)
        self.assertEqual(len(results), 0)

    def test_source_attribution_integrity(self):
        query = "database connection pool timeout 500 error"
        results, has_relevant = self.retriever.retrieve(query, top_k=1, threshold=0.10)

        self.assertTrue(has_relevant)
        doc = results[0]
        self.assertTrue(doc["article_id"].startswith("KB-"))
        self.assertIn("source_metadata", doc)
        self.assertIn("runbook_version", doc["source_metadata"])


if __name__ == "__main__":
    unittest.main()
