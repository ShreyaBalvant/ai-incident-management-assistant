"""
Unit tests for Similar Incident Detection Module.
"""

import unittest
from similar_incidents import SimilarIncidentMatcher


class TestSimilarIncidents(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.matcher = SimilarIncidentMatcher()

    def test_similar_incident_lookup(self):
        query = "Users cannot connect to VPN gateway"
        results = self.matcher.find_similar(query, top_k=2, threshold=0.10)
        self.assertGreater(len(results), 0)
        top = results[0]
        self.assertIn("incident_id", top)
        self.assertIn("resolution_notes", top)
        self.assertIn("similarity_score", top)
        self.assertIn("root_cause_caveat", top)

    def test_irrelevant_query_returns_empty(self):
        query = "gardening and planting roses in the backyard"
        results = self.matcher.find_similar(query, top_k=3, threshold=0.25)
        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()
