"""
Unit tests for SQLite database persistence and analytics.
"""

import tempfile
import unittest
from pathlib import Path
import database


class TestDatabase(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_incidents.db"
        database.init_db(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_and_get_incident(self):
        record = {
            "incident_id": "TEST-1001",
            "description": "Test outage on payment gateway",
            "affected_service": "Customer-Facing E-Commerce API",
            "affected_users": 150,
            "business_impact": "High",
            "urgency": "High",
            "predicted_category": "Application/Server",
            "recommended_priority": "P1",
            "generated_summary": "Major outage impacting payment flow.",
            "recommended_team": "Site Reliability Engineering (SRE)",
            "suggested_actions": ["Rollback canary", "Inspect 500 error logs"],
            "retrieved_article_ids": ["KB-APP-401"],
            "analysis_mode": "Local Fallback",
            "confidence": 85,
        }

        row_id = database.save_incident(record, self.db_path)
        self.assertGreater(row_id, 0)

        fetched = database.get_incident("TEST-1001", self.db_path)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["incident_id"], "TEST-1001")
        self.assertEqual(fetched["recommended_priority"], "P1")
        self.assertEqual(fetched["review_status"], "Pending Review")
        self.assertEqual(len(fetched["suggested_actions"]), 2)
        self.assertEqual(fetched["retrieved_article_ids"], ["KB-APP-401"])

    def test_human_review_update(self):
        record = {
            "incident_id": "TEST-1002",
            "description": "Password reset failure",
            "predicted_category": "General IT",
            "recommended_priority": "P3",
        }
        database.save_incident(record, self.db_path)

        success = database.update_review(
            incident_id="TEST-1002",
            review_status="Modified",
            reviewer_category="Access Management",
            reviewer_priority="P2",
            reviewer_notes="Upgraded priority due to VIP executive impact.",
            db_path=self.db_path,
        )
        self.assertTrue(success)

        updated = database.get_incident("TEST-1002", self.db_path)
        self.assertEqual(updated["review_status"], "Modified")
        self.assertEqual(updated["reviewer_category"], "Access Management")
        self.assertEqual(updated["reviewer_priority"], "P2")
        self.assertIn("VIP executive", updated["reviewer_notes"])
        self.assertIsNotNone(updated["reviewed_at"])

    def test_analytics_summary(self):
        records = [
            {"incident_id": "INC-1", "description": "D1", "predicted_category": "Network", "recommended_priority": "P1", "review_status": "Approved"},
            {"incident_id": "INC-2", "description": "D2", "predicted_category": "Database", "recommended_priority": "P2", "review_status": "Pending Review"},
            {"incident_id": "INC-3", "description": "D3", "predicted_category": "Network", "recommended_priority": "P1", "review_status": "Approved"},
        ]
        for r in records:
            database.save_incident(r, self.db_path)

        summary = database.get_analytics_summary(self.db_path)
        self.assertEqual(summary["total_incidents"], 3)
        self.assertEqual(summary["category_distribution"]["Network"], 2)
        self.assertEqual(summary["category_distribution"]["Database"], 1)
        self.assertEqual(summary["priority_distribution"]["P1"], 2)
        self.assertEqual(summary["review_status_distribution"]["Approved"], 2)


if __name__ == "__main__":
    unittest.main()
