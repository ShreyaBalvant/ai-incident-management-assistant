"""
Unit tests for the end-to-end Incident Analyzer module.
"""

import unittest
from incident_analyzer import IncidentAnalyzer, IncidentIntakePayload


class TestIncidentAnalyzer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.analyzer = IncidentAnalyzer()

    def test_end_to_end_network_incident(self):
        payload = IncidentIntakePayload(
            description="Users report VPN connectivity failure with TLS handshake timeout.",
            affected_service="Corporate VPN Gateway",
            affected_users=35,
            business_impact="Medium",
            urgency="High",
            error_message="TLS Error: Handshake timed out",
        )

        report = self.analyzer.process_incident(payload, persist=False)

        # Check intake reflection
        self.assertEqual(report.intake_data["affected_users"], 35)

        # Check category classification
        self.assertEqual(report.analysis.category, "Network")

        # Check deterministic priority (35 users >= 25 -> P2)
        self.assertIn("P2", report.priority_details["priority"])

        # Check knowledge retrieval
        self.assertTrue(report.has_relevant_knowledge)
        self.assertGreater(len(report.retrieved_articles), 0)
        self.assertIn("KB-NET-101", report.analysis.supporting_knowledge_article_ids)

        # Check suggested actions
        self.assertGreater(len(report.analysis.suggested_troubleshooting_actions), 0)

        # Check confidence
        self.assertGreaterEqual(report.analysis.confidence_score, 60)

    def test_database_incident(self):
        payload = IncidentIntakePayload(
            description="Production database connection pool exhausted returning 500 errors.",
            affected_service="Production Database Cluster",
            affected_users=150,
            business_impact="High",
            urgency="High",
            error_message="FATAL: remaining connection slots are reserved for non-replication superuser connections",
        )

        report = self.analyzer.process_incident(payload, persist=False)
        self.assertEqual(report.analysis.category, "Database")
        # 150 users + Tier 1 Critical -> P1
        self.assertEqual(report.priority_details["priority"], "P1")
        self.assertTrue(report.has_relevant_knowledge)


if __name__ == "__main__":
    unittest.main()
