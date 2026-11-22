"""
Unit tests for deterministic Priority Assessment Engine.
"""

import unittest
from priority_engine import assess_priority


class TestPriorityEngine(unittest.TestCase):

    def test_standard_itil_matrix(self):
        # High Impact x High Urgency -> P1
        res = assess_priority(business_impact="High", urgency="High", affected_users=5)
        self.assertEqual(res.priority, "P1")
        self.assertEqual(res.base_matrix_priority, "P1")

        # Medium Impact x Medium Urgency -> P3
        res = assess_priority(business_impact="Medium", urgency="Medium", affected_users=5)
        self.assertEqual(res.priority, "P3")

        # Low Impact x Low Urgency -> P4
        res = assess_priority(business_impact="Low", urgency="Low", affected_users=1)
        self.assertEqual(res.priority, "P4")

    def test_affected_users_escalation(self):
        # Base matrix is P3 (Medium x Medium), but 120 users -> P1
        res = assess_priority(business_impact="Medium", urgency="Medium", affected_users=120)
        self.assertEqual(res.priority, "P1")
        self.assertEqual(res.base_matrix_priority, "P3")
        self.assertTrue(any("Widespread impact" in o for o in res.applied_overrides))

        # Base matrix is P4 (Low x Low), but 30 users -> P2
        res = assess_priority(business_impact="Low", urgency="Low", affected_users=30)
        self.assertEqual(res.priority, "P2")
        self.assertTrue(any("Departmental impact" in o for o in res.applied_overrides))

    def test_service_criticality_escalation(self):
        # Production Database Cluster is Tier 1 Critical. High impact + Medium Urgency (base P2) -> P1
        res = assess_priority(
            business_impact="High",
            urgency="Medium",
            affected_users=10,
            service_name="Production Database Cluster",
        )
        self.assertEqual(res.priority, "P1")
        self.assertTrue(any("Tier-1 Critical" in o for o in res.applied_overrides))

    def test_missing_information_handling(self):
        # Missing impact and urgency -> default to Medium with warnings
        res = assess_priority(business_impact=None, urgency=None, affected_users=None)
        self.assertEqual(res.priority, "P3")
        self.assertTrue(len(res.missing_fields) > 0)
        self.assertIn("clarification", res.explanation.lower())


if __name__ == "__main__":
    unittest.main()
