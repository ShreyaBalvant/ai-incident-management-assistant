"""
Unit tests for the RAG Pipeline and Multi-Tier Fallback System:
- Priority 1: Groq LLaMA
- Priority 2: Google Gemini
- Priority 3: Local Deterministic Rule-Based Fallback
"""

import json
import unittest
from unittest.mock import MagicMock, patch

import config
from priority_engine import assess_priority
from rag_pipeline import (
    IncidentAnalysisSchema,
    build_rag_prompt,
    generate_local_fallback,
    parse_and_validate_schema,
    run_rag_analysis,
)


class TestRAGPipelineFallback(unittest.TestCase):

    def setUp(self):
        self.priority_eval = assess_priority(
            business_impact="High",
            urgency="High",
            affected_users=150,
            service_name="Production Database Cluster",
        )
        self.sample_kb = [
            {
                "article_id": "KB-DB-301",
                "title": "PostgreSQL Connection Pool Saturation",
                "category": "Database",
                "content": "Remediation guide for pool exhaustion.",
                "troubleshooting_steps": [
                    "Inspect pg_stat_activity.",
                    "Terminate idle transactions.",
                ],
            }
        ]

    def test_schema_parsing_clean_json(self):
        raw = json.dumps(
            {
                "category": "Database",
                "priority_recommendation": "P1 - Critical",
                "incident_summary": "Database pool saturated.",
                "business_impact": "Severe disruption.",
                "recommended_support_team": "Database Administration (DBA)",
                "suggested_troubleshooting_actions": ["Restart pooler"],
                "supporting_knowledge_article_ids": ["KB-DB-301"],
                "missing_information": [],
                "limitations_or_uncertainty": "Decision support only.",
                "confidence_score": 90,
            }
        )
        parsed = parse_and_validate_schema(raw)
        self.assertIsInstance(parsed, IncidentAnalysisSchema)
        self.assertEqual(parsed.category, "Database")
        self.assertEqual(parsed.confidence_score, 90)

    def test_schema_parsing_markdown_wrapped_json(self):
        raw = """```json
        {
            "category": "Network",
            "priority_recommendation": "P2 - High",
            "incident_summary": "VPN gateway timeout.",
            "business_impact": "Remote users unable to connect.",
            "recommended_support_team": "Network Operations Center (NOC)",
            "suggested_troubleshooting_actions": ["Check RADIUS auth"],
            "supporting_knowledge_article_ids": ["KB-NET-101"],
            "missing_information": ["Client OS versions"],
            "limitations_or_uncertainty": "Advisory only.",
            "confidence_score": 85
        }
        ```"""
        parsed = parse_and_validate_schema(raw)
        self.assertIsInstance(parsed, IncidentAnalysisSchema)
        self.assertEqual(parsed.category, "Network")

    def test_local_fallback_deterministic(self):
        fallback = generate_local_fallback(
            description="Active Directory password lockout after multiple attempts",
            affected_service="Identity & SSO Authentication (Active Directory)",
            affected_users=5,
            business_impact="Medium",
            urgency="High",
            error_message="Event ID 4740",
            priority_assessment=self.priority_eval,
            retrieved_articles=[],
            similar_incidents=[],
        )
        self.assertEqual(fallback.category, "Access Management")
        self.assertEqual(fallback.recommended_support_team, "Identity & Access Management (IAM)")
        self.assertGreater(len(fallback.suggested_troubleshooting_actions), 0)

    def test_no_keys_triggers_local_fallback(self):
        result, mode = run_rag_analysis(
            description="Users report slow SQL query response times",
            affected_service="Production Database Cluster",
            affected_users=150,
            business_impact="High",
            urgency="High",
            error_message="",
            priority_assessment=self.priority_eval,
            retrieved_articles=self.sample_kb,
            similar_incidents=[],
            groq_api_key="your_api_key_here",
            gemini_api_key="your_api_key_here",
        )
        self.assertIn("Local Fallback", mode)
        self.assertEqual(result.category, "Database")
        self.assertIn("P1", result.priority_recommendation)

    @patch("rag_pipeline.call_groq_llm")
    def test_groq_priority_1_success(self, mock_groq):
        mock_groq.return_value = IncidentAnalysisSchema(
            category="Database",
            priority_recommendation="P1 - Critical",
            incident_summary="Groq synthesized triage summary.",
            business_impact="Production payment processing halted.",
            recommended_support_team="Database Administration (DBA)",
            suggested_troubleshooting_actions=["Scale read replica", "Kill runaway queries"],
            supporting_knowledge_article_ids=["KB-DB-301"],
            missing_information=[],
            limitations_or_uncertainty="AI decision support.",
            confidence_score=95,
        )

        result, mode = run_rag_analysis(
            description="Database connections maxed out",
            affected_service="Production Database Cluster",
            affected_users=150,
            business_impact="High",
            urgency="High",
            error_message="FATAL: connection pool full",
            priority_assessment=self.priority_eval,
            retrieved_articles=self.sample_kb,
            similar_incidents=[],
            provider_order=["groq", "gemini"],
            groq_api_key="gsk_test_valid_key_123456789",
        )

        self.assertTrue(mock_groq.called)
        self.assertIn("Groq LLaMA", mode)
        self.assertEqual(result.category, "Database")
        self.assertEqual(result.confidence_score, 95)

    @patch("rag_pipeline.call_gemini_llm")
    @patch("rag_pipeline.call_groq_llm")
    def test_fallback_to_gemini_when_groq_fails(self, mock_groq, mock_gemini):
        mock_groq.side_effect = Exception("Groq rate limit 429 exceeded")
        mock_gemini.return_value = IncidentAnalysisSchema(
            category="Database",
            priority_recommendation="P1 - Critical",
            incident_summary="Gemini synthesized triage summary after Groq failure.",
            business_impact="Database impact.",
            recommended_support_team="Database Administration (DBA)",
            suggested_troubleshooting_actions=["Restart connection pooler"],
            supporting_knowledge_article_ids=["KB-DB-301"],
            missing_information=[],
            limitations_or_uncertainty="AI decision support.",
            confidence_score=88,
        )

        result, mode = run_rag_analysis(
            description="Database connections maxed out",
            affected_service="Production Database Cluster",
            affected_users=150,
            business_impact="High",
            urgency="High",
            error_message="FATAL: connection pool full",
            priority_assessment=self.priority_eval,
            retrieved_articles=self.sample_kb,
            similar_incidents=[],
            provider_order=["groq", "gemini"],
            groq_api_key="gsk_test_valid_key_123456789",
            gemini_api_key="AIzaSy_test_valid_gemini_key_123456",
        )

        self.assertTrue(mock_groq.called)
        self.assertTrue(mock_gemini.called)
        self.assertIn("Gemini API", mode)
        self.assertIn("Fallback", mode)
        self.assertEqual(result.category, "Database")

    @patch("rag_pipeline.call_gemini_llm")
    @patch("rag_pipeline.call_groq_llm")
    def test_fallback_to_local_when_both_cloud_fail(self, mock_groq, mock_gemini):
        mock_groq.side_effect = Exception("Groq 503 service unavailable")
        mock_gemini.side_effect = Exception("Gemini QuotaExceeded payment required")

        result, mode = run_rag_analysis(
            description="Core network switch dropping packets on port 443",
            affected_service="Core Network Switches / Wi-Fi",
            affected_users=150,
            business_impact="High",
            urgency="High",
            error_message="Packet loss 80%",
            priority_assessment=self.priority_eval,
            retrieved_articles=self.sample_kb,
            similar_incidents=[],
            provider_order=["groq", "gemini"],
            groq_api_key="gsk_test_valid_key_123456789",
            gemini_api_key="AIzaSy_test_valid_gemini_key_123456",
        )

        self.assertIn("Local Fallback", mode)
        self.assertEqual(result.category, "Network")
        self.assertIn("P1", result.priority_recommendation)


if __name__ == "__main__":
    unittest.main()
