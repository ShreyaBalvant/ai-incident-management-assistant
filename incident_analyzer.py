"""
Central Incident Analyzer Module.
Coordinates the end-to-end incident lifecycle:
1. Intake validation & completeness checking (Module 1).
2. Deterministic priority assessment (Module 3).
3. Threshold-based knowledge retrieval (Module 4 & 5).
4. Similar historical incident discovery (Module 7).
5. RAG generation via Gemini or deterministic fallback (Module 6).
6. Local SQLite persistence (Module 9).
"""

import time
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

import config
import database
from priority_engine import PriorityAssessment, assess_priority
from rag_pipeline import IncidentAnalysisSchema, run_rag_analysis
from retriever import KnowledgeRetriever
from similar_incidents import SimilarIncidentMatcher


class IncidentIntakePayload(BaseModel):
    """Input validation model for incident submission."""

    description: str = Field(..., min_length=10, description="Detailed problem description.")
    affected_service: str = Field(default="Other / General Service")
    affected_users: int = Field(default=1, ge=1)
    business_impact: str = Field(default="Medium")
    urgency: str = Field(default="Medium")
    error_message: Optional[str] = Field(default="")


class IncidentTriageReport(BaseModel):
    """Unified triage response object delivered to the user interface."""

    incident_id: str
    analysis: IncidentAnalysisSchema
    priority_details: Dict[str, Any]
    retrieved_articles: List[Dict[str, Any]]
    has_relevant_knowledge: bool
    similar_incidents: List[Dict[str, Any]]
    analysis_mode: str
    intake_data: Dict[str, Any]


class IncidentAnalyzer:
    """Enterprise Incident Analysis Coordinator."""

    def __init__(self):
        self.retriever = KnowledgeRetriever()
        self.similar_matcher = SimilarIncidentMatcher()

    def process_incident(
        self,
        payload: IncidentIntakePayload,
        persist: bool = True,
        provider_order: Optional[List[str]] = None,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        groq_model: Optional[str] = None,
        gemini_model: Optional[str] = None,
    ) -> IncidentTriageReport:
        """
        Executes end-to-end AI incident analysis and persistence pipeline.
        Supports multi-tier provider prioritization (Groq LLaMA, Gemini, Local Fallback).
        """
        # 1. Deterministic Priority Assessment
        priority_eval: PriorityAssessment = assess_priority(
            business_impact=payload.business_impact,
            urgency=payload.urgency,
            affected_users=payload.affected_users,
            service_name=payload.affected_service,
        )

        # 2. Knowledge Base Retrieval (with relevance threshold filtering)
        retrieval_query = f"{payload.description} {payload.affected_service}"
        retrieved_docs, has_relevant = self.retriever.retrieve(
            query=retrieval_query,
            top_k=config.RETRIEVAL_TOP_K,
            threshold=config.RETRIEVAL_SIMILARITY_THRESHOLD,
        )

        # 3. Similar Incident Detection from historical tickets
        similar_matches = self.similar_matcher.find_similar(
            query=payload.description,
            top_k=config.SIMILAR_INCIDENTS_TOP_K,
            threshold=config.SIMILAR_INCIDENTS_THRESHOLD,
        )

        # 4. RAG Generation / Multi-Tier Fallback Execution
        analysis_result, mode_str = run_rag_analysis(
            description=payload.description,
            affected_service=payload.affected_service,
            affected_users=payload.affected_users,
            business_impact=payload.business_impact,
            urgency=payload.urgency,
            error_message=payload.error_message or "",
            priority_assessment=priority_eval,
            retrieved_articles=retrieved_docs,
            similar_incidents=similar_matches,
            provider_order=provider_order,
            groq_api_key=groq_api_key,
            gemini_api_key=gemini_api_key,
            groq_model=groq_model,
            gemini_model=gemini_model,
        )

        # 5. Generate unique incident ID
        incident_id = f"INC-{int(time.time())}-{uuid.uuid4().hex[:4].upper()}"

        # 6. Persist to SQLite Database
        if persist:
            record_to_save = {
                "incident_id": incident_id,
                "description": payload.description,
                "affected_service": payload.affected_service,
                "affected_users": payload.affected_users,
                "business_impact": payload.business_impact,
                "urgency": payload.urgency,
                "error_message": payload.error_message or "",
                "predicted_category": analysis_result.category,
                "recommended_priority": priority_eval.priority,
                "priority_override_reason": " | ".join(priority_eval.applied_overrides),
                "generated_summary": analysis_result.incident_summary,
                "recommended_team": analysis_result.recommended_support_team,
                "suggested_actions": analysis_result.suggested_troubleshooting_actions,
                "retrieved_article_ids": [d["article_id"] for d in retrieved_docs],
                "analysis_mode": mode_str,
                "confidence": analysis_result.confidence_score,
                "review_status": "Pending Review",
            }
            database.save_incident(record_to_save)

        return IncidentTriageReport(
            incident_id=incident_id,
            analysis=analysis_result,
            priority_details=priority_eval.to_dict(),
            retrieved_articles=retrieved_docs,
            has_relevant_knowledge=has_relevant,
            similar_incidents=similar_matches,
            analysis_mode=mode_str,
            intake_data=payload.model_dump(),
        )
