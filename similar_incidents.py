"""
Similar Incident Detection Module.
Uses TF-IDF vectorization and cosine similarity over historical resolved tickets
(data/sample_incidents.csv) to discover past incidents with related symptoms.

Important Guardrail:
Similarity indicates symptom correlation and prior resolution reference,
NOT a verified common root cause.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from retriever import KnowledgeRetriever


class SimilarIncidentMatcher:
    def __init__(self, incidents_path: Optional[Path] = None):
        self.incidents_path = incidents_path or config.INCIDENTS_PATH
        self.df: pd.DataFrame = pd.DataFrame()
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.load_and_index()

    def load_and_index(self) -> None:
        """Load historical incidents and fit TF-IDF matrix."""
        if not self.incidents_path.exists():
            return

        self.df = pd.read_csv(self.incidents_path)
        if self.df.empty or "description" not in self.df.columns:
            return

        # Build corpus using cleaned description + affected service
        corpus = []
        for _, row in self.df.iterrows():
            desc = str(row.get("description", ""))
            svc = str(row.get("affected_service", ""))
            text = f"{desc} {svc}"
            corpus.append(KnowledgeRetriever.preprocess_text(text))

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def find_similar(
        self,
        query: str,
        top_k: int = config.SIMILAR_INCIDENTS_TOP_K,
        threshold: float = config.SIMILAR_INCIDENTS_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        """
        Find historical incidents matching query above similarity threshold.
        Returns empty list if no matches exceed threshold.
        """
        if self.df.empty or self.vectorizer is None or self.tfidf_matrix is None:
            return []

        cleaned_query = KnowledgeRetriever.preprocess_text(query)
        if not cleaned_query:
            return []

        q_vec = self.vectorizer.transform([cleaned_query])
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        top_indices = scores.argsort()[::-1]
        results: List[Dict[str, Any]] = []

        for idx in top_indices:
            score = float(scores[idx])
            if score < threshold:
                continue

            row = self.df.iloc[idx]
            results.append(
                {
                    "incident_id": str(row.get("incident_id", f"HIST-{idx}")),
                    "description": str(row.get("description", "")),
                    "category": str(row.get("category", "General IT")),
                    "priority": str(row.get("priority", "P3")),
                    "affected_service": str(row.get("affected_service", "Unspecified")),
                    "resolution_notes": str(row.get("resolution_notes", "No resolution recorded.")),
                    "similarity_score": round(score, 4),
                    "root_cause_caveat": "Correlation reference only; does not prove identical root cause.",
                }
            )
            if len(results) >= top_k:
                break

        return results
