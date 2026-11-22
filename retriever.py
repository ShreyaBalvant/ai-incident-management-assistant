"""
Knowledge Base & Retrieval-Augmented Generation (RAG) Retriever Module.
Implements:
1. Knowledge base loading & validation.
2. Text preprocessing & token cleaning.
3. TF-IDF vectorization with cosine similarity.
4. Threshold-based relevance filtering (prevents passing irrelevant docs to LLM).
5. Optional Dense Semantic Retrieval (SentenceTransformers) with graceful fallback.
6. Source attribution with structured article metadata.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config


class KnowledgeRetriever:
    def __init__(self, kb_path: Optional[Path] = None):
        self.kb_path = kb_path or config.KB_PATH
        self.articles: List[Dict[str, Any]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.article_corpus: List[str] = []
        self.has_semantic: bool = False
        self.semantic_model = None
        self.semantic_embeddings = None

        self.load_and_index()

    @staticmethod
    def preprocess_text(text: str) -> str:
        """Normalize text: lowercase, remove special characters, clean extra whitespace."""
        if not text:
            return ""
        # Lowercase and replace non-alphanumeric with spaces
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        # Collapse whitespace
        return re.sub(r"\s+", " ", cleaned).strip()

    def load_and_index(self) -> None:
        """Load JSON knowledge base and compute TF-IDF indexing matrix."""
        if not self.kb_path.exists():
            raise FileNotFoundError(f"Knowledge base file not found at {self.kb_path}")

        with open(self.kb_path, "r", encoding="utf-8") as f:
            self.articles = json.load(f)

        # Build corpus by combining key fields
        self.article_corpus = []
        for a in self.articles:
            steps_joined = " ".join(a.get("troubleshooting_steps", []))
            services_joined = " ".join(a.get("applicable_service", []))
            raw_text = (
                f"{a.get('title', '')} "
                f"{a.get('category', '')} "
                f"{a.get('keywords', '')} "
                f"{a.get('content', '')} "
                f"{services_joined} "
                f"{steps_joined}"
            )
            self.article_corpus.append(self.preprocess_text(raw_text))

        # Fit TF-IDF
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.article_corpus)

        # Attempt to load SentenceTransformers optionally
        self._init_semantic_retrieval()

    def _init_semantic_retrieval(self) -> None:
        """Optional loading of sentence-transformers for semantic embeddings."""
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.semantic_embeddings = self.semantic_model.encode(
                self.article_corpus,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            self.has_semantic = True
        except ImportError:
            # Expected if sentence-transformers is not installed in the environment
            self.has_semantic = False
            self.semantic_model = None

    def retrieve_tfidf(
        self,
        query: str,
        top_k: int = config.RETRIEVAL_TOP_K,
        threshold: float = config.RETRIEVAL_SIMILARITY_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k articles using TF-IDF and Cosine Similarity.
        Enforces a minimum similarity threshold to avoid passing irrelevant noise.
        """
        if not self.vectorizer or self.tfidf_matrix is None or not self.articles:
            return []

        cleaned_query = self.preprocess_text(query)
        if not cleaned_query:
            return []

        q_vec = self.vectorizer.transform([cleaned_query])
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        # Sort descending
        top_indices = scores.argsort()[::-1]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            score = float(scores[idx])
            # Enforce similarity threshold
            if score < threshold:
                continue

            article = self.articles[idx]
            results.append(
                {
                    "article_id": article.get("article_id", f"KB-{idx}"),
                    "title": article.get("title", "Untitled"),
                    "category": article.get("category", "General IT"),
                    "content": article.get("content", ""),
                    "applicable_service": article.get("applicable_service", []),
                    "troubleshooting_steps": article.get("troubleshooting_steps", []),
                    "source_metadata": article.get("source_metadata", {}),
                    "score": round(score, 4),
                    "retrieval_method": "TF-IDF",
                }
            )
            if len(results) >= top_k:
                break

        return results

    def retrieve_semantic(
        self,
        query: str,
        top_k: int = config.RETRIEVAL_TOP_K,
        threshold: float = 0.30,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k articles using Dense Embeddings (SentenceTransformers).
        Falls back to TF-IDF if model is unavailable.
        """
        if not self.has_semantic or self.semantic_model is None:
            return self.retrieve_tfidf(query, top_k=top_k)

        cleaned_query = self.preprocess_text(query)
        if not cleaned_query:
            return []

        q_emb = self.semantic_model.encode([cleaned_query], convert_to_numpy=True)
        scores = cosine_similarity(q_emb, self.semantic_embeddings).flatten()

        top_indices = scores.argsort()[::-1]
        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < threshold:
                continue

            article = self.articles[idx]
            results.append(
                {
                    "article_id": article.get("article_id", f"KB-{idx}"),
                    "title": article.get("title", "Untitled"),
                    "category": article.get("category", "General IT"),
                    "content": article.get("content", ""),
                    "applicable_service": article.get("applicable_service", []),
                    "troubleshooting_steps": article.get("troubleshooting_steps", []),
                    "source_metadata": article.get("source_metadata", {}),
                    "score": round(score, 4),
                    "retrieval_method": "Semantic Embedding",
                }
            )
            if len(results) >= top_k:
                break

        return results

    def retrieve(
        self,
        query: str,
        top_k: int = config.RETRIEVAL_TOP_K,
        threshold: float = config.RETRIEVAL_SIMILARITY_THRESHOLD,
        mode: str = "tfidf",
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Main retrieval interface.
        Returns:
            Tuple of (list_of_articles, has_relevant_articles).
            If no article exceeds the threshold, returns ([], False).
        """
        if mode == "semantic" and self.has_semantic:
            results = self.retrieve_semantic(query, top_k=top_k, threshold=threshold)
        else:
            results = self.retrieve_tfidf(query, top_k=top_k, threshold=threshold)

        has_relevant = len(results) > 0
        return results, has_relevant

    def get_article_by_id(self, article_id: str) -> Optional[Dict[str, Any]]:
        """Lookup an article by its ID."""
        for a in self.articles:
            if a.get("article_id") == article_id:
                return a
        return None
