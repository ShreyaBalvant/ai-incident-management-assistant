"""
Evaluation & Benchmarking Module (Module 10).
Performs reproducible, empirical evaluation on labeled ground-truth dataset (data/evaluation_queries.csv):
1. Retrieval Metrics: Precision@K and Recall@K.
2. Classification Metrics: Accuracy, Precision, Recall, Macro F1, Weighted F1 (via scikit-learn).
3. Deterministic Priority Agreement Rate.
4. Latency Benchmarking (mean, min, max response latency).
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from sklearn.metrics import classification_report

import config
from incident_analyzer import IncidentAnalyzer, IncidentIntakePayload
from retriever import KnowledgeRetriever


class EvaluationHarness:
    """Automated benchmark harness for AI Incident Assistant evaluation."""

    def __init__(self, eval_path: Optional[Path] = None):
        self.eval_path = eval_path or config.EVAL_PATH
        self.retriever = KnowledgeRetriever()
        self.analyzer = IncidentAnalyzer()

    def load_dataset(self) -> pd.DataFrame:
        """Load and validate evaluation dataset."""
        if not self.eval_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found at {self.eval_path}")
        return pd.read_csv(self.eval_path)

    def evaluate_retrieval(
        self,
        top_k: int = 3,
        threshold: float = config.RETRIEVAL_SIMILARITY_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Calculates Precision@K and Recall@K against labeled relevant articles.
        """
        df = self.load_dataset()
        precision_scores: List[float] = []
        recall_scores: List[float] = []
        detailed_rows: List[Dict[str, Any]] = []

        for _, row in df.iterrows():
            q_text = str(row["query_text"])
            expected_ids = [
                x.strip() for x in str(row["relevant_article_ids"]).split(",") if x.strip()
            ]

            results, _ = self.retriever.retrieve(
                query=q_text,
                top_k=top_k,
                threshold=threshold,
            )
            retrieved_ids = [r["article_id"] for r in results]

            hits = [r_id for r_id in retrieved_ids if r_id in expected_ids]
            p_at_k = len(hits) / top_k if top_k > 0 else 0.0
            r_at_k = len(hits) / len(expected_ids) if expected_ids else 0.0

            precision_scores.append(p_at_k)
            recall_scores.append(r_at_k)

            detailed_rows.append(
                {
                    "query_id": row.get("query_id"),
                    "query_text": q_text[:60] + "...",
                    "expected_kb_ids": expected_ids,
                    "retrieved_kb_ids": retrieved_ids,
                    "hits": len(hits),
                    "precision_at_k": round(p_at_k, 3),
                    "recall_at_k": round(r_at_k, 3),
                }
            )

        mean_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
        mean_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0

        return {
            "top_k": top_k,
            "threshold": threshold,
            "total_queries": len(df),
            "mean_precision_at_k": round(mean_precision, 4),
            "mean_recall_at_k": round(mean_recall, 4),
            "details": detailed_rows,
        }

    def evaluate_classification_and_latency(self) -> Dict[str, Any]:
        """
        Evaluates category classification, priority alignment, and latency.
        """
        df = self.load_dataset()
        y_true_cat: List[str] = []
        y_pred_cat: List[str] = []
        priority_matches: int = 0
        latencies_ms: List[float] = []
        details: List[Dict[str, Any]] = []

        for _, row in df.iterrows():
            q_text = str(row["query_text"])
            exp_cat = str(row["expected_category"])
            exp_pri = str(row["expected_priority"])
            exp_svc = str(row.get("expected_service", "Other / General Service"))

            payload = IncidentIntakePayload(
                description=q_text,
                affected_service=exp_svc,
                affected_users=10,
                business_impact="Medium",
                urgency="Medium",
            )

            start_t = time.perf_counter()
            report = self.analyzer.process_incident(payload, persist=False)
            latency_ms = (time.perf_counter() - start_t) * 1000.0
            latencies_ms.append(latency_ms)

            pred_cat = report.analysis.category
            pred_pri = report.priority_details.get("priority", "P3")

            y_true_cat.append(exp_cat)
            y_pred_cat.append(pred_cat)

            pri_match = pred_pri == exp_pri
            if pri_match:
                priority_matches += 1

            details.append(
                {
                    "query_id": row.get("query_id"),
                    "query_text": q_text[:50] + "...",
                    "expected_category": exp_cat,
                    "predicted_category": pred_cat,
                    "category_match": pred_cat == exp_cat,
                    "expected_priority": exp_pri,
                    "predicted_priority": pred_pri,
                    "priority_match": pri_match,
                    "latency_ms": round(latency_ms, 2),
                }
            )

        report_dict = classification_report(
            y_true_cat,
            y_pred_cat,
            output_dict=True,
            zero_division=0,
        )

        total = len(df)
        cat_accuracy = sum(1 for t, p in zip(y_true_cat, y_pred_cat) if t == p) / total if total else 0.0
        pri_accuracy = priority_matches / total if total else 0.0

        avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
        p95_latency = sorted(latencies_ms)[int(0.95 * len(latencies_ms))] if latencies_ms else 0.0

        return {
            "total_queries": total,
            "category_accuracy": round(cat_accuracy, 4),
            "macro_f1": round(report_dict.get("macro avg", {}).get("f1-score", 0.0), 4),
            "weighted_f1": round(report_dict.get("weighted avg", {}).get("f1-score", 0.0), 4),
            "priority_agreement_rate": round(pri_accuracy, 4),
            "average_latency_ms": round(avg_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "classification_report": report_dict,
            "details": details,
        }


def run_full_evaluation() -> Dict[str, Any]:
    """Execute both retrieval and classification benchmarks."""
    harness = EvaluationHarness()
    retrieval_res = harness.evaluate_retrieval(top_k=3)
    clf_res = harness.evaluate_classification_and_latency()
    return {
        "retrieval": retrieval_res,
        "classification_and_latency": clf_res,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Running AI Incident Management Assistant Evaluation Harness")
    print("=" * 60)
    results = run_full_evaluation()

    ret = results["retrieval"]
    print(f"\n[Retrieval Benchmark - Top K={ret['top_k']}, Threshold={ret['threshold']}]")
    print(f"Total Test Queries: {ret['total_queries']}")
    print(f"Mean Precision@{ret['top_k']}: {ret['mean_precision_at_k'] * 100:.2f}%")
    print(f"Mean Recall@{ret['top_k']}: {ret['mean_recall_at_k'] * 100:.2f}%")

    clf = results["classification_and_latency"]
    print(f"\n[Classification & Latency Benchmark]")
    print(f"Category Classification Accuracy: {clf['category_accuracy'] * 100:.2f}%")
    print(f"Category Macro F1-Score: {clf['macro_f1']:.4f}")
    print(f"Category Weighted F1-Score: {clf['weighted_f1']:.4f}")
    print(f"Deterministic Priority Agreement: {clf['priority_agreement_rate'] * 100:.2f}%")
    print(f"Average Response Latency: {clf['average_latency_ms']} ms")
    print(f"95th Percentile Latency: {clf['p95_latency_ms']} ms")
    print("=" * 60)
