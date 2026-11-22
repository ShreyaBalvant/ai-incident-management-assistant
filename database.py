"""
Database persistence module for AI-Powered IT Incident Management Assistant.
Uses local SQLite for reliable audit logging, human review persistence, and analytics.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
import config


@contextmanager
def get_db_connection(db_path: Optional[Path] = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager that yields an SQLite connection and guarantees clean closure."""
    target_path = Path(db_path or config.DB_PATH)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize database tables if they do not exist."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id TEXT UNIQUE NOT NULL,
                description TEXT NOT NULL,
                affected_service TEXT,
                affected_users INTEGER DEFAULT 1,
                business_impact TEXT,
                urgency TEXT,
                error_message TEXT,
                predicted_category TEXT,
                recommended_priority TEXT,
                priority_override_reason TEXT,
                generated_summary TEXT,
                recommended_team TEXT,
                suggested_actions TEXT,
                retrieved_article_ids TEXT,
                analysis_mode TEXT,
                confidence INTEGER,
                review_status TEXT DEFAULT 'Pending Review',
                reviewer_category TEXT,
                reviewer_priority TEXT,
                reviewer_notes TEXT,
                created_at TEXT NOT NULL,
                reviewed_at TEXT
            )
            """
        )
        conn.commit()


def save_incident(record: Dict[str, Any], db_path: Optional[Path] = None) -> int:
    """Insert a newly analyzed incident into the database."""
    init_db(db_path)
    now_str = datetime.now().isoformat()

    actions_json = json.dumps(record.get("suggested_actions", []))
    articles_json = json.dumps(record.get("retrieved_article_ids", []))

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO incidents (
                incident_id, description, affected_service, affected_users,
                business_impact, urgency, error_message, predicted_category,
                recommended_priority, priority_override_reason, generated_summary,
                recommended_team, suggested_actions, retrieved_article_ids,
                analysis_mode, confidence, review_status, reviewer_category,
                reviewer_priority, reviewer_notes, created_at, reviewed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("incident_id"),
                record.get("description"),
                record.get("affected_service", "General Service"),
                record.get("affected_users", 1),
                record.get("business_impact", "Medium"),
                record.get("urgency", "Medium"),
                record.get("error_message", ""),
                record.get("predicted_category", "General IT"),
                record.get("recommended_priority", "P3"),
                record.get("priority_override_reason", ""),
                record.get("generated_summary", ""),
                record.get("recommended_team", "IT Service Desk Tier 1"),
                actions_json,
                articles_json,
                record.get("analysis_mode", "Local Fallback"),
                record.get("confidence", 70),
                record.get("review_status", "Pending Review"),
                record.get("reviewer_category"),
                record.get("reviewer_priority"),
                record.get("reviewer_notes", ""),
                record.get("created_at", now_str),
                record.get("reviewed_at"),
            ),
        )
        conn.commit()
        return cursor.lastrowid or 0


def update_review(
    incident_id: str,
    review_status: str,
    reviewer_category: Optional[str] = None,
    reviewer_priority: Optional[str] = None,
    reviewer_notes: str = "",
    db_path: Optional[Path] = None,
) -> bool:
    """Update human review status and edits for an incident."""
    init_db(db_path)
    now_str = datetime.now().isoformat()

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE incidents
            SET review_status = ?,
                reviewer_category = ?,
                reviewer_priority = ?,
                reviewer_notes = ?,
                reviewed_at = ?
            WHERE incident_id = ?
            """,
            (
                review_status,
                reviewer_category,
                reviewer_priority,
                reviewer_notes,
                now_str,
                incident_id,
            ),
        )
        conn.commit()
        return cursor.rowcount > 0


def get_incident(incident_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve an individual incident by incident_id."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM incidents WHERE incident_id = ?", (incident_id,))
        row = cursor.fetchone()
        if not row:
            return None
        res = dict(row)
        res["suggested_actions"] = json.loads(res["suggested_actions"] or "[]")
        res["retrieved_article_ids"] = json.loads(res["retrieved_article_ids"] or "[]")
        return res


def list_incidents(
    limit: int = 100,
    status_filter: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """List recorded incidents ordered by latest creation."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        if status_filter:
            cursor.execute(
                "SELECT * FROM incidents WHERE review_status = ? ORDER BY id DESC LIMIT ?",
                (status_filter, limit),
            )
        else:
            cursor.execute("SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (limit,))

        rows = cursor.fetchall()
        results = []
        for r in rows:
            item = dict(r)
            item["suggested_actions"] = json.loads(item["suggested_actions"] or "[]")
            item["retrieved_article_ids"] = json.loads(item["retrieved_article_ids"] or "[]")
            results.append(item)
        return results


def get_analytics_summary(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Calculate aggregated metrics for the analytics dashboard."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        # Total incidents
        cursor.execute("SELECT COUNT(*) FROM incidents")
        total = cursor.fetchone()[0]

        # Category breakdown
        cursor.execute(
            """
            SELECT COALESCE(predicted_category, 'Unclassified') as cat, COUNT(*)
            FROM incidents GROUP BY cat
            """
        )
        categories = dict(cursor.fetchall())

        # Priority breakdown
        cursor.execute(
            """
            SELECT COALESCE(recommended_priority, 'P3') as pri, COUNT(*)
            FROM incidents GROUP BY pri
            """
        )
        priorities = dict(cursor.fetchall())

        # Review status breakdown
        cursor.execute(
            """
            SELECT COALESCE(review_status, 'Pending Review') as stat, COUNT(*)
            FROM incidents GROUP BY stat
            """
        )
        review_statuses = dict(cursor.fetchall())

        # Mode breakdown
        cursor.execute(
            """
            SELECT COALESCE(analysis_mode, 'Local Fallback') as mode, COUNT(*)
            FROM incidents GROUP BY mode
            """
        )
        modes = dict(cursor.fetchall())

        # Recent daily trend
        cursor.execute(
            """
            SELECT SUBSTR(created_at, 1, 10) as day, COUNT(*)
            FROM incidents
            GROUP BY day
            ORDER BY day DESC
            LIMIT 7
            """
        )
        daily_trend = dict(cursor.fetchall())

        return {
            "total_incidents": total,
            "category_distribution": categories,
            "priority_distribution": priorities,
            "review_status_distribution": review_statuses,
            "mode_distribution": modes,
            "daily_trend": daily_trend,
        }


def seed_demo_incidents_if_empty(db_path: Optional[Path] = None) -> None:
    """Pre-seeds the database with demonstration incidents if currently empty."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM incidents")
        count = cursor.fetchone()[0]
        if count > 0:
            return

    # Seed initial representative incidents
    demo_records = [
        {
            "incident_id": "INC-2026-101",
            "description": "Users in regional office unable to access corporate VPN AnyConnect gateway since morning.",
            "affected_service": "Corporate VPN Gateway",
            "affected_users": 45,
            "business_impact": "High",
            "urgency": "High",
            "error_message": "TLS Handshake timeout: 504 Gateway Error",
            "predicted_category": "Network",
            "recommended_priority": "P1",
            "priority_override_reason": "Departmental impact: 45 users; Tier-1 Critical Service.",
            "generated_summary": "Corporate VPN gateway handshake timeouts impacting 45 remote users.",
            "recommended_team": "Network Operations Center (NOC)",
            "suggested_actions": [
                "Verify session capacity and CPU health on primary VPN concentrator.",
                "Check RADIUS and MFA authentication reachability.",
                "Review edge firewall connection drops.",
            ],
            "retrieved_article_ids": ["KB-NET-101"],
            "analysis_mode": "Local Fallback",
            "confidence": 85,
            "review_status": "Approved",
            "reviewer_category": "Network",
            "reviewer_priority": "P1",
            "reviewer_notes": "Confirmed secondary ISP failover stalled. NOC dispatched.",
            "created_at": "2026-10-29T09:30:00",
            "reviewed_at": "2026-10-29T09:45:00",
        },
        {
            "incident_id": "INC-2026-102",
            "description": "Production database connection pool exhausted; customer checkout returning HTTP 500 errors.",
            "affected_service": "Production Database Cluster",
            "affected_users": 200,
            "business_impact": "High",
            "urgency": "High",
            "error_message": "FATAL: remaining connection slots are reserved for non-replication superuser connections",
            "predicted_category": "Database",
            "recommended_priority": "P1",
            "priority_override_reason": "Widespread impact: 200 users >= 100 threshold; Tier-1 Critical Service.",
            "generated_summary": "Critical connection pool saturation on production database cluster.",
            "recommended_team": "Database Administration (DBA)",
            "suggested_actions": [
                "Inspect active connections via pg_stat_activity.",
                "Terminate idle in transaction queries.",
                "Verify PgBouncer pooler status and scale client queue.",
            ],
            "retrieved_article_ids": ["KB-DB-301"],
            "analysis_mode": "Local Fallback",
            "confidence": 90,
            "review_status": "Approved",
            "reviewer_category": "Database",
            "reviewer_priority": "P1",
            "reviewer_notes": "Leaky microservice restarted. Major incident declared.",
            "created_at": "2026-10-30T14:15:00",
            "reviewed_at": "2026-10-30T14:25:00",
        },
        {
            "incident_id": "INC-2026-103",
            "description": "User cannot sign in after failing password reset token verification.",
            "affected_service": "Identity & SSO Authentication (Active Directory)",
            "affected_users": 1,
            "business_impact": "Low",
            "urgency": "Medium",
            "error_message": "Error 401: Invalid or expired password reset token",
            "predicted_category": "Access Management",
            "recommended_priority": "P4",
            "priority_override_reason": "Baseline ITIL Low impact x Medium urgency.",
            "generated_summary": "Single user authentication lock after expired reset token.",
            "recommended_team": "Identity & Access Management (IAM)",
            "suggested_actions": [
                "Inspect Active Directory event log ID 4740 on PDC.",
                "Verify user identity via out-of-band manager callback.",
                "Unlock account and issue temporary one-time password.",
            ],
            "retrieved_article_ids": ["KB-ACC-201"],
            "analysis_mode": "Local Fallback",
            "confidence": 80,
            "review_status": "Pending Review",
            "reviewer_category": None,
            "reviewer_priority": None,
            "reviewer_notes": "",
            "created_at": "2026-10-31T08:00:00",
            "reviewed_at": None,
        },
        {
            "incident_id": "INC-2026-104",
            "description": "Entire marketing department reports corporate Outlook emails delayed by several hours.",
            "affected_service": "Corporate Email (Exchange/Outlook)",
            "affected_users": 85,
            "business_impact": "High",
            "urgency": "Medium",
            "error_message": "SMTP queue delay code 451: message deferred",
            "predicted_category": "Email Service",
            "recommended_priority": "P2",
            "priority_override_reason": "Departmental impact: 85 users >= 25 threshold.",
            "generated_summary": "Exchange mail queue delay affecting marketing staff.",
            "recommended_team": "Messaging & Collaboration Engineering",
            "suggested_actions": [
                "Check Microsoft 365 tenant service health dashboard.",
                "Review edge spam filter queue depth and delivery retry logs.",
                "Verify inbound MX record resolution and TLS connectors.",
            ],
            "retrieved_article_ids": ["KB-EML-501"],
            "analysis_mode": "Local Fallback",
            "confidence": 85,
            "review_status": "Pending Review",
            "reviewer_category": None,
            "reviewer_priority": None,
            "reviewer_notes": "",
            "created_at": "2026-10-31T10:10:00",
            "reviewed_at": None,
        },
    ]

    for rec in demo_records:
        save_incident(rec, db_path)

