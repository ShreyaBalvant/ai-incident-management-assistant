"""
Deterministic Priority Assessment Engine.
Implements the ITIL Impact x Urgency Priority Matrix with deterministic overrides
for service criticality and volume of affected users.

Per Module 3 requirements:
- Matrix is configurable via config.py.
- LLM explains priority recommendations but cannot arbitrarily override business rules.
- Missing information is identified and flagged for human clarification.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import config


@dataclass
class PriorityAssessment:
    priority: str
    priority_label: str
    base_matrix_priority: str
    applied_overrides: List[str] = field(default_factory=list)
    explanation: str = ""
    missing_fields: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "priority": self.priority,
            "priority_label": self.priority_label,
            "base_matrix_priority": self.base_matrix_priority,
            "applied_overrides": self.applied_overrides,
            "explanation": self.explanation,
            "missing_fields": self.missing_fields,
        }


def assess_priority(
    business_impact: Optional[str] = None,
    urgency: Optional[str] = None,
    affected_users: Optional[int] = None,
    service_name: Optional[str] = None,
) -> PriorityAssessment:
    """
    Deterministically computes incident priority (P1-P4).

    Algorithm:
    1. Validate presence of fields; flag missing info.
    2. Normalize Impact & Urgency to High, Medium, or Low.
    3. Look up baseline priority from config.PRIORITY_MATRIX.
    4. Apply deterministic escalation overrides:
       - If affected_users >= 100 -> Escalates to P1.
       - If affected_users >= 25 and base is P3/P4 -> Escalates to P2.
       - If service is Tier 1 (Critical) and impact is High -> Escalates to P1.
       - If service is Tier 1 (Critical) and impact is Medium and base is P3/P4 -> Escalates to P2.
    5. Construct an auditable explanation.
    """
    missing_fields: List[str] = []

    # 1. Validation & Fallbacks
    impact_norm = (business_impact or "").strip().capitalize()
    if impact_norm not in config.IMPACT_LEVELS:
        missing_fields.append("business_impact (defaulted to Medium)")
        impact_norm = "Medium"

    urgency_norm = (urgency or "").strip().capitalize()
    if urgency_norm not in config.URGENCY_LEVELS:
        missing_fields.append("urgency (defaulted to Medium)")
        urgency_norm = "Medium"

    if affected_users is None or affected_users < 1:
        missing_fields.append("affected_users (defaulted to 1)")
        user_count = 1
    else:
        user_count = int(affected_users)

    service_clean = (service_name or "").strip()
    if not service_clean:
        missing_fields.append("service_name (unspecified)")
        service_info = {"criticality": "Medium", "category": "General IT"}
    else:
        service_info = config.SERVICE_CATALOG.get(
            service_clean,
            {"criticality": "Medium", "category": "General IT"},
        )

    # 2. Baseline lookup from ITIL Matrix
    base_priority = config.PRIORITY_MATRIX.get(impact_norm, {}).get(urgency_norm, "P3")
    current_priority = base_priority
    applied_overrides: List[str] = []

    # Priority ranking order for comparisons
    rank = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}

    # Helper function to upgrade priority if higher
    def upgrade_to(target_p: str, reason: str):
        nonlocal current_priority
        if rank[target_p] < rank[current_priority]:
            current_priority = target_p
            applied_overrides.append(reason)

    # 3. Deterministic Overrides
    # Override A: Large user population impact
    if user_count >= config.AFFECTED_USERS_P1_THRESHOLD:
        upgrade_to(
            "P1",
            f"Widespread impact override: {user_count} affected users exceeds P1 threshold ({config.AFFECTED_USERS_P1_THRESHOLD}).",
        )
    elif user_count >= config.AFFECTED_USERS_P2_THRESHOLD:
        upgrade_to(
            "P2",
            f"Departmental impact override: {user_count} affected users exceeds P2 threshold ({config.AFFECTED_USERS_P2_THRESHOLD}).",
        )

    # Override B: Service Criticality Tier
    service_criticality = service_info.get("criticality", "Medium")
    if service_criticality == "Critical" and impact_norm == "High":
        upgrade_to(
            "P1",
            f"Mission-critical service override: '{service_clean}' is Tier-1 Critical with High impact.",
        )
    elif service_criticality == "Critical" and impact_norm == "Medium":
        upgrade_to(
            "P2",
            f"Tier-1 service safeguard: '{service_clean}' is Tier-1 Critical with Medium impact.",
        )

    # 4. Formulate audit explanation
    label = config.PRIORITY_LEVELS.get(current_priority, f"{current_priority} - Standard")
    explanation_parts = [
        f"Baseline ITIL calculation: Impact '{impact_norm}' x Urgency '{urgency_norm}' = {base_priority}."
    ]
    if applied_overrides:
        explanation_parts.append(
            "Escalations applied: " + " ".join(applied_overrides)
        )
    else:
        explanation_parts.append("No automated override rules triggered.")

    if missing_fields:
        explanation_parts.append(
            f"Notice: Missing inputs ({', '.join(missing_fields)}) require support engineer clarification."
        )

    final_explanation = " ".join(explanation_parts)

    return PriorityAssessment(
        priority=current_priority,
        priority_label=label,
        base_matrix_priority=base_priority,
        applied_overrides=applied_overrides,
        explanation=final_explanation,
        missing_fields=missing_fields,
    )
