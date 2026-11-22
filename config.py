"""
Configuration module for AI-Powered IT Incident Management Assistant.
Contains centralized settings for models, fallback hierarchies, thresholds,
category taxonomy, ITIL priority matrix, service criticality mappings, and file paths.
"""

import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
TESTS_DIR = BASE_DIR / "tests"

# File paths (customizable via environment variable for cloud volumes)
KB_PATH = DATA_DIR / "knowledge_base.json"
INCIDENTS_PATH = DATA_DIR / "sample_incidents.csv"
EVAL_PATH = DATA_DIR / "evaluation_queries.csv"
DB_PATH = Path(os.getenv("DB_PATH", str(BASE_DIR / "incidents.db")))

# ------------------------------------------------------------------------------
# LLM Configuration & Multi-Tier Fallback Hierarchy
# ------------------------------------------------------------------------------
# Priority 1: Groq LLaMA (Fast, generous free tier, open-source models)
# Priority 2: Google Gemini (Secondary cloud fallback)
# Priority 3: Local Deterministic Rule-Based Fallback (100% uptime guaranteed)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# Provider order: "groq,gemini" puts Groq LLaMA on Priority 1!
_raw_order = os.getenv("LLM_PROVIDER_ORDER", "groq,gemini")
DEFAULT_PROVIDER_ORDER: List[str] = [
    p.strip().lower() for p in _raw_order.split(",") if p.strip()
]

# Supported models list for UI / configuration
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
]

GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

LLM_TEMPERATURE = 0.2
LLM_TIMEOUT_SECONDS = 15

# Placeholder strings that indicate an unconfigured API key
KEY_PLACEHOLDERS = {
    "",
    "your_api_key_here",
    "your_actual_api_key_here",
    "your_gemini_api_key_here",
    "your_groq_api_key_here",
    "gsk_your_key_here",
    "none",
}


def is_valid_key(key: Optional[str]) -> bool:
    """Checks whether an API key is set and is not a default placeholder string."""
    if not key:
        return False
    stripped = key.strip()
    return stripped.lower() not in KEY_PLACEHOLDERS and len(stripped) > 8


# ------------------------------------------------------------------------------
# Incident Taxonomy & ITIL Settings
# ------------------------------------------------------------------------------

# Supported Incident Categories
CATEGORIES = [
    "Network",
    "Access Management",
    "Database",
    "Application/Server",
    "Email Service",
    "Software",
    "Security",
    "General IT",
]

# Standard Priority Levels
PRIORITY_LEVELS = {
    "P1": "P1 - Critical",
    "P2": "P2 - High",
    "P3": "P3 - Medium",
    "P4": "P4 - Low",
}

# Impact Levels
IMPACT_LEVELS = ["High", "Medium", "Low"]

# Urgency Levels
URGENCY_LEVELS = ["High", "Medium", "Low"]

# ITIL Standard 3x3 Impact x Urgency Matrix
# Structure: PRIORITY_MATRIX[impact][urgency] -> "P1" | "P2" | "P3" | "P4"
PRIORITY_MATRIX = {
    "High": {
        "High": "P1",
        "Medium": "P2",
        "Low": "P3",
    },
    "Medium": {
        "High": "P2",
        "Medium": "P3",
        "Low": "P4",
    },
    "Low": {
        "High": "P3",
        "Medium": "P4",
        "Low": "P4",
    },
}

# Known Services & Criticality Tiers
# Tier 1 = Critical, Tier 2 = High, Tier 3 = Medium, Tier 4 = Low
SERVICE_CATALOG = {
    "Production Database Cluster": {"criticality": "Critical", "category": "Database"},
    "Corporate VPN Gateway": {"criticality": "Critical", "category": "Network"},
    "Identity & SSO Authentication (Active Directory)": {"criticality": "Critical", "category": "Access Management"},
    "Customer-Facing E-Commerce API": {"criticality": "Critical", "category": "Application/Server"},
    "Corporate Email (Exchange/Outlook)": {"criticality": "High", "category": "Email Service"},
    "Enterprise ERP & Billing": {"criticality": "High", "category": "Application/Server"},
    "Core Network Switches / Wi-Fi": {"criticality": "High", "category": "Network"},
    "Internal Collaboration (Slack/Teams)": {"criticality": "Medium", "category": "General IT"},
    "Internal Documentation Wiki": {"criticality": "Low", "category": "General IT"},
    "Workstation Desktop Software": {"criticality": "Low", "category": "Software"},
    "Security Endpoint Protection (EDR)": {"criticality": "Critical", "category": "Security"},
    "Other / General Service": {"criticality": "Medium", "category": "General IT"},
}

# Overrides for Priority Engine
AFFECTED_USERS_P1_THRESHOLD = 100
AFFECTED_USERS_P2_THRESHOLD = 25

# RAG & Similarity Thresholds
RETRIEVAL_SIMILARITY_THRESHOLD = 0.15  # Minimum cosine score to consider article relevant
RETRIEVAL_TOP_K = 3
SIMILAR_INCIDENTS_TOP_K = 3
SIMILAR_INCIDENTS_THRESHOLD = 0.12

# Default Support Teams mapped to categories
CATEGORY_TEAM_MAP = {
    "Network": "Network Operations Center (NOC)",
    "Access Management": "Identity & Access Management (IAM)",
    "Database": "Database Administration (DBA)",
    "Application/Server": "Site Reliability Engineering (SRE) / Application Support",
    "Email Service": "Messaging & Collaboration Engineering",
    "Software": "Desktop Support / Client Platform",
    "Security": "Security Operations Center (SOC)",
    "General IT": "IT Service Desk Tier 1",
}
