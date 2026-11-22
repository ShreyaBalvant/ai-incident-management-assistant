"""
AI-Powered IT Incident Management and Resolution Assistant.
Enterprise decision-support prototype combining NLP, RAG, deterministic ITIL priority matrix,
and human-in-the-loop review controls.
"""

import json
import os
from datetime import datetime
import pandas as pd
import streamlit as st

import config
import database
from incident_analyzer import IncidentAnalyzer, IncidentIntakePayload, IncidentTriageReport
from retriever import KnowledgeRetriever
from similar_incidents import SimilarIncidentMatcher
from evaluation import run_full_evaluation

# Page configuration
st.set_page_config(
    page_title="AI Incident Management Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize database and seed demo data if empty
database.init_db()
database.seed_demo_incidents_if_empty()


@st.cache_resource
def get_analyzer():
    return IncidentAnalyzer()


@st.cache_resource
def get_retriever():
    return KnowledgeRetriever()


@st.cache_resource
def get_similar_matcher():
    return SimilarIncidentMatcher()


analyzer = get_analyzer()
retriever = get_retriever()
similar_matcher = get_similar_matcher()

# Session state initialization
if "latest_report" not in st.session_state:
    st.session_state.latest_report = None
if "selected_incident_id" not in st.session_state:
    st.session_state.selected_incident_id = None
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None


def get_effective_provider_params():
    """Extracts user-configured LLM provider priority and credentials from UI session state."""
    strategy = st.session_state.get(
        "provider_strategy", "Groq LLaMA (Priority 1) → Gemini Fallback"
    )
    if "Groq LLaMA (Priority 1)" in strategy:
        order = ["groq", "gemini"]
    elif "Gemini (Priority 1)" in strategy:
        order = ["gemini", "groq"]
    elif "Groq LLaMA Only" in strategy:
        order = ["groq"]
    elif "Gemini Only" in strategy:
        order = ["gemini"]
    else:
        order = []  # Local deterministic fallback only

    groq_k = st.session_state.get("user_groq_key")
    gemini_k = st.session_state.get("user_gemini_key")

    return {
        "provider_order": order,
        "groq_api_key": groq_k if config.is_valid_key(groq_k) else None,
        "gemini_api_key": gemini_k if config.is_valid_key(gemini_k) else None,
        "groq_model": st.session_state.get("selected_groq_model"),
        "gemini_model": st.session_state.get("selected_gemini_model"),
    }

# Custom styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 10px;
    }
    .badge-p1 { color: #d32f2f; font-weight: bold; }
    .badge-p2 { color: #f57c00; font-weight: bold; }
    .badge-p3 { color: #1976d2; font-weight: bold; }
    .badge-p4 { color: #388e3c; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("ITSM AI Assistant")
    st.caption("AI Decision-Support System")

    nav_choice = st.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "📝 Submit Incident",
            "🤖 AI Incident Analysis",
            "📚 Knowledge Base",
            "🔍 Similar Incidents",
            "👤 Human Review",
            "📜 Incident History",
            "📈 Analytics",
            "🎯 Evaluation & Benchmarks",
        ],
        index=0,
    )

    st.divider()

    # Multi-Tier GenAI Engine & Fallback Controls
    st.subheader("🤖 GenAI & Fallback System")

    groq_online = config.is_valid_key(config.GROQ_API_KEY)
    gemini_online = config.is_valid_key(config.GEMINI_API_KEY)

    if groq_online:
        st.success(f"● Groq LLaMA: Online (Priority 1: {config.GROQ_MODEL})", icon="⚡")
    else:
        st.caption("○ Groq LLaMA: No API Key (using env / fallback)")

    if gemini_online:
        st.success(f"● Gemini API: Online (Secondary: {config.GEMINI_MODEL})", icon="🛡️")
    else:
        st.caption("○ Gemini API: No API Key (using env / fallback)")

    st.success("● Local Fallback: 100% Ready (Fail-Safe)", icon="🔒")

    with st.expander("⚙️ Fallback & Model Config", expanded=False):
        provider_strategy = st.selectbox(
            "Model Priority Strategy",
            [
                "Groq LLaMA (Priority 1) → Gemini Fallback",
                "Gemini (Priority 1) → Groq LLaMA Fallback",
                "Groq LLaMA Only (Open Source)",
                "Gemini Only",
                "Deterministic Local Fallback Only",
            ],
            index=0,
            help="Groq LLaMA is open-source and free-tier friendly. If payment is required on Gemini, Groq executes seamlessly on Priority 1.",
        )

        selected_groq_model = st.selectbox(
            "Groq Model",
            config.GROQ_MODELS,
            index=config.GROQ_MODELS.index(config.GROQ_MODEL) if config.GROQ_MODEL in config.GROQ_MODELS else 0,
        )

        selected_gemini_model = st.selectbox(
            "Gemini Model",
            config.GEMINI_MODELS,
            index=config.GEMINI_MODELS.index(config.GEMINI_MODEL) if config.GEMINI_MODEL in config.GEMINI_MODELS else 0,
        )

        user_groq_key = st.text_input(
            "Groq API Key (Override)",
            type="password",
            value=config.GROQ_API_KEY if config.is_valid_key(config.GROQ_API_KEY) else "",
            placeholder="gsk_...",
            help="Free key from console.groq.com/keys",
        )

        user_gemini_key = st.text_input(
            "Gemini API Key (Override)",
            type="password",
            value=config.GEMINI_API_KEY if config.is_valid_key(config.GEMINI_API_KEY) else "",
            placeholder="AIzaSy...",
            help="Key from aistudio.google.com",
        )

        st.session_state.provider_strategy = provider_strategy
        st.session_state.selected_groq_model = selected_groq_model
        st.session_state.selected_gemini_model = selected_gemini_model
        st.session_state.user_groq_key = user_groq_key
        st.session_state.user_gemini_key = user_gemini_key

    st.caption("v2.1 • Resilience: Groq LLaMA (P1) → Gemini (P2) → Local ITIL Fallback")


# ==============================================================================
# SECTION 1: DASHBOARD
# ==============================================================================
if nav_choice == "📊 Dashboard":
    st.title("🛡️ IT Incident Operations Dashboard")
    st.markdown("Real-time overview of incoming incidents, automated triaging, and human review queues.")

    summary = database.get_analytics_summary()
    incidents = database.list_incidents(limit=5)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Incidents", summary["total_incidents"])
    with col2:
        pending_count = summary["review_status_distribution"].get("Pending Review", 0)
        st.metric("Pending Human Review", pending_count, delta="Action Required" if pending_count > 0 else "Clear")
    with col3:
        p1_count = summary["priority_distribution"].get("P1", 0)
        p2_count = summary["priority_distribution"].get("P2", 0)
        st.metric("High/Critical Incidents", p1_count + p2_count)
    with col4:
        st.metric("KB Runbooks Indexed", len(retriever.articles))

    st.markdown("---")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("⚡ Quick Incident Triage Feed")
        if incidents:
            for inc in incidents:
                p_label = inc.get("recommended_priority", "P3")
                status = inc.get("review_status", "Pending Review")
                with st.expander(f"**{inc['incident_id']}** • [{p_label}] {inc['predicted_category']} — {status}"):
                    st.write(f"**Description:** {inc['description']}")
                    st.write(f"**Affected Service:** {inc.get('affected_service')} ({inc.get('affected_users', 1)} users)")
                    st.write(f"**AI Summary:** {inc.get('generated_summary')}")
                    st.caption(f"Created: {inc.get('created_at')} | Mode: {inc.get('analysis_mode')}")
        else:
            st.info("No incidents logged in database yet.")

    with c2:
        st.subheader("📌 System Architecture Status")
        st.markdown(
            r"""
            * **Input Intake:** Multi-field validation & schema enforcement.
            * **Priority Engine:** Deterministic ITIL $3 \times 3$ Matrix + Service overrides.
            * **Knowledge Base:** TF-IDF with cosine relevance gating ($\ge 0.15$).
            * **Similar Incidents:** Historical vector matching with prior resolutions.
            * **Human in the Loop:** Mandatory review before operational execution.
            * **Local Storage:** SQLite (`incidents.db`) with full audit trail.
            """
        )
        if st.button("Submit New Incident Now", type="primary"):
            st.session_state["nav_goto"] = "📝 Submit Incident"
            st.rerun()


# ==============================================================================
# SECTION 2: SUBMIT INCIDENT
# ==============================================================================
elif nav_choice == "📝 Submit Incident":
    st.title("📝 IT Incident Intake Form")
    st.markdown("Submit a technical incident for automated NLP classification, deterministic ITIL prioritization, and RAG resolution guidance.")

    # Demonstration Templates
    template_options = {
        "Custom (Blank)": {
            "desc": "",
            "service": "Other / General Service",
            "users": 1,
            "impact": "Medium",
            "urgency": "Medium",
            "error": "",
        },
        "Example 1: Corporate VPN Gateway Outage": {
            "desc": "Users across the regional office are completely unable to connect to the corporate VPN gateway. TLS connection timeout errors reported.",
            "service": "Corporate VPN Gateway",
            "users": 45,
            "impact": "High",
            "urgency": "High",
            "error": "TLS Handshake timeout: 504 Gateway Error",
        },
        "Example 2: Production Database Connection Exhaustion": {
            "desc": "Production customer-facing checkout API is throwing 500 errors. Database logs show max client connection pool capacity has been reached.",
            "service": "Production Database Cluster",
            "users": 250,
            "impact": "High",
            "urgency": "High",
            "error": "FATAL: remaining connection slots are reserved for non-replication superuser connections",
        },
        "Example 3: Outlook Mail Delivery Delay": {
            "desc": "Multiple employees in marketing report external and internal corporate Outlook emails are delayed by over 2 hours.",
            "service": "Corporate Email (Exchange/Outlook)",
            "users": 65,
            "impact": "Medium",
            "urgency": "Medium",
            "error": "SMTP 451: 4.4.0 Message delivery deferred",
        },
        "Example 4: Workstation Password Reset Failure": {
            "desc": "Single accountant cannot sign into Windows domain workstation because the self-service password reset link expired.",
            "service": "Identity & SSO Authentication (Active Directory)",
            "users": 1,
            "impact": "Low",
            "urgency": "Medium",
            "error": "Error 401: Invalid or expired password reset token",
        },
    }

    selected_template = st.selectbox("Load Example Incident Template", list(template_options.keys()))
    t_data = template_options[selected_template]

    with st.form("incident_submission_form"):
        description = st.text_area(
            "Incident Description *",
            value=t_data["desc"],
            height=120,
            placeholder="Describe the issue, symptoms, and circumstances...",
            help="Minimum 10 characters required.",
        )

        col1, col2 = st.columns(2)
        with col1:
            service_list = list(config.SERVICE_CATALOG.keys())
            def_idx = service_list.index(t_data["service"]) if t_data["service"] in service_list else 0
            affected_service = st.selectbox("Affected Service *", service_list, index=def_idx)
            affected_users = st.number_input(
                "Estimated Affected Users *",
                min_value=1,
                max_value=50000,
                value=t_data["users"],
                step=1,
            )

        with col2:
            impact_idx = config.IMPACT_LEVELS.index(t_data["impact"])
            business_impact = st.selectbox("Business Impact *", config.IMPACT_LEVELS, index=impact_idx)
            urgency_idx = config.URGENCY_LEVELS.index(t_data["urgency"])
            urgency = st.selectbox("Urgency *", config.URGENCY_LEVELS, index=urgency_idx)

        error_message = st.text_input(
            "Error Message / Log Snippet (Optional)",
            value=t_data["error"],
            placeholder="e.g., HTTP 500, TLS handshake error, ORA-12541",
        )

        submitted = st.form_submit_button("🚀 Analyze & Triage Incident", type="primary")

    if submitted:
        if len(description.strip()) < 10:
            st.error("Please provide a meaningful incident description (at least 10 characters).")
        else:
            with st.spinner("Processing incident intake, evaluating ITIL priority, and querying knowledge base..."):
                payload = IncidentIntakePayload(
                    description=description,
                    affected_service=affected_service,
                    affected_users=int(affected_users),
                    business_impact=business_impact,
                    urgency=urgency,
                    error_message=error_message,
                )
                provider_params = get_effective_provider_params()
                report: IncidentTriageReport = analyzer.process_incident(
                    payload,
                    persist=True,
                    **provider_params,
                )
                st.session_state.latest_report = report
                st.session_state.selected_incident_id = report.incident_id

            st.success(f"Incident analyzed and recorded successfully! Ticket ID: **{report.incident_id}**")
            st.info("Navigate to **🤖 AI Incident Analysis** to view the full triage recommendations.")


# ==============================================================================
# SECTION 3: AI INCIDENT ANALYSIS
# ==============================================================================
elif nav_choice == "🤖 AI Incident Analysis":
    st.title("🤖 AI Incident Triage & Decision Support")
    report: IncidentTriageReport = st.session_state.latest_report

    if not report:
        # Load the latest incident from database if available
        recent = database.list_incidents(limit=1)
        if recent:
            st.info(f"Displaying most recent incident recorded: **{recent[0]['incident_id']}**")
            inc = recent[0]
            # Construct a viewable representation
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Category", inc.get("predicted_category"))
            c2.metric("Recommended Priority", inc.get("recommended_priority"))
            c3.metric("Confidence Score", f"{inc.get('confidence', 75)}%")

            st.markdown("### 📋 Incident Summary")
            st.info(inc.get("generated_summary"))

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("### 🏢 Business Impact")
                st.write(inc.get("business_impact"))
            with col_b:
                st.markdown("### 🛠️ Recommended Support Team")
                st.write(f"**{inc.get('recommended_team')}**")

            st.markdown("### 🔧 Suggested Troubleshooting Actions")
            for i, act in enumerate(inc.get("suggested_actions", []), 1):
                st.markdown(f"**{i}.** {act}")

            st.markdown("### 📚 Supporting Knowledge Runbooks")
            st.write(f"Referenced Article IDs: `{', '.join(inc.get('retrieved_article_ids', [])) or 'None'}`")
            st.caption(f"Analysis Mode: {inc.get('analysis_mode')} | Human review status: {inc.get('review_status')}")
        else:
            st.warning("No incident has been analyzed yet. Please submit an incident via 'Submit Incident'.")
    else:
        # Display rich interactive report
        st.caption(f"Ticket ID: `{report.incident_id}` | Mode: `{report.analysis_mode}`")

        # Provider / Fallback Notice
        if "Groq" in report.analysis_mode:
            st.success(f"⚡ **Served by Groq LLaMA (Priority 1 Open Source Engine)** — `{report.analysis_mode}`")
        elif "Gemini" in report.analysis_mode:
            st.info(f"🛡️ **Served by Google Gemini** — `{report.analysis_mode}`")
        else:
            st.warning(f"🔒 **Served by Fail-Safe Deterministic Fallback Engine** — `{report.analysis_mode}`")

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Category", report.analysis.category)
        c2.metric("Recommended Priority", report.analysis.priority_recommendation)
        c3.metric("Confidence Score", f"{report.analysis.confidence_score}%")

        # ITIL Priority Explanation Banner
        p_details = report.priority_details
        with st.container():
            st.markdown(
                f"""
                <div class="metric-card">
                    <strong>🎯 Deterministic ITIL Priority Engine:</strong> {p_details.get('priority_label')}<br>
                    <small>{p_details.get('explanation')}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### 📋 Incident Summary")
        st.write(report.analysis.incident_summary)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### 🏢 Business Impact")
            st.write(report.analysis.business_impact)
        with col_b:
            st.markdown("### 🛠️ Recommended Support Team")
            st.success(f"**{report.analysis.recommended_support_team}**")

        st.markdown("### 🔧 Suggested Troubleshooting Actions")
        for i, act in enumerate(report.analysis.suggested_troubleshooting_actions, 1):
            st.markdown(f"**{i}.** {act}")

        # Missing Information Callout
        if report.analysis.missing_information:
            with st.expander("⚠️ Missing Information / Clarification Needed", expanded=True):
                for m in report.analysis.missing_information:
                    st.write(f"• {m}")

        # Knowledge Base Attribution
        st.markdown("### 📚 Retrieved Knowledge Articles")
        if report.has_relevant_knowledge and report.retrieved_articles:
            for art in report.retrieved_articles:
                with st.expander(f"📖 [{art['article_id']}] {art['title']} (Cosine Relevance: {art['score']})"):
                    st.write(f"**Category:** {art['category']}")
                    st.write(f"**Runbook Content:** {art['content']}")
                    st.markdown("**Actionable Steps in KB:**")
                    for s in art.get("troubleshooting_steps", []):
                        st.write(f"  - {s}")
                    st.caption(f"Source: {art.get('source_metadata', {}).get('author', 'IT Engineering')}")
        else:
            st.info("ℹ️ No knowledge base articles met the relevance threshold (score >= 0.15). General IT triage guidance applied.")

        # Guardrail & Limitations Note
        st.warning(f"⚖️ **Operational Safeguard:** {report.analysis.limitations_or_uncertainty}")


# ==============================================================================
# SECTION 4: KNOWLEDGE BASE
# ==============================================================================
elif nav_choice == "📚 Knowledge Base":
    st.title("📚 Troubleshooting Knowledge Base & Runbooks")
    st.markdown("Search and inspect verified enterprise IT runbooks used for retrieval-augmented generation.")

    col1, col2 = st.columns([3, 1])
    with col1:
        search_kw = st.text_input("🔍 Search Knowledge Base", placeholder="e.g., VPN, database connection, password reset, 502...")
    with col2:
        cat_filter = st.selectbox("Category Filter", ["All Categories"] + config.CATEGORIES)

    articles = retriever.articles
    if cat_filter != "All Categories":
        articles = [a for a in articles if a.get("category") == cat_filter]

    if search_kw.strip():
        search_kw_lower = search_kw.lower()
        articles = [
            a for a in articles
            if search_kw_lower in a.get("title", "").lower()
            or search_kw_lower in a.get("content", "").lower()
            or search_kw_lower in a.get("keywords", "").lower()
        ]

    st.write(f"Showing **{len(articles)}** knowledge runbooks:")

    for a in articles:
        with st.expander(f"**[{a.get('article_id')}]** {a.get('title')} — *{a.get('category')}*"):
            st.write(f"**Applicable Services:** {', '.join(a.get('applicable_service', []))}")
            st.write(f"**Overview:** {a.get('content')}")
            st.markdown("**Troubleshooting Runbook Checklist:**")
            for step in a.get("troubleshooting_steps", []):
                st.markdown(f"- [ ] {step}")
            meta = a.get("source_metadata", {})
            st.caption(f"Author: {meta.get('author')} | Version: {meta.get('runbook_version')} | Last Reviewed: {meta.get('last_updated')}")


# ==============================================================================
# SECTION 5: SIMILAR INCIDENTS
# ==============================================================================
elif nav_choice == "🔍 Similar Incidents":
    st.title("🔍 Historical Similar Incident Detection")
    st.markdown("Discover past resolved incidents with similar symptoms to assist root-cause analysis and resolution.")

    query_text = st.text_input(
        "Search Incident Symptoms",
        value="Users cannot connect to corporate VPN AnyConnect gateway",
        placeholder="Enter symptoms or incident description...",
    )

    col1, col2 = st.columns(2)
    with col1:
        top_k_sim = st.slider("Top K Results", min_value=1, max_value=5, value=3)
    with col2:
        sim_threshold = st.slider("Similarity Threshold", min_value=0.05, max_value=0.50, value=0.12, step=0.01)

    if query_text.strip():
        matches = similar_matcher.find_similar(query_text, top_k=top_k_sim, threshold=sim_threshold)
        if matches:
            st.success(f"Found **{len(matches)}** potentially similar historical incidents:")
            for m in matches:
                with st.container():
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h4>🎟️ {m['incident_id']} — {m['category']} ({m['priority']}) | Similarity: {m['similarity_score']}</h4>
                            <p><strong>Affected Service:</strong> {m['affected_service']}</p>
                            <p><strong>Past Description:</strong> {m['description']}</p>
                            <p><strong>Prior Resolution Notes:</strong> <span style="color:#2e7d32;">{m['resolution_notes']}</span></p>
                            <small><em>{m['root_cause_caveat']}</em></small>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No historical incidents matched above the similarity threshold.")


# ==============================================================================
# SECTION 6: HUMAN REVIEW
# ==============================================================================
elif nav_choice == "👤 Human Review":
    st.title("👤 Human-in-the-Loop Review & Governance")
    st.markdown("Enterprise safety control: IT Support Engineers review, approve, modify, or reject AI recommendations before operational execution.")

    pending_incidents = database.list_incidents(status_filter="Pending Review")

    if not pending_incidents:
        st.success("🎉 All submitted incidents have been reviewed! No items in the review queue.")
        # Provide option to view reviewed incidents
        if st.checkbox("Show all incidents to re-review"):
            pending_incidents = database.list_incidents(limit=20)

    if pending_incidents:
        inc_options = {
            f"{i['incident_id']} — {i['predicted_category']} ({i['recommended_priority']}): {i['description'][:50]}...": i["incident_id"]
            for i in pending_incidents
        }
        selected_label = st.selectbox("Select Incident from Queue", list(inc_options.keys()))
        selected_id = inc_options[selected_label]
        inc_data = database.get_incident(selected_id)

        if inc_data:
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Submitted Incident Data")
                st.write(f"**Ticket ID:** `{inc_data['incident_id']}`")
                st.write(f"**Description:** {inc_data['description']}")
                st.write(f"**Service:** {inc_data.get('affected_service')} ({inc_data.get('affected_users')} users)")
                st.write(f"**Submitted Impact:** {inc_data.get('business_impact')} | **Urgency:** {inc_data.get('urgency')}")
                if inc_data.get("error_message"):
                    st.code(inc_data.get("error_message"))

            with col2:
                st.subheader("AI Recommendations")
                st.write(f"**Predicted Category:** `{inc_data.get('predicted_category')}`")
                st.write(f"**Recommended Priority:** `{inc_data.get('recommended_priority')}`")
                st.write(f"**Summary:** {inc_data.get('generated_summary')}")
                st.write(f"**Team:** {inc_data.get('recommended_team')}")

            st.markdown("### ✍️ Support Engineer Review Decision")
            with st.form("human_review_form"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    review_decision = st.selectbox(
                        "Decision *",
                        ["Approved", "Modified", "Rejected"],
                        index=0 if inc_data.get("review_status") == "Approved" else 1,
                    )
                with c2:
                    cat_idx = config.CATEGORIES.index(inc_data["predicted_category"]) if inc_data["predicted_category"] in config.CATEGORIES else 0
                    reviewer_category = st.selectbox("Final Category", config.CATEGORIES, index=cat_idx)
                with c3:
                    pri_keys = list(config.PRIORITY_LEVELS.keys())
                    p_val = inc_data["recommended_priority"]
                    p_idx = pri_keys.index(p_val) if p_val in pri_keys else 0
                    reviewer_priority = st.selectbox("Final Priority", pri_keys, index=p_idx)

                reviewer_notes = st.text_area(
                    "Reviewer Audit Comments",
                    value=inc_data.get("reviewer_notes") or "",
                    placeholder="Document validation checks, root cause verification, or reason for modifications...",
                )

                submit_review = st.form_submit_button("💾 Save Review Decision", type="primary")

            if submit_review:
                database.update_review(
                    incident_id=selected_id,
                    review_status=review_decision,
                    reviewer_category=reviewer_category,
                    reviewer_priority=reviewer_priority,
                    reviewer_notes=reviewer_notes,
                )
                st.success(f"Review decision for **{selected_id}** saved successfully as **{review_decision}**!")
                st.rerun()


# ==============================================================================
# SECTION 7: INCIDENT HISTORY
# ==============================================================================
elif nav_choice == "📜 Incident History":
    st.title("📜 Incident Audit History & Log")
    st.markdown("Inspect all recorded incidents, AI predictions, and human review outcomes.")

    col1, col2 = st.columns([2, 1])
    with col1:
        status_filter = st.selectbox("Filter by Review Status", ["All Statuses", "Pending Review", "Approved", "Modified", "Rejected"])
    with col2:
        max_rows = st.number_input("Max Records", min_value=5, max_value=500, value=50, step=10)

    f_arg = None if status_filter == "All Statuses" else status_filter
    records = database.list_incidents(limit=max_rows, status_filter=f_arg)

    if records:
        table_rows = []
        for r in records:
            table_rows.append(
                {
                    "Ticket ID": r["incident_id"],
                    "Category": r.get("reviewer_category") or r.get("predicted_category"),
                    "Priority": r.get("reviewer_priority") or r.get("recommended_priority"),
                    "Service": r.get("affected_service"),
                    "Users": r.get("affected_users"),
                    "Status": r.get("review_status"),
                    "Mode": r.get("analysis_mode"),
                    "Created": r.get("created_at")[:16],
                }
            )
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
    else:
        st.info("No records match the selected filter.")


# ==============================================================================
# SECTION 8: ANALYTICS
# ==============================================================================
elif nav_choice == "📈 Analytics":
    st.title("📈 Incident Analytics & Metrics")
    st.markdown("Operational distribution metrics and trends across categorized incidents.")

    summary = database.get_analytics_summary()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Recorded", summary["total_incidents"])
    col2.metric("Approved", summary["review_status_distribution"].get("Approved", 0))
    col3.metric("Modified by Human", summary["review_status_distribution"].get("Modified", 0))
    col4.metric("Pending Review", summary["review_status_distribution"].get("Pending Review", 0))

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Category Distribution")
        cat_df = pd.DataFrame(
            list(summary["category_distribution"].items()),
            columns=["Category", "Count"],
        )
        if not cat_df.empty:
            st.bar_chart(cat_df.set_index("Category"))
        else:
            st.info("No category data available.")

    with col_b:
        st.subheader("Priority Distribution")
        pri_df = pd.DataFrame(
            list(summary["priority_distribution"].items()),
            columns=["Priority", "Count"],
        )
        if not pri_df.empty:
            st.bar_chart(pri_df.set_index("Priority"))
        else:
            st.info("No priority data available.")

    st.subheader("Triage Mode & Review Status Breakdown")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Human Review Statuses:**")
        st.json(summary["review_status_distribution"])
    with c2:
        st.write("**Analysis Execution Engine:**")
        st.json(summary["mode_distribution"])


# ==============================================================================
# SECTION 9: EVALUATION & BENCHMARKS
# ==============================================================================
elif nav_choice == "🎯 Evaluation & Benchmarks":
    st.title("🎯 Evaluation & Model Benchmarks (Module 10)")
    st.markdown(
        """
        Reproducible, empirical evaluation on the labeled ground-truth dataset (`data/evaluation_queries.csv`).
        Computes **Retrieval Precision@K, Recall@K**, **Classification Accuracy & F1**, and **System Latency**.
        """
    )

    if st.button("▶️ Run Evaluation Benchmark Suite", type="primary"):
        with st.spinner("Running retrieval evaluation and classification benchmarks across test queries..."):
            results = run_full_evaluation()
            st.session_state.evaluation_results = results
        st.success("Evaluation completed successfully!")

    if st.session_state.evaluation_results:
        res = st.session_state.evaluation_results
        ret = res["retrieval"]
        clf = res["classification_and_latency"]

        st.subheader("📊 High-Level Benchmark Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retrieval Recall@3", f"{ret['mean_recall_at_k'] * 100:.1f}%")
        c2.metric("Retrieval Precision@3", f"{ret['mean_precision_at_k'] * 100:.1f}%")
        c3.metric("Category Accuracy", f"{clf['category_accuracy'] * 100:.1f}%")
        c4.metric("Avg Latency", f"{clf['average_latency_ms']:.1f} ms")

        st.markdown("---")

        tab_ret, tab_clf, tab_details = st.tabs(["📚 Retrieval Metrics", "🏷️ Classification & F1", "📋 Query Details"])

        with tab_ret:
            st.markdown(f"**Knowledge Retrieval Benchmark (Top K = {ret['top_k']}, Cosine Threshold = {ret['threshold']})**")
            st.write(f"- **Evaluated Queries:** {ret['total_queries']}")
            st.write(f"- **Mean Recall@{ret['top_k']}:** {ret['mean_recall_at_k'] * 100:.2f}% (Percentage of true relevant runbooks retrieved in Top-3)")
            st.write(f"- **Mean Precision@{ret['top_k']}:** {ret['mean_precision_at_k'] * 100:.2f}% (Average proportion of relevant runbooks among Top-3 retrieved)")
            st.caption("Note: Since each test query maps to exactly 1 ground-truth runbook, the maximum possible Precision@3 is 33.33%.")

        with tab_clf:
            st.markdown("**Incident Classification & Deterministic Priority Engine**")
            st.write(f"- **Overall Category Accuracy:** {clf['category_accuracy'] * 100:.2f}%")
            st.write(f"- **Macro F1-Score:** {clf['macro_f1']:.4f}")
            st.write(f"- **Weighted F1-Score:** {clf['weighted_f1']:.4f}")
            st.write(f"- **Deterministic Priority Agreement Rate:** {clf['priority_agreement_rate'] * 100:.2f}%")
            st.write(f"- **Average End-to-End Latency:** {clf['average_latency_ms']} ms")
            st.write(f"- **95th Percentile Latency (P95):** {clf['p95_latency_ms']} ms")

        with tab_details:
            st.markdown("**Detailed Query Breakdown**")
            detail_df = pd.DataFrame(clf["details"])
            st.dataframe(detail_df, use_container_width=True)
    else:
        st.info("Click **'Run Evaluation Benchmark Suite'** above to compute empirical metrics on the test dataset.")
