"""
Enterprise IT Incident Management & Resolution Assistant.
Centralized IT Operations and Triage Decision-Support Platform.
Conforms to ITIL v4 framework, retrieval-augmented generation (RAG),
and SOC2-aligned human-in-the-loop review governance.
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
    page_title="Enterprise IT Incident Management Console",
    page_icon=None,
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
        "provider_strategy", "Groq LLaMA (Priority 1) -> Gemini Fallback"
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


# Custom styling for enterprise appearance
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8fafc;
        border-radius: 6px;
        padding: 16px;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0284c7;
        margin-bottom: 12px;
    }
    .badge-p1 { color: #b91c1c; font-weight: 600; }
    .badge-p2 { color: #c2410c; font-weight: 600; }
    .badge-p3 { color: #0369a1; font-weight: 600; }
    .badge-p4 { color: #15803d; font-weight: 600; }
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        padding: 24px 28px;
        border-radius: 8px;
        margin-bottom: 24px;
        border: 1px solid #334155;
    }
    .hero-title {
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        margin-bottom: 8px;
        color: #ffffff;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #cbd5e1;
        line-height: 1.5;
        margin-bottom: 0px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Navigation
with st.sidebar:
    st.markdown("### Enterprise Service Desk")
    st.caption("IT Operations & Incident Triage Platform")

    nav_choice = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Submit Incident",
            "Incident Triage Analysis",
            "Knowledge Base & SOPs",
            "Historical Incident Search",
            "Engineer Review Queue",
            "Incident Audit Log",
            "Service Operations Analytics",
            "Benchmark & Evaluation Suite",
        ],
        index=0,
    )

    st.divider()

    st.markdown("**Inference Engine Status**")

    groq_online = config.is_valid_key(config.GROQ_API_KEY)
    gemini_online = config.is_valid_key(config.GEMINI_API_KEY)

    if groq_online:
        st.success(f"Groq LLaMA: Active (Priority 1: {config.GROQ_MODEL})")
    else:
        st.caption("Groq LLaMA: Standby / Unconfigured")

    if gemini_online:
        st.success(f"Google Gemini: Active (Secondary: {config.GEMINI_MODEL})")
    else:
        st.caption("Google Gemini: Standby / Unconfigured")

    st.info("Deterministic ITIL Engine: Always Active (Offline Fallback)")

    with st.expander("Engine Configuration", expanded=False):
        provider_strategy = st.selectbox(
            "Inference Policy",
            [
                "Groq LLaMA (Priority 1) -> Gemini Fallback",
                "Gemini (Priority 1) -> Groq LLaMA Fallback",
                "Groq LLaMA Only",
                "Gemini Only",
                "Deterministic Local Rule Engine Only",
            ],
            index=0,
            help="Configures the primary LLM provider and fallback routing order.",
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
            "Groq API Key (Session Override)",
            type="password",
            value=config.GROQ_API_KEY if config.is_valid_key(config.GROQ_API_KEY) else "",
            placeholder="gsk_...",
        )

        user_gemini_key = st.text_input(
            "Gemini API Key (Session Override)",
            type="password",
            value=config.GEMINI_API_KEY if config.is_valid_key(config.GEMINI_API_KEY) else "",
            placeholder="AIzaSy...",
        )

        st.session_state.provider_strategy = provider_strategy
        st.session_state.selected_groq_model = selected_groq_model
        st.session_state.selected_gemini_model = selected_gemini_model
        st.session_state.user_groq_key = user_groq_key
        st.session_state.user_gemini_key = user_gemini_key

    st.caption("ITSM Core v2.4 • ITIL v4 Standard Matrix • Enterprise Governance")


# ==============================================================================
# SECTION 1: DASHBOARD
# ==============================================================================
if nav_choice == "Dashboard":
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">IT Operations & Incident Management Console</div>
            <div class="hero-subtitle">
                Enterprise service desk decision-support platform for IT operations and engineering teams.
                Automates initial incident intake, calculates deterministic ITIL v4 priority ratings,
                correlates historical ticket resolutions, and provides vector-retrieved standard operating procedures (SOPs)
                with mandatory human-in-the-loop review governance.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    summary = database.get_analytics_summary()
    incidents = database.list_incidents(limit=5)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Incidents Logged", summary["total_incidents"])
    with col2:
        pending_count = summary["review_status_distribution"].get("Pending Review", 0)
        st.metric("Pending Engineer Review", pending_count, delta="Action Required" if pending_count > 0 else "Normal")
    with col3:
        p1_count = summary["priority_distribution"].get("P1", 0)
        p2_count = summary["priority_distribution"].get("P2", 0)
        st.metric("Active P1 / P2 Incidents", p1_count + p2_count)
    with col4:
        st.metric("Verified Runbooks Indexed", len(retriever.articles))

    st.markdown("---")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Active Incident Feed")
        if incidents:
            for inc in incidents:
                p_label = inc.get("recommended_priority", "P3")
                status = inc.get("review_status", "Pending Review")
                with st.expander(f"**{inc['incident_id']}** • [{p_label}] {inc['predicted_category']} — {status}"):
                    st.write(f"**Description:** {inc['description']}")
                    st.write(f"**Affected Service:** {inc.get('affected_service')} ({inc.get('affected_users', 1)} users)")
                    st.write(f"**Triage Summary:** {inc.get('generated_summary')}")
                    st.caption(f"Created: {inc.get('created_at')} | Mode: {inc.get('analysis_mode')}")
        else:
            st.info("No incidents logged in database yet.")

    with c2:
        st.subheader("Operational Framework & Governance")
        st.markdown(
            r"""
            * **Intake Standard:** Structured schema with mandatory service attribution and scope estimates.
            * **ITIL v4 Severity Engine:** Deterministic $3 \times 3$ Impact x Urgency calculation with Tier 1 service escalation.
            * **Runbook Retrieval:** Threshold-gated vector similarity (cosine score threshold $\ge 0.15$).
            * **Historical Correlator:** TF-IDF symptom matching against closed ticket resolution histories.
            * **Human Governance:** Mandatory engineer verification prior to operational dispatch.
            * **Audit Repository:** Thread-safe SQLite ledger with immutable timestamping and review histories.
            """
        )
        if st.button("Submit New Incident Ticket", type="primary"):
            st.session_state["nav_goto"] = "Submit Incident"
            st.rerun()


# ==============================================================================
# SECTION 2: SUBMIT INCIDENT
# ==============================================================================
elif nav_choice == "Submit Incident":
    st.title("Incident Intake & Triage Submission")
    st.markdown("Submit production incidents for automated technical categorization, ITIL priority calculation, and runbook recommendation.")

    # Demonstration Templates
    template_options = {
        "Blank Form (Manual Entry)": {
            "desc": "",
            "service": "Other / General Service",
            "users": 1,
            "impact": "Medium",
            "urgency": "Medium",
            "error": "",
        },
        "Template: Corporate VPN Gateway Outage": {
            "desc": "Users across the regional office are completely unable to connect to the corporate VPN gateway. TLS connection timeout errors reported.",
            "service": "Corporate VPN Gateway",
            "users": 45,
            "impact": "High",
            "urgency": "High",
            "error": "TLS Handshake timeout: 504 Gateway Error",
        },
        "Template: Production Database Pool Exhaustion": {
            "desc": "Production customer-facing checkout API is throwing 500 errors. Database logs show max client connection pool capacity has been reached.",
            "service": "Production Database Cluster",
            "users": 250,
            "impact": "High",
            "urgency": "High",
            "error": "FATAL: remaining connection slots are reserved for non-replication superuser connections",
        },
        "Template: Exchange Mail Flow Latency": {
            "desc": "Multiple employees in marketing report external and internal corporate Outlook emails are delayed by over 2 hours.",
            "service": "Corporate Email (Exchange/Outlook)",
            "users": 65,
            "impact": "Medium",
            "urgency": "Medium",
            "error": "SMTP 451: 4.4.0 Message delivery deferred",
        },
        "Template: Domain Workstation Authentication Lockout": {
            "desc": "Single accountant cannot sign into Windows domain workstation because the self-service password reset link expired.",
            "service": "Identity & SSO Authentication (Active Directory)",
            "users": 1,
            "impact": "Low",
            "urgency": "Medium",
            "error": "Error 401: Invalid or expired password reset token",
        },
    }

    selected_template = st.selectbox("Load Standard Incident Template", list(template_options.keys()))
    t_data = template_options[selected_template]

    with st.form("incident_submission_form"):
        description = st.text_area(
            "Incident Description *",
            value=t_data["desc"],
            height=120,
            placeholder="Describe the issue, symptoms, affected infrastructure, and user impact...",
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
            "Error Message / Diagnostic Log Snippet (Optional)",
            value=t_data["error"],
            placeholder="e.g., HTTP 500, TLS handshake error, ORA-12541",
        )

        submitted = st.form_submit_button("Analyze & Triage Incident", type="primary")

    if submitted:
        if len(description.strip()) < 10:
            st.error("Please provide a meaningful incident description (at least 10 characters).")
        else:
            with st.spinner("Processing intake validation, evaluating ITIL priority matrix, and querying verified runbooks..."):
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

            st.success(f"Incident analyzed and recorded successfully. Reference Ticket ID: **{report.incident_id}**")
            st.info("Navigate to **Incident Triage Analysis** to view comprehensive technical recommendations.")


# ==============================================================================
# SECTION 3: INCIDENT TRIAGE ANALYSIS
# ==============================================================================
elif nav_choice == "Incident Triage Analysis":
    st.title("Incident Triage & Technical Decision Support")
    report: IncidentTriageReport = st.session_state.latest_report

    if not report:
        recent = database.list_incidents(limit=1)
        if recent:
            st.info(f"Displaying latest incident recorded: **{recent[0]['incident_id']}**")
            inc = recent[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Category", inc.get("predicted_category"))
            c2.metric("Recommended Priority", inc.get("recommended_priority"))
            c3.metric("Confidence Score", f"{inc.get('confidence', 75)}%")

            st.markdown("### Incident Summary")
            st.info(inc.get("generated_summary"))

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("### Business Impact")
                st.write(inc.get("business_impact"))
            with col_b:
                st.markdown("### Assigned Support Team")
                st.write(f"**{inc.get('recommended_team')}**")

            st.markdown("### Recommended Diagnostic & Resolution Steps")
            for i, act in enumerate(inc.get("suggested_actions", []), 1):
                st.markdown(f"**{i}.** {act}")

            st.markdown("### Supporting Knowledge Runbooks")
            st.write(f"Referenced Article IDs: `{', '.join(inc.get('retrieved_article_ids', [])) or 'None'}`")
            st.caption(f"Analysis Mode: {inc.get('analysis_mode')} | Review status: {inc.get('review_status')}")
        else:
            st.warning("No incident has been analyzed yet. Please submit an incident via 'Submit Incident'.")
    else:
        st.caption(f"Ticket ID: `{report.incident_id}` | Mode: `{report.analysis_mode}`")

        # Provider Status Notice
        if "Groq" in report.analysis_mode:
            st.success(f"Processed via Groq LLaMA (Priority 1 Engine) — `{report.analysis_mode}`")
        elif "Gemini" in report.analysis_mode:
            st.info(f"Processed via Google Gemini — `{report.analysis_mode}`")
        else:
            st.warning(f"Processed via Deterministic Offline Engine — `{report.analysis_mode}`")

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
                    <strong>Deterministic ITIL v4 Priority Engine:</strong> {p_details.get('priority_label')}<br>
                    <small>{p_details.get('explanation')}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### Incident Summary")
        st.write(report.analysis.incident_summary)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### Business Impact Analysis")
            st.write(report.analysis.business_impact)
        with col_b:
            st.markdown("### Assigned Engineering Support Team")
            st.success(f"**{report.analysis.recommended_support_team}**")

        st.markdown("### Recommended Diagnostic & Resolution Steps")
        for i, act in enumerate(report.analysis.suggested_troubleshooting_actions, 1):
            st.markdown(f"**{i}.** {act}")

        # Missing Information Callout
        if report.analysis.missing_information:
            with st.expander("Missing Technical Information / Diagnostic Gaps", expanded=True):
                for m in report.analysis.missing_information:
                    st.write(f"- {m}")

        # Knowledge Base Attribution
        st.markdown("### Referenced Standard Operating Procedures (Runbooks)")
        if report.has_relevant_knowledge and report.retrieved_articles:
            for art in report.retrieved_articles:
                with st.expander(f"[{art['article_id']}] {art['title']} (Cosine Relevance: {art['score']})"):
                    st.write(f"**Category:** {art['category']}")
                    st.write(f"**Runbook Overview:** {art['content']}")
                    st.markdown("**Actionable Steps in KB:**")
                    for s in art.get("troubleshooting_steps", []):
                        st.write(f"  - {s}")
                    st.caption(f"Source: {art.get('source_metadata', {}).get('author', 'IT Engineering')}")
        else:
            st.info("No knowledge base articles met the relevance threshold (score >= 0.15). General IT triage guidance applied.")

        # Guardrail & Limitations Note
        st.warning(f"Operational Safeguard: {report.analysis.limitations_or_uncertainty}")


# ==============================================================================
# SECTION 4: KNOWLEDGE BASE & SOPS
# ==============================================================================
elif nav_choice == "Knowledge Base & SOPs":
    st.title("IT Operations Knowledge Base & SOP Runbooks")
    st.markdown("Enterprise repository of verified standard operating procedures (SOPs) and technical resolution checklists.")

    col1, col2 = st.columns([3, 1])
    with col1:
        search_kw = st.text_input("Search Runbooks", placeholder="e.g., VPN, database connection, password reset, 502...")
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

    st.write(f"Showing **{len(articles)}** active standard operating runbooks:")

    for a in articles:
        with st.expander(f"[{a.get('article_id')}] {a.get('title')} — {a.get('category')}"):
            st.write(f"**Applicable Services:** {', '.join(a.get('applicable_service', []))}")
            st.write(f"**Overview:** {a.get('content')}")
            st.markdown("**Troubleshooting Runbook Checklist:**")
            for step in a.get("troubleshooting_steps", []):
                st.markdown(f"- [ ] {step}")
            meta = a.get("source_metadata", {})
            st.caption(f"Author: {meta.get('author')} | Version: {meta.get('runbook_version')} | Last Reviewed: {meta.get('last_updated')}")


# ==============================================================================
# SECTION 5: HISTORICAL INCIDENT SEARCH
# ==============================================================================
elif nav_choice == "Historical Incident Search":
    st.title("Historical Incident Symptom Correlator")
    st.markdown("Query the historical ticket archive to identify previous incidents with correlated technical symptoms.")

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
            st.success(f"Identified **{len(matches)}** correlated historical incidents:")
            for m in matches:
                with st.container():
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h4>Incident {m['incident_id']} — {m['category']} ({m['priority']}) | Similarity: {m['similarity_score']}</h4>
                            <p><strong>Affected Service:</strong> {m['affected_service']}</p>
                            <p><strong>Past Description:</strong> {m['description']}</p>
                            <p><strong>Prior Resolution Notes:</strong> <span style="color:#15803d;">{m['resolution_notes']}</span></p>
                            <small><em>{m['root_cause_caveat']}</em></small>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No historical incidents matched above the similarity threshold.")


# ==============================================================================
# SECTION 6: ENGINEER REVIEW QUEUE
# ==============================================================================
elif nav_choice == "Engineer Review Queue":
    st.title("Engineer Review & Governance Queue")
    st.markdown("Mandatory ITIL compliance gate: Operations engineers review, calibrate, or approve AI recommendations prior to operational dispatch.")

    pending_incidents = database.list_incidents(status_filter="Pending Review")

    if not pending_incidents:
        st.success("All submitted incidents have been reviewed. Queue status: Clear.")
        if st.checkbox("Show all historical incidents for secondary review"):
            pending_incidents = database.list_incidents(limit=20)

    if pending_incidents:
        inc_options = {
            f"{i['incident_id']} — {i['predicted_category']} ({i['recommended_priority']}): {i['description'][:50]}...": i["incident_id"]
            for i in pending_incidents
        }
        selected_label = st.selectbox("Select Incident Ticket from Queue", list(inc_options.keys()))
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

            st.markdown("### Support Engineer Review Determination")
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
                    reviewer_category = st.selectbox("Calibrated Category", config.CATEGORIES, index=cat_idx)
                with c3:
                    pri_keys = list(config.PRIORITY_LEVELS.keys())
                    p_val = inc_data["recommended_priority"]
                    p_idx = pri_keys.index(p_val) if p_val in pri_keys else 0
                    reviewer_priority = st.selectbox("Calibrated Priority", pri_keys, index=p_idx)

                reviewer_notes = st.text_area(
                    "Reviewer Audit Comments",
                    value=inc_data.get("reviewer_notes") or "",
                    placeholder="Document validation checks, root cause verification, or rationale for calibration...",
                )

                submit_review = st.form_submit_button("Commit Review Decision", type="primary")

            if submit_review:
                database.update_review(
                    incident_id=selected_id,
                    review_status=review_decision,
                    reviewer_category=reviewer_category,
                    reviewer_priority=reviewer_priority,
                    reviewer_notes=reviewer_notes,
                )
                st.success(f"Review decision for **{selected_id}** committed successfully as **{review_decision}**.")
                st.rerun()


# ==============================================================================
# SECTION 7: INCIDENT AUDIT LOG
# ==============================================================================
elif nav_choice == "Incident Audit Log":
    st.title("Incident Audit Log & Records")
    st.markdown("Comprehensive audit trail of submitted incident tickets, triage determinations, and engineer review outcomes.")

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
                    "Review Status": r.get("review_status"),
                    "Processing Engine": r.get("analysis_mode"),
                    "Timestamp": r.get("created_at")[:16],
                }
            )
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
    else:
        st.info("No records match the selected filter.")


# ==============================================================================
# SECTION 8: SERVICE OPERATIONS ANALYTICS
# ==============================================================================
elif nav_choice == "Service Operations Analytics":
    st.title("Service Operations & Incident Analytics")
    st.markdown("Aggregate metrics, category distributions, priority breakdowns, and inference engine telemetry.")

    summary = database.get_analytics_summary()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Recorded", summary["total_incidents"])
    col2.metric("Approved", summary["review_status_distribution"].get("Approved", 0))
    col3.metric("Modified by Engineer", summary["review_status_distribution"].get("Modified", 0))
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

    st.subheader("Telemetry & Review Governance Breakdown")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Engineer Review Status Breakdown:**")
        st.json(summary["review_status_distribution"])
    with c2:
        st.write("**Inference Engine Distribution:**")
        st.json(summary["mode_distribution"])


# ==============================================================================
# SECTION 9: BENCHMARK & EVALUATION SUITE
# ==============================================================================
elif nav_choice == "Benchmark & Evaluation Suite":
    st.title("System Benchmark & Model Evaluation Suite")
    st.markdown(
        """
        Automated empirical evaluation harness executing against labeled ground-truth incident test queries (`data/evaluation_queries.csv`).
        Measures **Retrieval Precision@K, Recall@K**, **Classification Accuracy & F1**, and **System Latency Telemetry**.
        """
    )

    if st.button("Run Benchmark Suite", type="primary"):
        with st.spinner("Executing retrieval evaluation and classification benchmarks across test queries..."):
            results = run_full_evaluation()
            st.session_state.evaluation_results = results
        st.success("Evaluation completed successfully.")

    if st.session_state.evaluation_results:
        res = st.session_state.evaluation_results
        ret = res["retrieval"]
        clf = res["classification_and_latency"]

        st.subheader("Benchmark Performance Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retrieval Recall@3", f"{ret['mean_recall_at_k'] * 100:.1f}%")
        c2.metric("Retrieval Precision@3", f"{ret['mean_precision_at_k'] * 100:.1f}%")
        c3.metric("Category Accuracy", f"{clf['category_accuracy'] * 100:.1f}%")
        c4.metric("Avg Latency", f"{clf['average_latency_ms']:.1f} ms")

        st.markdown("---")

        tab_ret, tab_clf, tab_details = st.tabs(["Retrieval Metrics", "Classification Performance", "Query Diagnostic Log"])

        with tab_ret:
            st.markdown(f"**Knowledge Retrieval Benchmark (Top K = {ret['top_k']}, Cosine Threshold = {ret['threshold']})**")
            st.write(f"- **Evaluated Queries:** {ret['total_queries']}")
            st.write(f"- **Mean Recall@{ret['top_k']}:** {ret['mean_recall_at_k'] * 100:.2f}% (Proportion of relevant runbooks retrieved in Top-3)")
            st.write(f"- **Mean Precision@{ret['top_k']}:** {ret['mean_precision_at_k'] * 100:.2f}% (Proportion of relevant runbooks among Top-3 retrieved)")
            st.caption("Benchmark Note: Each test query maps to 1 ground-truth SOP runbook; maximum theoretical Precision@3 is 33.33%.")

        with tab_clf:
            st.markdown("**Incident Classification & Deterministic Priority Engine**")
            st.write(f"- **Overall Category Accuracy:** {clf['category_accuracy'] * 100:.2f}%")
            st.write(f"- **Macro F1-Score:** {clf['macro_f1']:.4f}")
            st.write(f"- **Weighted F1-Score:** {clf['weighted_f1']:.4f}")
            st.write(f"- **Deterministic Priority Agreement Rate:** {clf['priority_agreement_rate'] * 100:.2f}%")
            st.write(f"- **Average End-to-End Latency:** {clf['average_latency_ms']} ms")
            st.write(f"- **95th Percentile Latency (P95):** {clf['p95_latency_ms']} ms")

        with tab_details:
            st.markdown("**Detailed Query Diagnostic Log**")
            detail_df = pd.DataFrame(clf["details"])
            st.dataframe(detail_df, use_container_width=True)
    else:
        st.info("Click **'Run Benchmark Suite'** above to compute empirical metrics on the test dataset.")
