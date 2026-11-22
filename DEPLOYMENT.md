# 🚀 Production Deployment & Multi-Tier Fallback Guide

This guide details how to deploy the **AI-Powered IT Incident Management Assistant** to production environments, configure API credentials, and leverage the **multi-tier resilient LLM fallback architecture**.

---

## 🏗️ Multi-Tier Resilient AI Architecture

The assistant implements an enterprise 3-tier fallback architecture to guarantee **zero downtime**, avoid paid API lock-in, and maintain ITIL compliance even during provider outages or rate limits:

```
                  ┌──────────────────────────────────────────────┐
                  │          Incoming Incident Ticket            │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │   Deterministic ITIL Priority Calculation    │
                  │       (Impact x Urgency Matrix Engine)       │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │       RAG Knowledge Runbook Retrieval        │
                  │   (TF-IDF + Cosine Similarity > Threshold)   │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
            ┌───────────────────────────────────────────────────────────┐
            │                  LLM Provider Pipeline                    │
            │                                                           │
            │   [Priority 1] ⚡ Groq LLaMA 3.3 70B (Open Source, Fast)  │
            │       │ (If Rate-Limited, Quota Exceeded, or Failed)      │
            │       ▼                                                   │
            │   [Priority 2] 🛡️ Google Gemini (gemini-3.8-flash)        │
            │       │ (If Payment Required, Quota Exceeded, or Failed)  │
            │       ▼                                                   │
            │   [Priority 3] 🔒 Deterministic Rule-Based Fallback       │
            │                (100% Uptime Guaranteed Offline Engine)    │
            └────────────────────────────┬──────────────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │      Interactive Human-in-the-Loop Review    │
                  │     Audit Logging to SQLite (incidents.db)   │
                  └──────────────────────────────────────────────┘
```

### Why Groq LLaMA is on Priority 1:
- **Free Tier & Zero Cost**: Groq offers high rate limits on open-source LLaMA 3.3 70B and LLaMA 3.1 8B without requiring a credit card or upfront payment.
- **Ultra-Low Latency**: Generates triage JSON structured responses in under 500ms.
- **Dynamic Fallback**: If Groq encounters rate limits or errors, requests automatically route to Google Gemini. If Gemini requires payments or is unavailable, the deterministic offline engine handles the ticket. The application **never crashes**.

---

## 🔑 Environment Variables & API Keys

Copy `.env.example` to `.env` or set these in your cloud provider's **Secrets / Environment Variables** panel:

| Variable | Description | Default | Required? |
|---|---|---|---|
| `GROQ_API_KEY` | Groq API Key ([Get free key](https://console.groq.com/keys)) | `""` | Recommended (Priority 1) |
| `GROQ_MODEL` | Groq Model name | `llama-3.3-70b-versatile` | No |
| `GEMINI_API_KEY` | Google Gemini API Key ([Get key](https://aistudio.google.com/)) | `""` | Optional (Priority 2) |
| `GEMINI_MODEL` | Gemini Model name | `gemini-3.8-flash` | No |
| `LLM_PROVIDER_ORDER` | Comma-separated provider priority | `groq,gemini` | No |
| `DB_PATH` | Path to SQLite database file | `incidents.db` | No (for volume mounts) |
| `PORT` | Web server listening port | `8501` | Injected by cloud PaaS |

> **Note**: If neither `GROQ_API_KEY` nor `GEMINI_API_KEY` is provided, the application automatically boots into **Local Fallback Mode**, providing full end-to-end ITIL triaging and runbook recommendations with zero cloud dependencies.

---

## 🌐 Deployment Option 1: Streamlit Community Cloud (Recommended & Free)

The fastest and easiest zero-cost deployment:

1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Configure production deployment and Groq LLaMA fallback"
   git push origin main
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **"New app"**:
   - **Repository**: Select your GitHub repository.
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **"Advanced settings..."** -> **Secrets**:
   Add your secrets in TOML format:
   ```toml
   GROQ_API_KEY = "gsk_your_groq_api_key_here"
   GROQ_MODEL = "llama-3.3-70b-versatile"
   GEMINI_API_KEY = "AIzaSy_your_gemini_api_key_here"
   GEMINI_MODEL = "gemini-3.8-flash"
   LLM_PROVIDER_ORDER = "groq,gemini"
   ```
5. Click **"Deploy"**! Your application will be live with an SSL HTTPS URL in under 2 minutes.

---

## 🌐 Deployment Option 2: Render.com (Web Service / Docker)

Render supports free web service tiers with automated builds:

### Method A: Native Python Service
1. Create a new **Web Service** on [Render.com](https://render.com/).
2. Connect your GitHub repository.
3. Configure the service settings:
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
4. Under **Environment Variables**, add:
   - `GROQ_API_KEY`: `gsk_...`
   - `GEMINI_API_KEY`: `AIzaSy...` (optional)
   - `LLM_PROVIDER_ORDER`: `groq,gemini`
5. Click **Create Web Service**. Render dynamically injects `$PORT`, which our configuration handles automatically.

### Method B: Docker on Render
1. Select **Docker** as the environment.
2. Render detects the `Dockerfile` automatically.
3. Add the environment variables and deploy.

---

## 🌐 Deployment Option 3: Hugging Face Spaces (Free Cloud Hosting)

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces).
2. Choose **Streamlit** as the Space SDK.
3. Push your repository to the Space remote:
   ```bash
   git remote add space https://huggingface.co/spaces/<username>/<space-name>
   git push --force space main
   ```
4. Navigate to **Settings** -> **Variables and secrets**, and add `GROQ_API_KEY` and `GEMINI_API_KEY`.
5. Hugging Face builds and hosts the app with a public URL.

---

## 🐳 Deployment Option 4: Docker & Docker Compose (VPS, AWS, DigitalOcean)

For self-hosting on Linux, AWS EC2, or DigitalOcean Droplets:

### 1. Using Docker Compose (Recommended)
Clone the repository, create your `.env` file, and run:
```bash
# Start in background with automatic restart
docker compose up -d --build

# View container logs
docker compose logs -f

# Check container health status
docker ps
```
The database is safely persisted in the named Docker volume `sqlite_data`.

### 2. Using Standalone Docker
```bash
# Build the Docker image
docker build -t ai-incident-assistant .

# Run container with persistent local volume and environment file
docker run -d \
  --name ai-incident-assistant \
  -p 8501:8501 \
  --env-file .env \
  -v $(pwd)/incidents.db:/app/incidents.db \
  --restart unless-stopped \
  ai-incident-assistant
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Deployment Option 5: Google Cloud Run (Serverless Container)

Deploy directly to Google Cloud Run in a single command using Google Cloud SDK:

```bash
gcloud run deploy ai-incident-assistant \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GROQ_API_KEY=gsk_...,GROQ_MODEL=llama-3.3-70b-versatile,LLM_PROVIDER_ORDER=groq,gemini
```
Cloud Run automatically scales to zero when idle to minimize costs and provisions instances on incoming web traffic.

---

## 🩺 Health Check & Monitoring

The Docker container includes a built-in healthcheck:
```bash
curl --fail http://localhost:8501/_stcore/health
```
- Returns HTTP `200 OK` when Streamlit is active and ready to serve requests.
- Integrated into Docker Compose, Kubernetes, and AWS ECS readiness probes.

---

## 🔒 Security Best Practices for Production

1. **Never Commit Secrets**: Ensure `.env` is listed in `.dockerignore` and `.gitignore`.
2. **Non-Root Execution**: The provided `Dockerfile` creates and runs under an unprivileged user (`appuser` with UID `1000`).
3. **Prompt Injection Boundaries**: The RAG prompt uses strict `<USER_INPUT>` and `<RETRIEVED_KB>` security delimiters to prevent adversarial jailbreaks.
4. **Deterministic Priority Integrity**: Regardless of LLM generation, the deterministic ITIL matrix verifies and enforces the incident priority.
