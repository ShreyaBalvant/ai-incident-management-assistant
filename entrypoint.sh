#!/bin/sh
set -e

# Default to 8501 if PORT is not set by cloud provider (Render, Cloud Run, Railway, etc.)
PORT="${PORT:-8501}"

echo "Starting AI Incident Management Assistant on port ${PORT}..."
exec streamlit run app.py \
    --server.port="${PORT}" \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=true
