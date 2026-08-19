#!/usr/bin/env bash
set -e

# Ingest + build the search index on first run only (data/ is a volume,
# so this is skipped on subsequent container restarts).
if [ ! -f "data/chunks.json" ]; then
    echo "No knowledge base found -- running ingestion..."
    python -m app.ingest
fi

if [ ! -f "data/search_index.pkl" ]; then
    echo "Building search index..."
    python -c "from app.search import build_index; build_index()"
fi

exec streamlit run app/streamlit_app.py \
    --server.address=0.0.0.0 \
    --server.port=8501 \
    --server.headless=true
