#!/bin/bash
set -e

echo "[entrypoint] Iniciando API FastAPI (puerto interno 8000)..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

echo "[entrypoint] Esperando a que la API cargue los modelos..."
sleep 8

echo "[entrypoint] Iniciando dashboard Streamlit (puerto público ${PORT:-8501})..."
streamlit run dashboard/app.py \
    --server.port="${PORT:-8501}" \
    --server.address=0.0.0.0 \
    --server.headless=true