#!/bin/bash
# Render/Railway startup script - sets PYTHONPATH and launches API
set -e
export PYTHONPATH="${PYTHONPATH}:src:api"
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
