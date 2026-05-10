#!/bin/bash
set -e
exec uvicorn wsgi:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
