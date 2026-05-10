"""
wsgi.py
=======
Clean entry point for Render/Railway/Fly/Heroku.

This file lives at the repo root so uvicorn can be called as:
    uvicorn wsgi:app

It sets sys.path correctly before importing the app.
"""
import sys
import os
from pathlib import Path

# Ensure src/ and api/ are importable from any working directory
ROOT = Path(__file__).parent
for d in [ROOT / "src", ROOT / "api"]:
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

# Set env vars for quiet TF logging
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Import the app - all paths resolve from ROOT since __file__ is set correctly
from main import app  # noqa: E402

__all__ = ["app"]
