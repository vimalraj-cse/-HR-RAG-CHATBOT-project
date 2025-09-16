# backend/app/__init__.py
"""
This package contains the backend app for the RAG Chatbot.
It initializes the backend module for FastAPI.
"""

# You can leave this as a marker file.
# Or explicitly import your API for convenience:

from .api import app

__all__ = ["app"]
