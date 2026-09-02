"""uvicorn entry point: `uvicorn app.main:app --reload`."""

from app.api import app

__all__ = ["app"]
