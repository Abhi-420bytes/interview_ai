import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Every test runs offline: deterministic, no network, no spend.
os.environ["INTERVIEW_OFFLINE"] = "true"

from app.config import reset_settings  # noqa: E402
from app.models import CandidateProfile, Project, Role, Seniority  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERVIEW_DATA_DIR", str(tmp_path))
    reset_settings()
    import app.store

    app.store._store = None
    yield
    reset_settings()
    app.store._store = None


@pytest.fixture
def profile() -> CandidateProfile:
    return CandidateProfile(
        name="Abhiram Challa",
        role=Role.fullstack,
        seniority=Seniority.junior,
        years_experience=1.0,
        skills=["python", "javascript", "django", "react", "postgresql", "docker"],
        projects=[
            Project(
                name="Sri Vari Water Solutions",
                summary="Water delivery platform with real-time tracking.",
                tech=["next.js", "postgresql", "socket.io", "twilio"],
                probe_topics=["Postgres index choice", "Socket.io fan-out", "Twilio retries"],
            ),
            Project(
                name="Legal Document Analysis",
                summary="NLP pipeline for contract clause extraction.",
                tech=["python", "fastapi", "transformers"],
                probe_topics=["chunking long contracts"],
            ),
        ],
        strengths=["full-stack delivery"],
        gaps=["no evidence of testing"],
    )
