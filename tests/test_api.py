"""The HTTP surface, offline end to end."""

import pytest
from fastapi.testclient import TestClient

from app.api import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_reports_offline_mode(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["offline"] is True
    assert body["fast_model"] == "claude-haiku-4-5"


def test_profile_requires_some_input(client):
    assert client.post("/profile", json={}).status_code == 400


def test_profile_from_resume_text(client):
    body = client.post(
        "/profile",
        json={"resume_text": "Skills: Python, Django, PostgreSQL\nExperience: 3 years"},
    ).json()
    assert "django" in body["skills"]
    assert body["years_experience"] == 3.0


def test_full_interview_over_http(client, profile):
    payload = profile.model_dump(mode="json")
    started = client.post("/sessions", json={"profile": payload, "mode": "fast"}).json()
    session_id = started["session_id"]
    assert started["bank_size"] >= 10
    assert started["question"]["text"]

    answered = 0
    while True:
        response = client.post(
            f"/sessions/{session_id}/answer",
            json={"answer": "We used Postgres because writes and disk space were the tradeoff."},
        )
        assert response.status_code == 200
        result = response.json()
        answered += 1
        assert 0 <= result["evaluation"]["score"] <= 100
        if result["done"] or answered > 20:
            break
    assert result["done"] is True

    report = client.get(f"/sessions/{session_id}/report").json()
    assert 0 <= report["overall"] <= 100
    assert report["by_domain"]
    assert len(report["transcript"]) == answered


def test_unknown_session_is_404(client):
    assert client.post("/sessions/nope/answer", json={"answer": "hi"}).status_code == 404
    assert client.get("/sessions/nope/report").status_code == 404


def test_session_state_tracks_progress(client, profile):
    started = client.post("/sessions", json={"profile": profile.model_dump(mode="json")}).json()
    session_id = started["session_id"]
    client.post(f"/sessions/{session_id}/answer", json={"answer": "A reasonable answer here."})
    state = client.get(f"/sessions/{session_id}").json()
    assert state["answered"] == 1
    assert state["current_question"] is not None


def test_empty_answer_is_rejected(client, profile):
    started = client.post("/sessions", json={"profile": profile.model_dump(mode="json")}).json()
    response = client.post(
        f"/sessions/{started['session_id']}/answer", json={"answer": ""}
    )
    assert response.status_code == 422
