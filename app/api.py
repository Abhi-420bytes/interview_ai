"""HTTP surface.

Two phases, deliberately split: `/profile` and `/bank` are the slow
pre-interview indexing, `/sessions/*` is the latency-critical loop. A client
that has already indexed a candidate can go straight to `/sessions`.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import get_settings
from app.ingest import resume as resume_mod
from app.ingest.profile import build_profile, profile_key
from app.interview import bank as bank_mod
from app.interview import engine, report as report_mod
from app.llm.client import usage_totals
from app.models import (
    CandidateProfile,
    Mode,
    Question,
    QuestionBank,
    Report,
    TurnResult,
)
from app.store import get_store

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="AI Interview Coach",
    description="Adaptive mock technical interviews grounded in a candidate's resume and GitHub.",
    version="0.1.0",
)


# --------------------------------------------------------------------------
# Pre-interview indexing
# --------------------------------------------------------------------------


class ProfileRequest(BaseModel):
    resume_text: str | None = None
    github_login: str | None = None
    repo_limit: int = Field(default=5, ge=1, le=20)


@app.post("/profile", response_model=CandidateProfile)
async def create_profile(request: ProfileRequest) -> CandidateProfile:
    if not request.resume_text and not request.github_login:
        raise HTTPException(400, "provide resume_text, github_login, or both")
    return await build_profile(
        resume_text=request.resume_text,
        github_login=request.github_login,
        repo_limit=request.repo_limit,
    )


@app.post("/profile/upload", response_model=CandidateProfile)
async def upload_resume(
    file: UploadFile, github_login: str | None = None
) -> CandidateProfile:
    """PDF or text resume upload."""
    import tempfile
    from pathlib import Path

    suffix = Path(file.filename or "resume.txt").suffix or ".txt"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        text = resume_mod.read_resume(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not text.strip():
        raise HTTPException(400, "could not extract any text from that file")
    return await build_profile(resume_text=text, github_login=github_login)


class BankRequest(BaseModel):
    profile: CandidateProfile
    size: int = Field(default=50, ge=10, le=120)


@app.post("/bank", response_model=QuestionBank)
async def create_bank(request: BankRequest) -> QuestionBank:
    return await bank_mod.build_bank(request.profile, size=request.size)


# --------------------------------------------------------------------------
# Interview loop
# --------------------------------------------------------------------------


class StartRequest(BaseModel):
    profile: CandidateProfile
    mode: Mode = Mode.fast


class StartResponse(BaseModel):
    session_id: str
    question: Question
    bank_size: int


@app.post("/sessions", response_model=StartResponse)
async def start(request: StartRequest) -> StartResponse:
    session, question = await engine.start_session(request.profile, mode=request.mode)
    get_store().put(session)
    return StartResponse(
        session_id=session.id, question=question, bank_size=len(session.bank.questions)
    )


class AnswerRequest(BaseModel):
    answer: str = Field(min_length=1)


@app.post("/sessions/{session_id}/answer", response_model=TurnResult)
async def answer(session_id: str, request: AnswerRequest) -> TurnResult:
    store = get_store()
    session = store.get(session_id)
    if session is None:
        raise HTTPException(404, "unknown session")
    if session.current is None or session.current.answered_at is not None:
        raise HTTPException(409, "no question is currently open")

    result = await engine.submit_answer(session, request.answer)
    store.put(session)
    return result


@app.get("/sessions/{session_id}/report", response_model=Report)
async def session_report(session_id: str) -> Report:
    session = get_store().get(session_id)
    if session is None:
        raise HTTPException(404, "unknown session")
    return await report_mod.build_report(session)


@app.get("/sessions/{session_id}")
async def session_state(session_id: str):
    session = get_store().get(session_id)
    if session is None:
        raise HTTPException(404, "unknown session")
    return {
        "id": session.id,
        "phase": session.phase,
        "mode": session.mode,
        "difficulty": round(session.difficulty, 2),
        "answered": sum(1 for t in session.turns if t.evaluation),
        "of": get_settings().session_length,
        "domain_scores": {
            name: round(sum(s) / len(s), 1) for name, s in session.domain_scores.items() if s
        },
        "current_question": session.current.question if session.current else None,
    }


@app.get("/health")
async def health():
    settings = get_settings()
    usage = usage_totals()
    return {
        "status": "ok",
        "offline": settings.offline,
        "fast_model": settings.fast_model,
        "deep_model": settings.deep_model,
        "eval_budget_s": settings.eval_budget_s,
        "usage": usage.model_dump(),
    }
