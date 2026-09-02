"""Domain model for the interview engine.

These Pydantic models double as the JSON schemas handed to the model for
structured extraction, so field descriptions are part of the prompt surface.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------
# Candidate profile (built offline from resume + GitHub)
# --------------------------------------------------------------------------


class Seniority(str, Enum):
    junior = "junior"
    mid = "mid"
    senior = "senior"


class Role(str, Enum):
    swe = "swe"
    fullstack = "fullstack"
    data_science = "data_science"
    devops = "devops"
    ml = "ml"


class Project(BaseModel):
    name: str
    source: Literal["resume", "github", "both"] = "resume"
    summary: str = Field(description="One or two sentences on what it does.")
    tech: list[str] = Field(default_factory=list, description="Languages, frameworks, datastores.")
    url: str | None = None
    # GitHub-only signals
    stars: int = 0
    last_pushed: str | None = None
    is_fork: bool = False
    collaborators: int = 1
    # What this project makes it fair to ask about.
    probe_topics: list[str] = Field(
        default_factory=list,
        description="Specific things worth asking about, e.g. 'Postgres index choice'.",
    )


class Discrepancy(BaseModel):
    """A resume claim the GitHub evidence does not support, or vice versa."""

    claim: str
    evidence: str
    severity: Literal["low", "medium", "high"] = "low"


class CandidateProfile(BaseModel):
    name: str = "Candidate"
    role: Role = Role.swe
    seniority: Seniority = Seniority.junior
    years_experience: float = 0.0
    education: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    discrepancies: list[Discrepancy] = Field(default_factory=list)
    github_login: str | None = None

    def project(self, name: str) -> Project | None:
        lowered = name.lower()
        return next((p for p in self.projects if p.name.lower() == lowered), None)


# --------------------------------------------------------------------------
# Questions
# --------------------------------------------------------------------------


class Domain(str, Enum):
    behavioral = "behavioral"
    project_deep_dive = "project_deep_dive"
    algorithms = "algorithms"
    systems_design = "systems_design"
    databases = "databases"
    frontend = "frontend"
    backend = "backend"
    devops = "devops"
    ml = "ml"
    communication = "communication"


class Question(BaseModel):
    id: str = Field(default_factory=lambda: _uid("q"))
    text: str
    domain: Domain
    difficulty: int = Field(default=3, ge=1, le=5)
    # What a strong answer contains — drives both heuristic scoring and the
    # model answer shown in fast mode.
    expected_points: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(
        default_factory=list,
        description="Lowercase terms a correct answer is likely to contain.",
    )
    model_answer: str = ""
    project_ref: str | None = None
    # Pre-generated branches so a follow-up needs no model call.
    followup_if_strong: str | None = None
    followup_if_weak: str | None = None
    # An easier restatement used when the candidate is struggling.
    simpler_variant: str | None = None


class QuestionBank(BaseModel):
    profile_key: str
    questions: list[Question] = Field(default_factory=list)
    generated_at: float = Field(default_factory=time.time)

    def by_domain(self, domain: Domain) -> list[Question]:
        return [q for q in self.questions if q.domain == domain]


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


class Verdict(str, Enum):
    excellent = "excellent"
    good = "good"
    partial = "partial"
    wrong = "wrong"
    off_topic = "off_topic"

    @classmethod
    def from_score(cls, score: int) -> "Verdict":
        if score >= 85:
            return cls.excellent
        if score >= 70:
            return cls.good
        if score >= 45:
            return cls.partial
        if score >= 20:
            return cls.wrong
        return cls.off_topic


class ScoreBreakdown(BaseModel):
    keyword_match: int = Field(default=0, ge=0, le=100)
    technical_accuracy: int = Field(default=0, ge=0, le=100)
    completeness: int = Field(default=0, ge=0, le=100)
    communication: int = Field(default=0, ge=0, le=100)


class Evaluation(BaseModel):
    question_id: str
    score: int = Field(ge=0, le=100)
    verdict: Verdict
    breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    bonuses: list[str] = Field(default_factory=list)
    penalties: list[str] = Field(default_factory=list)
    hit_points: list[str] = Field(default_factory=list)
    missed_points: list[str] = Field(default_factory=list)
    feedback: str = ""
    # True when the LLM refinement missed its latency budget and this is the
    # heuristic verdict. Surfaced so the UI can be honest about it.
    heuristic_only: bool = False
    latency_ms: int = 0


class LLMEvaluation(BaseModel):
    """Exactly what the fast model is asked to return. Kept minimal for speed."""

    score: int = Field(ge=0, le=100, description="Overall 0-100.")
    technical_accuracy: int = Field(ge=0, le=100)
    completeness: int = Field(ge=0, le=100)
    communication: int = Field(ge=0, le=100)
    hit_points: list[str] = Field(description="Expected points the answer covered.")
    missed_points: list[str] = Field(description="Expected points the answer missed.")
    bonuses: list[str] = Field(
        description="Credited extras: own-project example, tradeoffs, stated limits."
    )
    penalties: list[str] = Field(
        description="Deductions: vagueness, confident wrongness, recited script."
    )
    feedback: str = Field(description="Two sentences, addressed to the candidate.")
    followup: str = Field(
        description="The single best next question given this answer. One sentence."
    )


# --------------------------------------------------------------------------
# Session
# --------------------------------------------------------------------------


class Mode(str, Enum):
    fast = "fast"
    audio = "audio"


class Turn(BaseModel):
    question: Question
    answer: str = ""
    evaluation: Evaluation | None = None
    asked_at: float = Field(default_factory=time.time)
    answered_at: float | None = None


class SpeechDirective(BaseModel):
    """How the audio mode should say the next line."""

    wpm: int = 130
    tone: str = "neutral"
    lead_in: str = ""


class SessionPhase(str, Enum):
    warmup = "warmup"
    technical = "technical"
    complete = "complete"


class Session(BaseModel):
    id: str = Field(default_factory=lambda: _uid("s"))
    profile: CandidateProfile
    bank: QuestionBank
    mode: Mode = Mode.fast
    phase: SessionPhase = SessionPhase.warmup
    turns: list[Turn] = Field(default_factory=list)
    asked_ids: list[str] = Field(default_factory=list)
    # Follow-ups are minted at runtime with fresh ids, so ids alone don't stop
    # the same wording coming round twice. Normalised text does.
    asked_texts: list[str] = Field(default_factory=list)
    # Rolling per-domain performance, drives adaptation.
    domain_scores: dict[str, list[int]] = Field(default_factory=dict)
    difficulty: float = 2.0
    started_at: float = Field(default_factory=time.time)

    @property
    def current(self) -> Turn | None:
        return self.turns[-1] if self.turns else None

    def domain_mean(self, domain: Domain) -> float | None:
        scores = self.domain_scores.get(domain.value, [])
        return sum(scores) / len(scores) if scores else None


class TurnResult(BaseModel):
    """What the client renders after each answer."""

    session_id: str
    evaluation: Evaluation
    model_answer: str
    next_question: Question | None
    speech: SpeechDirective | None = None
    done: bool = False
    elapsed_ms: int = 0


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


class DomainScore(BaseModel):
    domain: Domain
    mean_score: float
    questions: int


class Report(BaseModel):
    session_id: str
    overall: float
    by_domain: list[DomainScore]
    strengths: list[str]
    growth_areas: list[str]
    next_steps: list[str]
    transcript: list[Turn]
    duration_s: float
