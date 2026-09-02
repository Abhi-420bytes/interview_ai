"""Post-interview report.

The numbers are computed locally so a report always exists; the narrative
(strengths, growth areas, next steps) is written by the deep model when
credentials allow, because that's the part a candidate actually reads.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field

from app.config import get_settings
from app.llm import client as llm
from app.models import Domain, DomainScore, Report, Session


class Narrative(BaseModel):
    strengths: list[str] = Field(
        description="Three things they did well, each citing what they actually said."
    )
    growth_areas: list[str] = Field(
        description="Three things to work on, each naming the specific gap, not the topic."
    )
    next_steps: list[str] = Field(
        description=(
            "Three concrete actions, each doable in a week. Name the thing to "
            "build, read, or measure — not 'study system design'."
        )
    )


NARRATIVE_SYSTEM = """You write the closing feedback for a technical mock interview.

You get the full transcript with per-answer scores. Write for the candidate, in \
second person, plainly. Quote or paraphrase what they actually said — generic \
advice is worthless here.

Be honest about weak performance without being discouraging: name the specific \
missing knowledge, not a character judgement. If they were strong, say so \
without padding."""


def build_report_sync(session: Session) -> Report:
    """Scores-only report. Always available, no model needed."""
    by_domain = _domain_scores(session)
    scored = [t.evaluation.score for t in session.turns if t.evaluation]
    overall = round(sum(scored) / len(scored), 1) if scored else 0.0

    ranked = sorted(by_domain, key=lambda d: d.mean_score, reverse=True)
    return Report(
        session_id=session.id,
        overall=overall,
        by_domain=by_domain,
        strengths=[f"{d.domain.value} ({d.mean_score:.0f}/100)" for d in ranked[:3]],
        growth_areas=[f"{d.domain.value} ({d.mean_score:.0f}/100)" for d in reversed(ranked[-3:])],
        next_steps=_default_next_steps(ranked),
        transcript=session.turns,
        duration_s=round(time.time() - session.started_at, 1),
    )


async def build_report(session: Session) -> Report:
    report = build_report_sync(session)
    narrative = await llm.structured(
        schema=Narrative,
        system=NARRATIVE_SYSTEM,
        user=_transcript(session),
        model=get_settings().deep_model,
        max_tokens=get_settings().precompute_max_tokens,
        effort="medium",
    )
    if narrative:
        report.strengths = narrative.strengths
        report.growth_areas = narrative.growth_areas
        report.next_steps = narrative.next_steps
    return report


def _domain_scores(session: Session) -> list[DomainScore]:
    return [
        DomainScore(
            domain=Domain(name), mean_score=round(sum(scores) / len(scores), 1),
            questions=len(scores),
        )
        for name, scores in session.domain_scores.items()
        if scores
    ]


def _default_next_steps(ranked: list[DomainScore]) -> list[str]:
    if not ranked:
        return ["Run a full session to get a baseline."]
    weakest = ranked[-1].domain
    library = {
        Domain.databases: "Take one slow query from your own project, run EXPLAIN ANALYZE, and write down what the plan changed after you indexed it.",
        Domain.systems_design: "Draw your last project's architecture and mark every point that breaks at 100x traffic.",
        Domain.algorithms: "Work through complexity analysis on the data structures you already use, out loud.",
        Domain.behavioral: "Write out three project stories in problem / action / outcome form and time yourself telling each in two minutes.",
        Domain.frontend: "Profile one of your React pages and find what re-renders that shouldn't.",
        Domain.backend: "Add request tracing to one endpoint and find where the time actually goes.",
        Domain.devops: "Containerise one project end to end, then break it deliberately and read the logs back.",
        Domain.ml: "Take one model you trained and write down the failure modes you never tested for.",
        Domain.project_deep_dive: "For each project on your resume, write the three decisions you made and the alternatives you rejected.",
        Domain.communication: "Record yourself answering one technical question and cut it to ninety seconds.",
    }
    steps = [library.get(weakest, "Pick your weakest domain and build one small thing in it.")]
    if len(ranked) > 1:
        steps.append(library.get(ranked[-2].domain, "Revisit your second-weakest area."))
    steps.append(f"Re-run this interview in a week and compare the {weakest.value} score.")
    return steps


def _transcript(session: Session) -> str:
    parts = [
        f"Candidate: {session.profile.name} — {session.profile.role.value}, "
        f"{session.profile.seniority.value}"
    ]
    for i, turn in enumerate(session.turns, 1):
        if not turn.evaluation:
            continue
        parts.append(
            f"\nQ{i} [{turn.question.domain.value}, difficulty {turn.question.difficulty}]: "
            f"{turn.question.text}\n"
            f"A: {turn.answer}\n"
            f"Score: {turn.evaluation.score} ({turn.evaluation.verdict.value})"
            f"{' [heuristic only]' if turn.evaluation.heuristic_only else ''}\n"
            f"Missed: {'; '.join(turn.evaluation.missed_points) or 'nothing'}"
        )
    return "\n".join(parts)
