"""Answer evaluation.

Two scorers run against every answer:

* `heuristic_score` is pure Python and returns in well under a millisecond. It
  is the floor — the interview never waits on anything to have a verdict.
* `llm_score` asks the fast model for a real judgement, under a hard latency
  budget. If it lands in time it replaces the heuristic; if not, the heuristic
  stands and the result is flagged `heuristic_only`.

The weights follow the spec: keywords 40, technical accuracy 35, completeness
15, communication 10, then bonuses and penalties.
"""

from __future__ import annotations

import re
import time

from app.config import get_settings
from app.llm import client as llm
from app.models import (
    CandidateProfile,
    Domain,
    Evaluation,
    LLMEvaluation,
    Question,
    ScoreBreakdown,
    Verdict,
)

WEIGHTS = {"keyword_match": 0.40, "technical_accuracy": 0.35, "completeness": 0.15,
           "communication": 0.10}

# Phrases that signal the candidate is guessing rather than knowing.
HEDGES = ["i think", "i guess", "maybe", "might be", "not sure", "probably",
          "i believe", "sort of", "kind of", "or something"]
# Phrases that signal reasoning about tradeoffs.
TRADEOFF_MARKERS = ["tradeoff", "trade-off", "instead of", "rather than", "downside",
                    "the cost is", "in exchange", "versus", " vs ", "but it costs"]
# Phrases that signal honest limits, which the spec rewards most.
LIMIT_MARKERS = ["i don't know", "i haven't", "i'd have to look", "i'm not certain",
                 "we never tested", "that's a gap", "i've not used"]

# Behavioral answers are stories, not recall, so keyword overlap says almost
# nothing about them. These four structural signals do, and they are checkable.
OWNERSHIP_MARKERS = ["i built", "i wrote", "i implemented", "i chose", "i designed",
                     "i migrated", "i added", "i decided", "i led", "i fixed", "i learned",
                     "my job", "i was responsible", "i handled", "i refactored"]
REASON_MARKERS = ["because", "since", "so that", "the reason", "which meant", "in order to"]
OUTCOME_MARKERS = ["ended up", "turned out", "we shipped", "it worked", "the result",
                   "in the end", "afterwards", "now it", "since then", "it cost", "took about"]

# Domains where the story-shaped scorer applies instead of keyword matching.
NARRATIVE_DOMAINS = {Domain.behavioral, Domain.communication}

# The narrative scorer reads structure, not correctness, so it refuses to emit
# a verdict at either extreme — the model path is what grades these properly.
NARRATIVE_FLOOR, NARRATIVE_CEILING = 35, 88


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_.+#-]+", text.lower())


def _normalise(term: str) -> str:
    return re.sub(r"[^a-z0-9]", "", term.lower())


def heuristic_score(question: Question, answer: str, profile: CandidateProfile) -> Evaluation:
    """Instant, deterministic scoring. Never blocks, never fails."""
    started = time.perf_counter()
    lowered = answer.lower()
    tokens = set(_words(answer))
    normalised = {_normalise(t) for t in tokens}

    hits = [
        kw for kw in question.keywords
        if _normalise(kw) in normalised or kw.lower() in lowered
    ]
    keyword_match = round(100 * len(hits) / len(question.keywords)) if question.keywords else 50

    # Which expected points are plausibly covered: a point counts if a third of
    # its content words show up in the answer.
    hit_points, missed_points = [], []
    for point in question.expected_points:
        content = [w for w in _words(point) if len(w) > 3]
        if not content:
            continue
        overlap = sum(1 for w in content if w in tokens or _normalise(w) in normalised)
        (hit_points if overlap >= max(1, len(content) // 3) else missed_points).append(point)
    completeness = (
        round(100 * len(hit_points) / len(question.expected_points))
        if question.expected_points else 50
    )

    word_count = len(_words(answer))
    communication = _communication_score(word_count, answer)

    if question.domain in NARRATIVE_DOMAINS:
        keyword_match, technical_accuracy, completeness, hit_points, missed_points = (
            _narrative_signals(lowered, profile)
        )
    else:
        # Without a model, "technical accuracy" can only be a proxy: does the
        # answer engage with the question's own vocabulary at all?
        technical_accuracy = min(100, round(keyword_match * 0.7 + completeness * 0.3))

    breakdown = ScoreBreakdown(
        keyword_match=keyword_match,
        technical_accuracy=technical_accuracy,
        completeness=completeness,
        communication=communication,
    )
    score = sum(getattr(breakdown, field) * weight for field, weight in WEIGHTS.items())

    bonuses, penalties = [], []
    if any(p.name.lower() in lowered for p in profile.projects):
        score += 10
        bonuses.append("Grounded the answer in your own project")
    if any(m in lowered for m in TRADEOFF_MARKERS):
        score += 5
        bonuses.append("Weighed a tradeoff")
    if any(m in lowered for m in LIMIT_MARKERS):
        score += 15
        bonuses.append("Named the limit of what you know")

    hedge_count = sum(lowered.count(h) for h in HEDGES)
    if hedge_count and word_count < 60:
        score -= 5
        penalties.append("Hedged without committing to an answer")
    if word_count < 12:
        score -= 10
        penalties.append("Too short to show your reasoning")

    score = max(0, min(100, round(score)))
    if question.domain in NARRATIVE_DOMAINS:
        score = max(NARRATIVE_FLOOR, min(NARRATIVE_CEILING, score))
    return Evaluation(
        question_id=question.id,
        score=score,
        verdict=Verdict.from_score(score),
        breakdown=breakdown,
        bonuses=bonuses,
        penalties=penalties,
        hit_points=hit_points,
        missed_points=missed_points,
        feedback=_heuristic_feedback(score, missed_points),
        heuristic_only=True,
        latency_ms=round((time.perf_counter() - started) * 1000),
    )


def _narrative_signals(
    lowered: str, profile: CandidateProfile
) -> tuple[int, int, int, list[str], list[str]]:
    """Score a behavioral answer on the four things a good one always has."""
    named_project = any(p.name.lower() in lowered for p in profile.projects) or any(
        t in lowered for p in profile.projects for t in p.tech
    )
    checks = [
        ("Named a specific project or technology", named_project),
        ("Said what you personally did", any(m in lowered for m in OWNERSHIP_MARKERS)),
        ("Gave the reasoning behind a decision", any(m in lowered for m in REASON_MARKERS)),
        ("Said how it turned out", any(m in lowered for m in OUTCOME_MARKERS)),
    ]
    hit = [label for label, ok in checks if ok]
    missed = [label for label, ok in checks if not ok]
    coverage = round(100 * len(hit) / len(checks))
    return coverage, coverage, coverage, hit, missed


def _communication_score(word_count: int, answer: str) -> int:
    if word_count < 8:
        return 20
    if word_count > 400:
        return 60  # rambling
    score = 70
    if re.search(r"\b(first|second|then|because|so that|which means)\b", answer.lower()):
        score += 20  # structured reasoning
    if answer.count(".") >= 2:
        score += 10
    return min(100, score)


def _heuristic_feedback(score: int, missed: list[str]) -> str:
    if score >= 85:
        return "Strong, specific answer."
    if score >= 70:
        return "Solid. " + (f"Worth also covering: {missed[0]}" if missed else "")
    if score >= 45:
        return "Partly there. " + (f"You didn't get to: {missed[0]}" if missed else "")
    return "That didn't land. " + (f"The key idea is: {missed[0]}" if missed else "")


EVAL_SYSTEM = """You score one answer in a technical mock interview and pick the \
next question. You are terse, fair, and fast.

Scoring — weight these as: keyword/concept coverage 40%, technical accuracy 35%, \
completeness 15%, communication 10%. Then adjust:
  +10  they grounded it in their own project with specifics
  +5   they named a real tradeoff or edge case
  +15  they stated the limit of what they know instead of bluffing
  -5   vague hedging with no commitment ("I think it might be...")
  -10  stated confidently and wrong
  -20  a recited definition that doesn't answer what was asked

Calibrate to a real interview loop: 85+ means a strong hire signal on this \
question, 70-84 solid, 45-69 partial, below 45 didn't land. Do not inflate — \
a fluent answer that is technically empty scores low, and a rough answer that \
gets the mechanism right scores well.

`feedback` is two sentences, spoken to the candidate, second person. Say what \
they got and what was missing. No preamble, no praise sandwich.

`followup` is the single best next question given exactly what they just said — \
escalate if they were strong, narrow to the nearest unblocking sub-question if \
they were weak. One sentence."""


def eval_user_prompt(question: Question, answer: str) -> str:
    return (
        f"<question difficulty=\"{question.difficulty}\" domain=\"{question.domain.value}\">\n"
        f"{question.text}\n</question>\n\n"
        f"<expected_points>\n"
        + "\n".join(f"- {p}" for p in question.expected_points)
        + f"\n</expected_points>\n\n<answer>\n{answer.strip()}\n</answer>"
    )


def profile_context(profile: CandidateProfile) -> str:
    """The stable half of the eval prompt — cached across the whole session."""
    projects = "\n".join(
        f"- {p.name}: {p.summary} [{', '.join(p.tech[:6])}]" for p in profile.projects[:6]
    )
    return (
        f"{EVAL_SYSTEM}\n\n"
        f"<candidate>\n"
        f"{profile.name} — {profile.role.value}, {profile.seniority.value}, "
        f"{profile.years_experience} years\n"
        f"skills: {', '.join(profile.skills[:40])}\n"
        f"projects:\n{projects}\n"
        f"</candidate>"
    )


async def llm_score(
    question: Question, answer: str, profile: CandidateProfile
) -> tuple[Evaluation, str | None] | None:
    """Model-scored evaluation plus its suggested follow-up, or None if too slow."""
    settings = get_settings()
    started = time.perf_counter()
    result = await llm.structured(
        schema=LLMEvaluation,
        system=profile_context(profile),
        user=eval_user_prompt(question, answer),
        model=settings.fast_model,
        max_tokens=settings.eval_max_tokens,
        budget_s=settings.eval_budget_s,
    )
    if result is None:
        return None

    keyword_hits = [kw for kw in question.keywords if kw.lower() in answer.lower()]
    evaluation = Evaluation(
        question_id=question.id,
        score=max(0, min(100, result.score)),
        verdict=Verdict.from_score(max(0, min(100, result.score))),
        breakdown=ScoreBreakdown(
            keyword_match=(
                round(100 * len(keyword_hits) / len(question.keywords))
                if question.keywords else 50
            ),
            technical_accuracy=result.technical_accuracy,
            completeness=result.completeness,
            communication=result.communication,
        ),
        bonuses=result.bonuses,
        penalties=result.penalties,
        hit_points=result.hit_points,
        missed_points=result.missed_points,
        feedback=result.feedback,
        heuristic_only=False,
        latency_ms=round((time.perf_counter() - started) * 1000),
    )
    return evaluation, result.followup
