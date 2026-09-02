"""The interview engine: session state, adaptation, and the online turn.

The turn is where the latency budget is spent, so its shape matters:

    heuristic score   (instant, always available)
    llm score         ─┐ raced under a hard budget
    candidate slate   ─┘ computed concurrently, one option per outcome

When the model lands in time we take its verdict and its context-aware
follow-up. When it doesn't, the heuristic verdict and the *pre-generated*
follow-up branch carry the turn — no extra round trip, no stall.
"""

from __future__ import annotations

import asyncio
import time

from app.config import get_settings
from app.interview import bank as bank_mod
from app.interview.scoring import heuristic_score, llm_score
from app.models import (
    CandidateProfile,
    Domain,
    Evaluation,
    Mode,
    Question,
    QuestionBank,
    Session,
    SessionPhase,
    SpeechDirective,
    Turn,
    TurnResult,
    Verdict,
)

# How far the difficulty dial moves per answer, by verdict.
DIFFICULTY_STEP = {
    Verdict.excellent: +0.8,
    Verdict.good: +0.4,
    Verdict.partial: -0.3,
    Verdict.wrong: -0.7,
    Verdict.off_topic: -0.5,
}

# Audio-mode delivery, straight from the spec's adaptation table.
SPEECH = {
    Verdict.excellent: SpeechDirective(
        wpm=150, tone="impressed", lead_in="That's exactly it."
    ),
    Verdict.good: SpeechDirective(wpm=130, tone="engaged", lead_in="Good — let me dig deeper."),
    Verdict.partial: SpeechDirective(
        wpm=110, tone="encouraging", lead_in="You're on the right track."
    ),
    Verdict.wrong: SpeechDirective(
        wpm=95, tone="reassuring", lead_in="No worries — here's the idea."
    ),
    Verdict.off_topic: SpeechDirective(
        wpm=100, tone="clarifying", lead_in="Let me rephrase that."
    ),
}

WEAK_DOMAIN_THRESHOLD = 50.0
STRONG_DOMAIN_THRESHOLD = 80.0


async def start_session(
    profile: CandidateProfile,
    mode: Mode = Mode.fast,
    question_bank: QuestionBank | None = None,
) -> tuple[Session, Question]:
    """Open a session and hand back the first question."""
    question_bank = question_bank or await bank_mod.build_bank(profile)
    session = Session(profile=profile, bank=question_bank, mode=mode)
    first = _select(session) or _fallback_question(profile)
    _ask(session, first)
    return session, first


async def submit_answer(session: Session, answer: str) -> TurnResult:
    """Score the answer, adapt, and return the next question."""
    started = time.perf_counter()
    turn = session.current
    if turn is None:
        raise ValueError("no question is currently open")
    turn.answer = answer
    turn.answered_at = time.time()

    question = turn.question
    fallback = heuristic_score(question, answer, session.profile)

    # The model call and the candidate slate are independent — run them together
    # so the slate is ready the instant the verdict is.
    scored, slate = await asyncio.gather(
        llm_score(question, answer, session.profile),
        _candidate_slate(session, question),
    )

    if scored is not None:
        evaluation, llm_followup = scored
    else:
        evaluation, llm_followup = fallback, None

    turn.evaluation = evaluation
    _record_score(session, question.domain, evaluation.score)
    session.difficulty = max(1.0, min(5.0, session.difficulty + DIFFICULTY_STEP[evaluation.verdict]))

    next_question = _next_question(session, question, evaluation, slate, llm_followup)
    done = next_question is None
    if done:
        session.phase = SessionPhase.complete
    else:
        _ask(session, next_question)

    return TurnResult(
        session_id=session.id,
        evaluation=evaluation,
        model_answer=question.model_answer,
        next_question=next_question,
        speech=SPEECH[evaluation.verdict] if session.mode is Mode.audio else None,
        done=done,
        elapsed_ms=round((time.perf_counter() - started) * 1000),
    )


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


async def _candidate_slate(session: Session, asked: Question) -> dict[str, Question | None]:
    """One pre-picked bank question per possible outcome.

    Pure computation, but it runs as a coroutine so it overlaps the model call
    rather than queueing behind it.
    """
    return {
        "harder": _select(session, difficulty=session.difficulty + 1),
        "same": _select(session, difficulty=session.difficulty),
        "easier": _select(session, difficulty=session.difficulty - 1, domain=asked.domain),
    }


def _next_question(
    session: Session,
    asked: Question,
    evaluation: Evaluation,
    slate: dict[str, Question | None],
    llm_followup: str | None,
) -> Question | None:
    if len(session.turns) >= get_settings().session_length:
        return None

    if len(session.turns) >= get_settings().warmup_questions:
        session.phase = SessionPhase.technical

    # One follow-up per root question, so a weak answer gets help without the
    # interview getting stuck in a loop on it.
    already_followed = asked.id.endswith("-f")
    if not already_followed:
        if follow := _followup(session, asked, evaluation, llm_followup):
            return follow

    if evaluation.verdict in (Verdict.excellent, Verdict.good):
        pick = slate["harder"] or slate["same"]
    elif evaluation.verdict is Verdict.partial:
        pick = slate["same"] or slate["easier"]
    else:
        pick = slate["easier"] or slate["same"]

    # A domain the candidate keeps failing earns more airtime than a fresh one.
    if weak := _weakest_domain(session):
        if targeted := _select(session, difficulty=session.difficulty - 0.5, domain=weak):
            pick = targeted

    return pick or _select(session)


def _followup(
    session: Session, asked: Question, evaluation: Evaluation, llm_followup: str | None
) -> Question | None:
    """Build the follow-up turn, preferring the model's context-aware one."""
    expected = asked.expected_points
    if evaluation.verdict is Verdict.excellent:
        text = llm_followup or asked.followup_if_strong
        difficulty = min(5, asked.difficulty + 1)
    elif evaluation.verdict is Verdict.good:
        text = llm_followup or asked.followup_if_strong
        difficulty = asked.difficulty
    elif evaluation.verdict is Verdict.partial:
        text = llm_followup or asked.followup_if_weak
        difficulty = max(1, asked.difficulty - 1)
    else:
        # Wrong or off-topic: the simpler restatement beats a new follow-up.
        text = asked.simpler_variant or asked.followup_if_weak or llm_followup
        difficulty = max(1, asked.difficulty - 2)
        expected = asked.expected_points[:1]

    if not text or _norm(text) in session.asked_texts:
        return None
    return Question(
        id=f"{asked.id}-f",
        text=text,
        domain=asked.domain,
        difficulty=difficulty,
        expected_points=expected,
        keywords=asked.keywords,
        model_answer=asked.model_answer,
        project_ref=asked.project_ref,
    )


def _select(
    session: Session, difficulty: float | None = None, domain: Domain | None = None
) -> Question | None:
    """Closest unasked question to the target difficulty, respecting the phase."""
    target = session.difficulty if difficulty is None else difficulty
    pool = [
        q for q in session.bank.questions
        if q.id not in session.asked_ids and _norm(q.text) not in session.asked_texts
    ]
    if not pool:
        return None

    warmup = len(session.turns) < get_settings().warmup_questions
    if warmup:
        # Warm-up ramps up rather than opening at the session's default level:
        # the first question should be the easiest one available.
        target = 1.0 + 0.5 * len(session.turns)
        behavioral = [q for q in pool if q.domain is Domain.behavioral]
        # First question after the behavioral openers eases in via a project.
        pool = behavioral or [q for q in pool if q.domain is Domain.project_deep_dive] or pool
    else:
        pool = [q for q in pool if q.domain is not Domain.behavioral] or pool

    if domain is not None:
        pool = [q for q in pool if q.domain is domain] or pool

    if not warmup:
        # Don't ask three databases questions in a row just because they scored well.
        recent = [t.question.domain for t in session.turns[-2:]]
        varied = [q for q in pool if q.domain not in recent]
        pool = varied or pool

    return min(pool, key=lambda q: (abs(q.difficulty - target), -q.difficulty))


def _weakest_domain(session: Session) -> Domain | None:
    """A domain with two or more answers averaging below the weak threshold."""
    worst, worst_mean = None, WEAK_DOMAIN_THRESHOLD
    for name, scores in session.domain_scores.items():
        if len(scores) < 2:
            continue
        mean = sum(scores) / len(scores)
        if mean < worst_mean:
            worst, worst_mean = Domain(name), mean
    return worst


def _record_score(session: Session, domain: Domain, score: int) -> None:
    session.domain_scores.setdefault(domain.value, []).append(score)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _ask(session: Session, question: Question) -> None:
    session.turns.append(Turn(question=question))
    session.asked_ids.append(question.id)
    session.asked_texts.append(_norm(question.text))


def _fallback_question(profile: CandidateProfile) -> Question:
    """Last resort if the bank is empty — the interview still starts."""
    return Question(
        text=(
            f"Hi {profile.name.split()[0]}! Tell me about the most complex thing "
            "you've built and what made it hard."
        ),
        domain=Domain.behavioral,
        difficulty=1,
        expected_points=["A specific project", "The technical difficulty", "Their own role"],
        keywords=["built", "because", "problem"],
        model_answer="A specific project, the concrete difficulty, and your own role in it.",
        followup_if_strong="What would you change about it now?",
        followup_if_weak="What does it do, in one sentence?",
        simpler_variant="What's something you've built that you're proud of?",
    )
