"""Session flow: adaptation, follow-ups, and termination."""

import pytest

from app.config import get_settings
from app.interview import bank as bank_mod, engine
from app.models import Mode, Verdict


def strong_answer(question) -> str:
    """A candidate who genuinely covers what this question asked for.

    Built from the question's own expected points so it works against any
    question the engine picks — the heuristic scorer is keyword-driven, so an
    excellent answer to the *wrong* question correctly scores badly.
    """
    return (
        "In Sri Vari Water Solutions, "
        + " ".join(question.expected_points)
        + ". "
        + " ".join(question.keywords)
        + ". I chose it because the tradeoff was worth it, rather than the "
        "alternative. I don't know how it behaves past that scale."
    )


WEAK = "Um, I think maybe indexes, not sure."


@pytest.fixture
async def started(profile):
    question_bank = await bank_mod.build_bank(profile)
    return await engine.start_session(profile, question_bank=question_bank)


async def test_bank_is_grounded_in_real_projects(profile):
    question_bank = await bank_mod.build_bank(profile)
    assert len(question_bank.questions) >= 10
    for q in question_bank.questions:
        if q.project_ref:
            assert profile.project(q.project_ref) is not None


async def test_interview_opens_with_behavioral(started):
    session, question = started
    assert question.domain.value == "behavioral"
    assert session.phase.value == "warmup"


async def test_strong_answers_raise_difficulty(started):
    session, question = started
    before = session.difficulty
    result = await engine.submit_answer(session, strong_answer(question))
    assert result.evaluation.verdict in (Verdict.good, Verdict.excellent)
    assert session.difficulty > before


async def test_weak_answers_lower_difficulty(started):
    session, _ = started
    before = session.difficulty
    result = await engine.submit_answer(session, WEAK)
    assert session.difficulty < before
    assert result.evaluation.verdict in (Verdict.partial, Verdict.wrong, Verdict.off_topic)


async def test_weak_answer_gets_an_easier_followup(started):
    session, question = started
    result = await engine.submit_answer(session, WEAK)
    assert result.next_question is not None
    assert result.next_question.difficulty <= question.difficulty


async def test_a_question_is_never_asked_twice(started):
    session, _ = started
    for _ in range(get_settings().session_length + 2):
        result = await engine.submit_answer(session, strong_answer(session.current.question))
        if result.done or result.next_question is None:
            break
    ids = [t.question.id for t in session.turns]
    assert len(ids) == len(set(ids))


async def test_session_terminates_at_configured_length(started):
    session, _ = started
    for _ in range(get_settings().session_length + 5):
        result = await engine.submit_answer(session, strong_answer(session.current.question))
        if result.done:
            break
    assert result.done is True
    assert session.phase.value == "complete"
    assert len(session.turns) <= get_settings().session_length


async def test_repeated_weakness_pulls_the_domain_back(profile):
    """Two bad answers in a domain should make the engine return to it."""
    question_bank = await bank_mod.build_bank(profile)
    session, _ = await engine.start_session(profile, question_bank=question_bank)
    for _ in range(6):
        result = await engine.submit_answer(session, WEAK)
        if result.done:
            break
    revisited = [t.question.domain for t in session.turns]
    assert len(revisited) > len(set(revisited)), "expected a domain to be revisited"


async def test_audio_mode_returns_a_speech_directive(profile):
    question_bank = await bank_mod.build_bank(profile)
    session, _ = await engine.start_session(profile, mode=Mode.audio, question_bank=question_bank)
    result = await engine.submit_answer(session, WEAK)
    assert result.speech is not None
    assert result.speech.wpm <= 130, "struggling candidates get a slower delivery"


async def test_fast_mode_has_no_speech_directive(started):
    session, question = started
    result = await engine.submit_answer(session, strong_answer(question))
    assert result.speech is None


async def test_offline_turn_is_well_inside_the_latency_budget(started):
    session, question = started
    result = await engine.submit_answer(session, strong_answer(question))
    assert result.elapsed_ms < 100
    assert result.evaluation.heuristic_only is True


async def test_warmup_ramps_from_the_easiest_question(started):
    """The opener should be the gentlest available, not the session default."""
    _, question = started
    assert question.difficulty == 1


async def test_no_question_is_repeated_verbatim(profile):
    """Follow-ups get fresh ids, so id-dedupe alone would let wording repeat."""
    question_bank = await bank_mod.build_bank(profile)
    session, _ = await engine.start_session(profile, question_bank=question_bank)
    for _ in range(get_settings().session_length + 2):
        result = await engine.submit_answer(session, WEAK)
        if result.done or result.next_question is None:
            break
    texts = [t.question.text.lower() for t in session.turns]
    assert len(texts) == len(set(texts))


async def test_walking_back_a_question_lowers_what_it_expects(started):
    """A simpler restatement must not be graded against the full expected points."""
    session, question = started
    result = await engine.submit_answer(session, "no idea")
    follow = result.next_question
    assert follow is not None
    assert len(follow.expected_points) <= len(question.expected_points)
