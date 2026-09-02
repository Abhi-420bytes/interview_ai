"""The heuristic scorer is the engine's floor — it has to be right on its own."""

from app.interview.scoring import heuristic_score
from app.models import Domain, Question, Verdict

QUESTION = Question(
    text="When you add an index in Postgres, what are you trading away?",
    domain=Domain.databases,
    difficulty=3,
    expected_points=[
        "Slower writes because every insert maintains the index",
        "Additional disk space",
        "The planner may still ignore it",
    ],
    keywords=["write", "insert", "slower", "space", "disk", "planner"],
)


def test_strong_answer_scores_well(profile):
    ev = heuristic_score(
        QUESTION,
        "You're trading write throughput: every insert and update has to maintain "
        "the index, so writes get slower. It also costs disk space, and the planner "
        "may still ignore it if the predicate doesn't match.",
        profile,
    )
    assert ev.score >= 70
    assert ev.verdict in (Verdict.good, Verdict.excellent)
    assert ev.heuristic_only is True


def test_empty_answer_scores_near_zero(profile):
    ev = heuristic_score(QUESTION, "no idea", profile)
    assert ev.score < 45
    assert "Too short to show your reasoning" in ev.penalties


def test_hedging_is_penalised(profile):
    ev = heuristic_score(QUESTION, "I think it might be slower maybe, not sure", profile)
    assert "Hedged without committing to an answer" in ev.penalties


def test_naming_own_project_earns_the_bonus(profile):
    ev = heuristic_score(
        QUESTION,
        "In Sri Vari Water Solutions the write path got slower once we indexed the "
        "deliveries table, because every insert had to maintain the index, and it "
        "cost us disk space too.",
        profile,
    )
    assert "Grounded the answer in your own project" in ev.bonuses


def test_stating_a_limit_is_rewarded_over_bluffing(profile):
    honest = heuristic_score(
        QUESTION, "Writes get slower. I don't know how the planner decides beyond that.", profile
    )
    bluffing = heuristic_score(
        QUESTION, "Writes get slower and that is the whole story, nothing else.", profile
    )
    assert honest.score > bluffing.score


def test_latency_is_sub_millisecond(profile):
    ev = heuristic_score(QUESTION, "Writes get slower and it costs disk space.", profile)
    assert ev.latency_ms <= 5


def test_missed_points_are_reported(profile):
    ev = heuristic_score(QUESTION, "It costs extra disk space.", profile)
    assert any("writes" in p.lower() for p in ev.missed_points)


BEHAVIORAL = Question(
    text="Tell me about the most technically demanding thing you've built.",
    domain=Domain.behavioral,
    difficulty=1,
    expected_points=["A specific project", "The difficulty", "Their own role", "The outcome"],
    keywords=["built", "because", "problem"],
)


def test_a_good_story_is_not_scored_as_failed_recall(profile):
    """Behavioral answers share almost no vocabulary with their expected points."""
    ev = heuristic_score(
        BEHAVIORAL,
        "I built Sri Vari Water Solutions, a delivery platform. The hard part was "
        "real-time tracking, because drivers lose signal, so I implemented state "
        "reconciliation on reconnect. It ended up being the most reliable part.",
        profile,
    )
    assert ev.verdict in (Verdict.good, Verdict.excellent), ev.score
    assert "Named a specific project or technology" in ev.hit_points


def test_a_vague_story_scores_below_a_specific_one(profile):
    specific = heuristic_score(
        BEHAVIORAL,
        "I built Sri Vari Water Solutions and I implemented the tracking myself, "
        "because polling was too slow. It ended up working well.",
        profile,
    )
    vague = heuristic_score(
        BEHAVIORAL, "We worked on some hard stuff on a team project last year.", profile
    )
    assert specific.score > vague.score


def test_narrative_scoring_refuses_the_extremes(profile):
    """Structure is not correctness — the heuristic won't pretend otherwise."""
    from app.interview.scoring import NARRATIVE_CEILING, NARRATIVE_FLOOR

    worst = heuristic_score(BEHAVIORAL, "no", profile)
    assert worst.score >= NARRATIVE_FLOOR
    best = heuristic_score(
        BEHAVIORAL,
        "I built Sri Vari Water Solutions with postgresql because we needed "
        "transactions, and I implemented it myself. It ended up shipping. I don't "
        "know how it scales past that. The tradeoff was write latency.",
        profile,
    )
    assert best.score <= NARRATIVE_CEILING
