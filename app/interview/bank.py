"""Pre-computed question bank.

Everything expensive happens here, once, before the interview starts: the deep
model writes a bank of questions grounded in the candidate's actual projects,
each carrying its own expected points, keywords, model answer, and both
follow-up branches. At interview time the engine only *selects* from this —
which is what keeps the online path inside its latency budget.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import get_settings
from app.ingest.profile import profile_key
from app.llm import client as llm
from app.models import CandidateProfile, Domain, Question, QuestionBank, Role

log = logging.getLogger(__name__)


class GeneratedBank(BaseModel):
    questions: list[Question] = Field(
        description="At least 50 questions spanning the domains and difficulties requested."
    )


BANK_SYSTEM = """You write the question bank for a technical mock interview, \
tailored to one specific candidate.

You get a candidate profile built from their resume and public GitHub repos. \
Write questions only that profile supports — never invent a project, a \
technology, or a claim they did not make.

Requirements for the bank:

- **Coverage.** Span the requested domains. Within each domain, spread \
difficulty 1-5 so the engine has somewhere to go in both directions.
- **Grounding.** Project deep-dive questions must name the real project and the \
real decision inside it, and set `project_ref` to that project's name. \
"In <project> you used Postgres — what made you pick it over a document store, \
and where did that choice cost you?" is right; "tell me about databases" is not.
- **Probing the gaps.** The profile lists gaps and discrepancies. Write \
questions that reach them without being hostile — a discrepancy is a thing to \
be curious about, not to catch someone out.
- **`expected_points`** are the 3-5 things a strong answer contains. They are \
scored against, so make them checkable, not vague.
- **`keywords`** are lowercase terms a correct answer plausibly contains, used \
for instant local scoring. Include synonyms — an answer saying "B-tree" and one \
saying "btree" are the same answer.
- **`model_answer`** is 2-3 sentences at the level a strong candidate would \
actually speak, not an essay.
- **Both branches, always.** `followup_if_strong` escalates (edge cases, \
tradeoffs, scale). `followup_if_weak` narrows to the nearest simpler thing that \
would unblock them. `simpler_variant` restates the question one rung lower.
- **Behavioral questions** (difficulty 1-2) reference their real work, and their \
`expected_points` describe structure and specificity, not correctness."""

# Which domains are worth asking about, per target role.
ROLE_DOMAINS: dict[Role, list[Domain]] = {
    Role.swe: [Domain.algorithms, Domain.systems_design, Domain.backend, Domain.databases],
    Role.fullstack: [Domain.frontend, Domain.backend, Domain.databases, Domain.systems_design],
    Role.data_science: [Domain.ml, Domain.databases, Domain.algorithms, Domain.communication],
    Role.devops: [Domain.devops, Domain.systems_design, Domain.backend, Domain.databases],
    Role.ml: [Domain.ml, Domain.systems_design, Domain.algorithms, Domain.databases],
}


def target_domains(profile: CandidateProfile) -> list[Domain]:
    return [Domain.behavioral, Domain.project_deep_dive] + ROLE_DOMAINS.get(
        profile.role, ROLE_DOMAINS[Role.swe]
    )


async def build_bank(profile: CandidateProfile, size: int = 50) -> QuestionBank:
    """Generate (or load) the bank for this profile."""
    key = profile_key(profile)
    if cached := load_bank(key):
        log.info("question bank cache hit (%s, %d questions)", key, len(cached.questions))
        return cached

    domains = target_domains(profile)
    generated = await llm.structured(
        schema=GeneratedBank,
        system=BANK_SYSTEM,
        user=(
            f"Write {size} questions for this candidate.\n\n"
            f"Domains to cover: {', '.join(d.value for d in domains)}\n"
            f"Target role: {profile.role.value} | Seniority: {profile.seniority.value}\n\n"
            f"<profile>\n{profile.model_dump_json(indent=2)}\n</profile>"
        ),
        model=get_settings().deep_model,
        max_tokens=get_settings().precompute_max_tokens,
        effort="high",
    )

    questions = generated.questions if generated else template_bank(profile)
    if generated:
        questions = [q for q in questions if _grounded(q, profile)]
        if len(questions) < 10:
            log.warning("generated bank too thin after grounding; adding templates")
            questions += template_bank(profile)

    bank = QuestionBank(profile_key=key, questions=questions)
    save_bank(bank)
    return bank


def _grounded(q: Question, profile: CandidateProfile) -> bool:
    """Drop project questions that reference a project the candidate doesn't have."""
    if q.project_ref is None:
        return True
    return profile.project(q.project_ref) is not None


# --------------------------------------------------------------------------
# Offline template bank
# --------------------------------------------------------------------------

# Keyed by skill; each entry is (difficulty, question, expected_points, keywords).
SKILL_QUESTIONS: dict[str, list[tuple[Domain, int, str, list[str], list[str]]]] = {
    "postgresql": [
        (
            Domain.databases, 3,
            "When you add an index in Postgres, what are you trading away?",
            ["Slower writes — every insert/update maintains the index",
             "Disk space", "Planner may still ignore it", "Index only helps matching predicates"],
            ["write", "insert", "slower", "space", "disk", "storage", "b-tree", "btree"],
        ),
        (
            Domain.databases, 4,
            "A query that was fast at 10k rows crawls at 10M. Walk me through diagnosing it.",
            ["EXPLAIN ANALYZE to see the actual plan", "Look for seq scan vs index scan",
             "Check row estimates vs actual — stale statistics", "N+1 at the application layer"],
            ["explain", "analyze", "plan", "seq scan", "index", "vacuum", "statistics", "n+1"],
        ),
    ],
    "react": [
        (
            Domain.frontend, 3,
            "What actually causes a React component to re-render?",
            ["State change in the component", "Parent re-render", "Context value change",
             "Not prop mutation — identity change"],
            ["state", "setstate", "parent", "context", "props", "reconcil", "memo"],
        ),
    ],
    "django": [
        (
            Domain.backend, 3,
            "What is the N+1 query problem in the Django ORM, and how do you fix it?",
            ["Loop over objects triggers a query per related object",
             "select_related for FK/one-to-one (SQL join)",
             "prefetch_related for many-to-many/reverse (second query)"],
            ["n+1", "select_related", "prefetch_related", "join", "lazy", "queryset"],
        ),
    ],
    "docker": [
        (
            Domain.devops, 2,
            "What's the difference between a Docker image and a container?",
            ["Image is the immutable layered filesystem + metadata",
             "Container is a running instance with a writable layer",
             "Many containers from one image"],
            ["image", "container", "layer", "immutable", "instance", "writable"],
        ),
    ],
    "python": [
        (
            Domain.algorithms, 2,
            "When would you reach for a dict over a list, and what does that cost you?",
            ["O(1) average lookup vs O(n) scan", "Requires hashable keys",
             "Higher memory overhead", "Insertion order preserved but not sorted"],
            ["hash", "o(1)", "constant", "lookup", "memory", "hashable", "collision"],
        ),
    ],
    "kubernetes": [
        (
            Domain.devops, 4,
            "A pod is stuck in CrashLoopBackOff. How do you work it out?",
            ["kubectl logs --previous for the dead container", "kubectl describe for events",
             "Check liveness/readiness probe config", "Resource limits — OOMKilled"],
            ["logs", "describe", "previous", "probe", "liveness", "oom", "limits", "events"],
        ),
    ],
    "pytorch": [
        (
            Domain.ml, 3,
            "Your training loss drops but validation loss climbs. What's happening and what do you do?",
            ["Overfitting", "Regularisation: dropout, weight decay, augmentation",
             "Early stopping", "Check for train/val leakage or distribution mismatch"],
            ["overfit", "dropout", "regulari", "weight decay", "early stopping", "augment"],
        ),
    ],
}

GENERIC_BEHAVIORAL = [
    (
        Domain.behavioral, 1,
        "Tell me about the most technically demanding thing you've built. What made it hard?",
        ["Concrete project with a stated problem", "The specific technical difficulty",
         "What they personally did", "How it turned out"],
        ["built", "because", "problem", "decided", "implemented"],
    ),
    (
        Domain.behavioral, 2,
        "Describe a time you had to learn a technology quickly. How did you go about it?",
        ["Named technology and the forcing deadline", "Concrete learning approach",
         "What they shipped with it", "What they'd do differently"],
        ["learn", "documentation", "built", "tutorial", "first", "shipped"],
    ),
    (
        Domain.behavioral, 2,
        "Tell me about a technical decision you got wrong. What did you do about it?",
        ["A real decision, owned without deflection", "Why it was wrong",
         "The correction and its cost", "The generalised lesson"],
        ["wrong", "mistake", "instead", "refactor", "learned", "should have"],
    ),
]


def _bare(topic: str) -> str:
    """'your use of postgresql' -> 'postgresql', so restatements read naturally."""
    return topic.removeprefix("your use of ").strip()


def template_bank(profile: CandidateProfile) -> list[Question]:
    """Deterministic bank used when no model is available.

    Shallower than a generated bank, but grounded in the same profile: project
    questions name real projects and technical questions follow real skills.
    """
    questions: list[Question] = []

    for domain, diff, text, points, keywords in GENERIC_BEHAVIORAL:
        questions.append(
            Question(
                text=text, domain=domain, difficulty=diff,
                expected_points=points, keywords=keywords,
                model_answer="A specific project, the concrete difficulty, your own role in it, and the outcome.",
                followup_if_strong="What would you do differently if you started it again today?",
                followup_if_weak="Pick one project and tell me just what it does end to end.",
                simpler_variant="What's a project you're proud of? Just walk me through what it does.",
            )
        )

    for project in profile.projects[:4]:
        tech = ", ".join(project.tech[:4]) or "your stack"
        questions.append(
            Question(
                text=f"Walk me through {project.name}. What does it do, and how is it put together?",
                domain=Domain.project_deep_dive, difficulty=2,
                expected_points=["What the project does and for whom", f"The stack: {tech}",
                                 "How the pieces fit together", "The hardest part"],
                keywords=[t.lower() for t in project.tech] + ["built", "used", "because"],
                model_answer=project.summary,
                project_ref=project.name,
                followup_if_strong=f"What was the hardest tradeoff you made in {project.name}?",
                followup_if_weak=f"What problem does {project.name} solve for its user?",
                simpler_variant=f"In one sentence, what does {project.name} do?",
            )
        )
        for topic in project.probe_topics[:3]:
            questions.append(
                Question(
                    text=f"In {project.name}, tell me about {topic}. Why that way?",
                    domain=Domain.project_deep_dive, difficulty=4,
                    expected_points=["The decision as actually made", "The alternatives considered",
                                     "Why this one won", "What it cost"],
                    keywords=[t.lower() for t in project.tech] + ["because", "instead", "tradeoff"],
                    model_answer=f"A specific account of {topic} in {project.name}, with the alternatives weighed.",
                    project_ref=project.name,
                    followup_if_strong="Where would that choice break down at ten times the scale?",
                    followup_if_weak="What alternatives did you look at before settling on it?",
                    simpler_variant=f"In {project.name}, what does {_bare(topic)} do?",
                )
            )

    skills = {s.lower() for s in profile.skills}
    aliases = {"postgres": "postgresql", "node": "node.js", "sklearn": "scikit-learn"}
    for skill in list(skills):
        skills.add(aliases.get(skill, skill))
    for skill, entries in SKILL_QUESTIONS.items():
        if skill not in skills:
            continue
        for domain, diff, text, points, keywords in entries:
            questions.append(
                Question(
                    text=text, domain=domain, difficulty=diff,
                    expected_points=points, keywords=keywords,
                    model_answer=" ".join(points[:3]) + ".",
                    followup_if_strong="Where does that stop being true?",
                    followup_if_weak="Let's take the first part only — what does it do?",
                    simpler_variant=f"At a high level, what is {skill} used for in a project like yours?",
                )
            )

    return questions


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def _bank_path(key: str) -> Path:
    directory = Path(get_settings().data_dir) / "banks"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{key}.json"


def save_bank(bank: QuestionBank) -> None:
    _bank_path(bank.profile_key).write_text(bank.model_dump_json(indent=2))


def load_bank(key: str) -> QuestionBank | None:
    path = _bank_path(key)
    if not path.exists():
        return None
    try:
        return QuestionBank.model_validate_json(path.read_text())
    except (json.JSONDecodeError, ValueError) as e:
        log.warning("discarding corrupt bank %s: %s", path, e)
        return None
