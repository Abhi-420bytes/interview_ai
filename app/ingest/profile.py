"""Merge resume + GitHub into one profile, and flag what doesn't line up.

The cross-reference is the interesting part: a resume claim with no supporting
code is exactly the thing worth asking about, so discrepancies become
interview material rather than a rejection signal.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import get_settings
from app.ingest import github, resume
from app.llm import client as llm
from app.models import CandidateProfile, Discrepancy, Project, Seniority


class CrossReference(BaseModel):
    discrepancies: list[Discrepancy] = Field(
        description=(
            "Resume claims the GitHub evidence does not support, and notable "
            "GitHub work the resume omits. Be fair: absence of public code is "
            "weak evidence (work may be private), so severity is 'high' only "
            "when a claimed primary skill has no trace across any repo."
        )
    )
    strengths: list[str] = Field(description="Top three areas of demonstrated depth.")
    gaps: list[str] = Field(description="Top three areas to probe hardest.")


CROSSREF_SYSTEM = """You reconcile a candidate's resume against their public \
GitHub activity to plan a technical interview.

Judge the evidence, not the candidate. Private and professional work is \
invisible on GitHub, so a missing repo is not proof of a false claim — say what \
the evidence does and does not show. Where a claim is unsupported, phrase the \
discrepancy as something an interviewer can *ask about*, not an accusation."""


async def build_profile(
    resume_path: str | Path | None = None,
    resume_text: str | None = None,
    github_login: str | None = None,
    repo_limit: int = 5,
) -> CandidateProfile:
    """Full pre-interview indexing pass. Resume and GitHub are fetched in parallel."""
    if resume_text is None and resume_path is not None:
        resume_text = resume.read_resume(resume_path)

    resume_task = (
        resume.parse_resume(resume_text)
        if resume_text
        else _empty_profile()
    )
    github_task = (
        github.analyse_user(github_login, limit=repo_limit)
        if github_login
        else _no_projects()
    )
    profile, repos = await asyncio.gather(resume_task, github_task)

    profile.github_login = github_login
    profile.projects = _merge_projects(profile.projects, repos)
    profile.skills = _merge_skills(profile.skills, repos)
    profile = _apply_recency_signal(profile, repos)

    if repos and resume_text:
        xref = await llm.structured(
            schema=CrossReference,
            system=CROSSREF_SYSTEM,
            user=(
                f"<resume>\n{resume_text.strip()[:20000]}\n</resume>\n\n"
                f"<github>\n{_render_repos(repos)}\n</github>"
            ),
            model=get_settings().deep_model,
            max_tokens=get_settings().precompute_max_tokens,
            effort="high",
        )
        if xref:
            profile.discrepancies = xref.discrepancies
            profile.strengths = xref.strengths or profile.strengths
            profile.gaps = xref.gaps or profile.gaps

    return profile


async def _empty_profile() -> CandidateProfile:
    return CandidateProfile()


async def _no_projects() -> list[Project]:
    return []


def _merge_projects(resume_projects: list[Project], repos: list[Project]) -> list[Project]:
    """GitHub wins on tech detail; resume wins on framing. Same name -> one project."""
    by_key: dict[str, Project] = {}
    for p in resume_projects:
        by_key[_key(p.name)] = p
    for repo in repos:
        key = _key(repo.name)
        if key in by_key:
            existing = by_key[key]
            repo.source = "both"
            repo.summary = existing.summary or repo.summary
            repo.tech = sorted(set(existing.tech) | set(repo.tech))
            by_key[key] = repo
        else:
            by_key[key] = repo
    return sorted(by_key.values(), key=lambda p: (p.source != "both", -p.stars, p.name))


def _key(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


def _merge_skills(skills: list[str], repos: list[Project]) -> list[str]:
    merged = set(skills)
    for repo in repos:
        merged.update(repo.tech)
    return sorted(merged)


def _apply_recency_signal(profile: CandidateProfile, repos: list[Project]) -> CandidateProfile:
    """A student with several substantial, collaborative repos reads as mid, not junior."""
    if profile.seniority is Seniority.junior:
        substantial = [r for r in repos if r.collaborators > 1 and r.stars >= 3]
        if len(substantial) >= 2 and profile.years_experience >= 1:
            profile.seniority = Seniority.mid
    return profile


def _render_repos(repos: list[Project]) -> str:
    return "\n\n".join(
        f"{r.name} ({r.url})\n  tech: {', '.join(r.tech)}\n  {r.summary}\n"
        f"  stars: {r.stars} | contributors: {r.collaborators} | last push: {r.last_pushed}"
        for r in repos
    )


def profile_key(profile: CandidateProfile) -> str:
    """Stable cache key for a profile's question bank."""
    material = "|".join(
        [profile.name, profile.role.value, profile.seniority.value]
        + sorted(profile.skills)
        + sorted(p.name for p in profile.projects)
    )
    return hashlib.sha256(material.encode()).hexdigest()[:16]
