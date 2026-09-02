"""GitHub ingestion: public repos -> per-project technical signal.

Fetches with the public REST API (add a token to lift the 60 req/hr limit),
then optionally has the deep model read each repo's README and languages to
produce the `probe_topics` the interview actually hangs its questions on.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from datetime import datetime, timezone

import httpx
from pydantic import BaseModel, Field

from app.config import get_settings
from app.llm import client as llm
from app.models import Project

log = logging.getLogger(__name__)

API = "https://api.github.com"


class RepoSnapshot(BaseModel):
    """Raw facts pulled from the API, before any model interpretation."""

    name: str
    description: str = ""
    url: str
    languages: dict[str, int] = Field(default_factory=dict)
    stars: int = 0
    forks: int = 0
    is_fork: bool = False
    pushed_at: str = ""
    created_at: str = ""
    size_kb: int = 0
    topics: list[str] = Field(default_factory=list)
    contributors: int = 1
    readme: str = ""

    @property
    def months_stale(self) -> float:
        if not self.pushed_at:
            return 999.0
        pushed = datetime.fromisoformat(self.pushed_at.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - pushed).days / 30.44


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "interview-ai"}
    if token := get_settings().github_token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def fetch_repos(login: str, limit: int = 5) -> list[RepoSnapshot]:
    """Top `limit` non-fork repos by (stars, recency), with languages and README."""
    async with httpx.AsyncClient(timeout=20.0, headers=_headers()) as http:
        resp = await http.get(
            f"{API}/users/{login}/repos", params={"per_page": 100, "sort": "pushed"}
        )
        resp.raise_for_status()
        raw = [r for r in resp.json() if not r["fork"]]
        raw.sort(key=lambda r: (r["stargazers_count"], r["pushed_at"]), reverse=True)
        raw = raw[:limit]

        snapshots = await asyncio.gather(
            *(_hydrate(http, login, r) for r in raw), return_exceptions=True
        )

    out: list[RepoSnapshot] = []
    for s in snapshots:
        if isinstance(s, Exception):
            log.warning("repo hydration failed: %s", s)
        else:
            out.append(s)
    return out


async def _hydrate(http: httpx.AsyncClient, login: str, repo: dict) -> RepoSnapshot:
    name = repo["name"]
    languages, readme, contributors = await asyncio.gather(
        _get_json(http, f"{API}/repos/{login}/{name}/languages", default={}),
        _get_readme(http, login, name),
        _get_json(http, f"{API}/repos/{login}/{name}/contributors", default=[]),
    )
    return RepoSnapshot(
        name=name,
        description=repo.get("description") or "",
        url=repo["html_url"],
        languages=languages,
        stars=repo["stargazers_count"],
        forks=repo["forks_count"],
        is_fork=repo["fork"],
        pushed_at=repo["pushed_at"],
        created_at=repo["created_at"],
        size_kb=repo["size"],
        topics=repo.get("topics", []),
        contributors=max(1, len(contributors) if isinstance(contributors, list) else 1),
        readme=readme[:8000],
    )


async def _get_json(http: httpx.AsyncClient, url: str, default):
    try:
        r = await http.get(url)
        return r.json() if r.status_code == 200 else default
    except httpx.HTTPError:
        return default


async def _get_readme(http: httpx.AsyncClient, login: str, repo: str) -> str:
    data = await _get_json(http, f"{API}/repos/{login}/{repo}/readme", default=None)
    if not data or "content" not in data:
        return ""
    try:
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    except (ValueError, TypeError):
        return ""


class RepoAnalysis(BaseModel):
    summary: str = Field(description="Two sentences: what it does and how it's built.")
    tech: list[str] = Field(description="Concrete technologies, lowercase.")
    probe_topics: list[str] = Field(
        description=(
            "3-5 specific technical decisions visible in this repo that an "
            "interviewer could fairly interrogate. Name the decision, not the topic: "
            "'chose Socket.io over SSE for delivery tracking', not 'real-time'."
        )
    )
    complexity: int = Field(ge=1, le=5, description="1 tutorial-grade, 5 production-grade.")


ANALYSIS_SYSTEM = """You analyse a GitHub repository to prepare technical \
interview questions about it.

You are given the repo metadata, language byte counts, and README. Ground every \
statement in that evidence. If the README is empty or boilerplate, say so in the \
summary and keep `probe_topics` to what the languages and structure actually \
support — do not invent architecture that isn't evidenced.

`complexity` reflects engineering substance: a single-file script is 1; a \
multi-service app with tests, CI, and a real datastore is 5. A tutorial \
follow-along is 1-2 regardless of line count."""


async def analyse_repo(snap: RepoSnapshot) -> Project:
    settings = get_settings()
    evidence = (
        f"repo: {snap.name}\n"
        f"description: {snap.description}\n"
        f"languages (bytes): {snap.languages}\n"
        f"topics: {snap.topics}\n"
        f"stars: {snap.stars} | contributors: {snap.contributors} | "
        f"size: {snap.size_kb}KB | last push: {snap.pushed_at}\n"
        f"README:\n{snap.readme or '(none)'}"
    )
    analysis = await llm.structured(
        schema=RepoAnalysis,
        system=ANALYSIS_SYSTEM,
        user=evidence,
        model=settings.deep_model,
        max_tokens=settings.precompute_max_tokens,
        effort="medium",
    )
    if analysis is None:
        return _heuristic_project(snap)

    return Project(
        name=snap.name,
        source="github",
        summary=analysis.summary,
        tech=[t.lower() for t in analysis.tech],
        url=snap.url,
        stars=snap.stars,
        last_pushed=snap.pushed_at,
        is_fork=snap.is_fork,
        collaborators=snap.contributors,
        probe_topics=analysis.probe_topics,
    )


def _heuristic_project(snap: RepoSnapshot) -> Project:
    langs = sorted(snap.languages, key=snap.languages.get, reverse=True)[:4]
    tech = [l.lower() for l in langs] + [t.lower() for t in snap.topics]
    return Project(
        name=snap.name,
        source="github",
        summary=snap.description or f"A {langs[0] if langs else 'code'} repository.",
        tech=sorted(set(tech)),
        url=snap.url,
        stars=snap.stars,
        last_pushed=snap.pushed_at,
        is_fork=snap.is_fork,
        collaborators=snap.contributors,
        probe_topics=[f"your use of {l}" for l in langs[:3]],
    )


async def analyse_user(login: str, limit: int = 5) -> list[Project]:
    snapshots = await fetch_repos(login, limit=limit)
    return list(await asyncio.gather(*(analyse_repo(s) for s in snapshots)))
