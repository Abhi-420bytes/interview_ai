"""Resume parsing and profile merge, offline."""

import pytest

from app.ingest import resume as resume_mod
from app.ingest.profile import _merge_projects, build_profile, profile_key
from app.models import Project, Role

RESUME = """Abhiram Challa
Final-year B.Tech Computer Science

Skills
Python, JavaScript, Django, React, PostgreSQL, Docker

Projects
Sri Vari Water Solutions | Next.js, PostgreSQL, Socket.io
- Real-time delivery tracking
Legal Document Analysis | Python, FastAPI, NLP
- Contract clause extraction

Experience
Software Engineering Intern, 2 years
"""



async def test_heuristic_parse_finds_skills_and_projects():
    profile = await resume_mod.parse_resume(RESUME)
    assert "django" in profile.skills
    assert "postgresql" in profile.skills
    assert profile.years_experience == 2.0
    assert any("Sri Vari" in p.name for p in profile.projects)


async def test_role_inferred_from_stack():
    profile = await resume_mod.parse_resume(RESUME)
    assert profile.role is Role.fullstack


async def test_skill_matching_does_not_fire_on_substrings():
    profile = await resume_mod.parse_resume("Skills: javascript, gopher, carbon")
    assert "go" not in profile.skills
    assert "r" not in profile.skills
    assert "javascript" in profile.skills


async def test_build_profile_works_from_resume_alone():
    profile = await build_profile(resume_text=RESUME)
    assert profile.name != ""
    assert profile.skills


def test_projects_from_both_sources_merge_into_one():
    """Names are normalised, so a repo slug matches its resume spelling."""
    merged = _merge_projects(
        [Project(name="Sri Vari", summary="Water delivery.", tech=["next.js"])],
        [Project(name="sri-vari", summary="repo description", tech=["typescript"], source="github")],
    )
    assert len(merged) == 1
    assert merged[0].source == "both"
    assert set(merged[0].tech) == {"next.js", "typescript"}
    assert merged[0].summary == "Water delivery.", "resume framing wins over repo description"


def test_unrelated_projects_stay_separate():
    merged = _merge_projects(
        [Project(name="Sri Vari", summary="Water delivery.", tech=["next.js"])],
        [Project(name="Legal Docs", summary="NLP pipeline", tech=["python"], source="github")],
    )
    assert len(merged) == 2


def test_profile_key_is_stable_and_order_independent(profile):
    first = profile_key(profile)
    profile.skills = list(reversed(profile.skills))
    assert profile_key(profile) == first


async def test_markdown_resume_sections_are_found():
    """Heading markers must not hide the PROJECTS section."""
    md = """# Abhiram Challa

## Skills
Python, Django, PostgreSQL

## Projects

Sri Vari Water Solutions — Water delivery platform
Built with Next.js and PostgreSQL, real-time tracking over Socket.io.

Legal Document Analysis — Contract review pipeline
Clause extraction with transformer models behind a FastAPI service.

## Experience
Intern, 1 year
"""
    profile = await resume_mod.parse_resume(md)
    assert profile.name == "Abhiram Challa"
    names = [p.name for p in profile.projects]
    assert "Sri Vari Water Solutions" in names
    assert "Legal Document Analysis" in names


async def test_project_probe_topics_do_not_repeat_the_project_name():
    profile = await resume_mod.parse_resume(RESUME)
    for project in profile.projects:
        for topic in project.probe_topics:
            assert project.name.lower() not in topic.lower()
