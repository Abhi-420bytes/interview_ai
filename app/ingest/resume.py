"""Resume ingestion: raw file -> structured candidate facts.

Runs offline (pre-interview), so it uses the deep model at high effort. The
heuristic path is a genuine fallback, not a stub: it produces a usable — if
shallower — profile with no API key.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import get_settings
from app.llm import client as llm
from app.models import CandidateProfile, Project, Role, Seniority

# Skills we can recognise without a model. Ordered longest-first at match time
# so "spring boot" wins over "spring".
KNOWN_SKILLS = [
    "python", "javascript", "typescript", "java", "kotlin", "swift", "go", "golang",
    "rust", "c++", "c#", ".net", "ruby", "php", "scala", "r", "matlab", "sql",
    "django", "flask", "fastapi", "spring boot", "spring", "express", "node.js",
    "node", "react native", "react", "next.js", "vue", "angular", "svelte",
    "flutter", "dart", "tailwind", "redux",
    "postgresql", "postgres", "mysql", "mongodb", "redis", "sqlite", "dynamodb",
    "cassandra", "elasticsearch", "neo4j", "snowflake", "bigquery",
    "docker", "kubernetes", "terraform", "ansible", "jenkins", "github actions",
    "aws", "gcp", "azure", "lambda", "s3", "ec2", "cloudflare",
    "kafka", "rabbitmq", "celery", "socket.io", "websockets", "graphql", "grpc",
    "rest", "microservices", "ci/cd",
    "pytorch", "tensorflow", "scikit-learn", "sklearn", "pandas", "numpy",
    "huggingface", "langchain", "nlp", "opencv", "spark", "airflow", "dbt",
    "llm", "rag", "transformers",
]

_DEGREE_RE = re.compile(
    r"\b(b\.?tech|b\.?e\.?|b\.?sc|m\.?tech|m\.?sc|mba|ph\.?d|bachelor|master)\b[^\n]{0,80}",
    re.I,
)
_YEARS_RE = re.compile(r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\b", re.I)


class ResumeExtract(BaseModel):
    """Schema handed to the model for resume parsing."""

    name: str = Field(description="Candidate's full name, or 'Candidate' if absent.")
    role: Role = Field(description="Best-fit target role given the whole resume.")
    seniority: Seniority
    years_experience: float = Field(
        description="Professional years, internships at 0.5 weight. 0 for students."
    )
    education: list[str] = Field(description="Degree, institution, year.")
    certifications: list[str]
    skills: list[str] = Field(description="Lowercase technologies actually evidenced.")
    projects: list[Project] = Field(
        description="Every project or significant work item, with its tech stack."
    )
    strengths: list[str] = Field(description="Three areas of genuine depth.")
    gaps: list[str] = Field(
        description="Areas thin or absent given the target role — probe these."
    )


def read_resume(path: str | Path) -> str:
    """Extract raw text from a PDF, or read a text/markdown resume as-is."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="replace")


SYSTEM = """You parse resumes into structured candidate profiles for a technical \
interview system.

Rules:
- Record only what the resume evidences. Never invent a project, skill, or date.
- `skills` is what the candidate has *used*, not what they list aspirationally; \
if a technology appears only in an "interests" or "familiar with" section, it \
belongs in `gaps`, not `skills`.
- For each project, `probe_topics` are the specific technical decisions an \
interviewer could fairly ask about given the stack named (e.g. a Postgres \
project -> "index choice", "transaction isolation"). Three to five per project.
- `gaps` drive the interview's hard questions. Be specific: "no evidence of \
testing or CI" beats "needs improvement"."""


async def parse_resume(text: str) -> CandidateProfile:
    """Structured profile from resume text, model-backed when credentials allow."""
    settings = get_settings()
    extract = await llm.structured(
        schema=ResumeExtract,
        system=SYSTEM,
        user=f"<resume>\n{text.strip()}\n</resume>",
        model=settings.deep_model,
        max_tokens=settings.precompute_max_tokens,
        effort="high",
    )
    if extract is None:
        return _heuristic_profile(text)

    return CandidateProfile(
        name=extract.name,
        role=extract.role,
        seniority=extract.seniority,
        years_experience=extract.years_experience,
        education=extract.education,
        certifications=extract.certifications,
        skills=[s.lower() for s in extract.skills],
        projects=extract.projects,
        strengths=extract.strengths,
        gaps=extract.gaps,
    )


def _heuristic_profile(text: str) -> CandidateProfile:
    lowered = text.lower()
    skills = [s for s in KNOWN_SKILLS if _mentions(lowered, s)]

    years = 0.0
    if m := _YEARS_RE.search(text):
        years = float(m.group(1))
    seniority = (
        Seniority.senior if years >= 6 else Seniority.mid if years >= 2.5 else Seniority.junior
    )

    name = "Candidate"
    for line in _strip_markup(text).splitlines():
        line = line.strip()
        if 2 <= len(line.split()) <= 4 and line.replace(" ", "").replace(".", "").isalpha():
            name = line.title()
            break

    return CandidateProfile(
        name=name,
        role=_infer_role(skills),
        seniority=seniority,
        years_experience=years,
        education=[m.group(0).strip() for m in _DEGREE_RE.finditer(text)][:3],
        skills=skills,
        projects=_heuristic_projects(text, skills),
        strengths=skills[:3],
        gaps=[],
    )


def _mentions(haystack: str, skill: str) -> bool:
    return re.search(rf"(?<![\w.+#]){re.escape(skill)}(?![\w+#])", haystack) is not None


def _infer_role(skills: list[str]) -> Role:
    sets = {
        Role.ml: {"pytorch", "tensorflow", "huggingface", "transformers", "nlp", "llm", "rag"},
        Role.data_science: {"pandas", "numpy", "scikit-learn", "sklearn", "spark", "r", "dbt"},
        Role.devops: {"kubernetes", "terraform", "ansible", "jenkins", "ci/cd", "docker"},
        Role.fullstack: {"react", "vue", "angular", "next.js", "django", "express", "flask"},
    }
    best, best_hits = Role.swe, 0
    for role, markers in sets.items():
        hits = len(markers & set(skills))
        if hits > best_hits:
            best, best_hits = role, hits
    return best


def _heuristic_projects(text: str, skills: list[str]) -> list[Project]:
    """Pull projects out of a PROJECTS section, block by block.

    A block is a title line plus the prose under it, up to a blank line, which
    is how nearly every resume lays this out regardless of markup.
    """
    lines = _strip_markup(text).splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^(projects?|personal projects?|selected projects?)\s*:?\s*$", line, re.I):
            start = i + 1
            break
    if start is None:
        return []

    end = len(lines)
    for i in range(start, len(lines)):
        if re.match(r"^(experience|education|skills?|certifications?|awards?)\b", lines[i], re.I):
            end = i
            break

    projects: list[Project] = []
    block: list[str] = []
    for line in lines[start:end] + [""]:
        if line.strip():
            block.append(line.strip())
            continue
        if block:
            if project := _project_from_block(block, skills):
                projects.append(project)
            block = []
    return projects[:6]


def _project_from_block(block: list[str], skills: list[str]) -> Project | None:
    title_line = block[0]
    # "Name — description", "Name | stack", "Name: description" all split the same way.
    name = re.split(r"\s[|—–:]\s|\s-\s", title_line)[0].strip(" -*•")
    if not name or len(name) > 80:
        return None
    body = " ".join(block).lower()
    tech = [s for s in skills if _mentions(body, s)]
    summary = " ".join(block[1:]).strip() or title_line
    return Project(
        name=name,
        summary=summary,
        tech=tech,
        probe_topics=[f"your use of {t}" for t in tech[:3]],
    )


def _strip_markup(text: str) -> str:
    """Drop markdown heading/bullet markers so section detection is format-agnostic."""
    out = []
    for line in text.splitlines():
        out.append(re.sub(r"^\s*(#{1,6}\s*|[-*•]\s+)", "", line).rstrip())
    return "\n".join(out)
