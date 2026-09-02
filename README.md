# AI Interview Coach — backend & interview engine

Adaptive mock technical interviews grounded in a candidate's actual resume and
GitHub repositories. This is the engine: profiling, question generation,
answer scoring, session adaptation, and reporting, behind a FastAPI service and
a terminal driver. No UI yet.

## Quick start

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt

# Runs with no API key (heuristic mode — see below)
./.venv/bin/python scripts/interview.py --resume data/sample_resume.md

# With credentials, the full model-backed path
export ANTHROPIC_API_KEY=sk-ant-...
./.venv/bin/python scripts/interview.py --resume data/sample_resume.md --github <username>

# As a service
./.venv/bin/uvicorn app.main:app --reload   # docs at /docs
./.venv/bin/python -m pytest                # 40 tests, all offline
```

## The shape of the thing

The design problem is that a realistic interview needs good judgement about an
answer, and a 2–3 second turnaround, and those pull against each other. The
resolution is to move everything that can be pre-computed off the online path,
and to never let the online path block on the model.

**Offline, once per candidate** (`app/ingest/`, `app/interview/bank.py`) — the
deep model reads the resume and each GitHub repo, cross-references the two, and
writes a question bank of ~50 questions. Each question carries its own expected
points, scoring keywords, model answer, *both* follow-up branches, and a simpler
restatement. This is the decision tree: at interview time the engine selects
from it rather than generating.

**Online, per answer** (`app/interview/engine.py`):

```
heuristic score   ── instant, always available, pure Python
llm score         ─┐ raced under a hard 1.6s budget
candidate slate   ─┘ next-question options computed concurrently
```

If the model lands in time, its verdict and its context-aware follow-up carry
the turn. If it doesn't, the heuristic verdict and the *pre-generated* branch do
— no second round trip, no stall. The result is flagged `heuristic_only` so the
client can be honest about which one it got.

| Path | Typical latency |
|---|---|
| Heuristic only (offline mode) | < 5 ms |
| Model-backed, budget met | ~0.8–1.6 s |
| Model-backed, budget missed | budget + ~10 ms, heuristic verdict |

## Models

| Path | Model | Why |
|---|---|---|
| Answer scoring, follow-ups | `claude-haiku-4-5` | On the latency-critical path; capped at 700 output tokens |
| Resume, GitHub, bank, report | `claude-opus-5` | Runs before or after the interview, where seconds don't matter |

The candidate profile is the stable prefix of every scoring call and is cached
(`cache_control: ephemeral`), so a whole session's evaluations re-read it at
~10% cost. Volatile content — this question, this answer — goes in the user
turn, after the cache breakpoint. `GET /health` reports cumulative token usage.

## Offline mode

With no credentials the engine runs end to end on heuristics: regex resume
parsing, a template question bank built from the candidate's real skills and
projects, and keyword/structure scoring. It is genuinely useful for development
and it is what the test suite runs against — but it is a floor, not a
substitute:

- **Technical scoring** is keyword overlap against the question's expected
  points. It rewards vocabulary, and it cannot tell a correct explanation from a
  fluent wrong one.
- **Behavioral scoring** doesn't even try keyword matching — a story shares
  almost no vocabulary with "names a specific project". It scores four
  structural signals instead (named a project, said what *you* did, gave
  reasoning, gave an outcome) and clamps itself to 35–88, because structure is
  not correctness and the heuristic shouldn't pretend to a verdict it can't
  justify.
- **Template questions** are shallow by construction (`In <project>, tell me
  about your use of <tech>. Why that way?`). The model-generated bank is where
  questions that name a real decision come from.

## Adaptation

Difficulty is a float on the session, moved per verdict (+0.8 excellent, −0.7
wrong). Selection then picks the nearest unasked question, with three overrides:
a domain averaging below 50 over two or more answers gets pulled back into
rotation; the last two domains asked are avoided so the interview doesn't tunnel;
and a wrong answer is met with the question's simpler restatement — graded
against fewer expected points, since it asks for less — before moving on. One
follow-up per root question, so a weak answer gets help without the interview
getting stuck on it.

Audio mode returns a `SpeechDirective` per turn (words-per-minute, tone,
lead-in) that slows and softens as the candidate struggles — 150 wpm and
"That's exactly it" for excellent, 95 wpm and "No worries, here's the idea" for
wrong.

## Scoring

Per the spec: keywords 40%, technical accuracy 35%, completeness 15%,
communication 10%. Then +10 for grounding an answer in their own project, +5 for
a real tradeoff, **+15 for naming the limit of what they know**, −5 for
uncommitted hedging, −10 for confident wrongness, −20 for recitation. The bonus
structure is deliberate: it pays more to say "I don't know how the planner
handles that" than to bluff past it.

## Layout

```
app/
  config.py            settings, model split, latency budgets
  models.py            domain model; doubles as the extraction schemas
  store.py             session storage (in-memory + JSON write-through)
  api.py               HTTP surface
  llm/client.py        SDK wrapper: budgets, caching, offline short-circuit
  ingest/
    resume.py          PDF/text -> structured facts
    github.py          public repos -> per-project probe topics
    profile.py         merge + cross-reference the two
  interview/
    bank.py            question-bank generation and caching
    scoring.py         heuristic + model scoring
    engine.py          session state machine and adaptation
    report.py          post-interview report
scripts/interview.py   terminal driver
```

## API

| Endpoint | Purpose |
|---|---|
| `POST /profile` | Build a profile from resume text and/or a GitHub login |
| `POST /profile/upload` | Same, from an uploaded PDF or text file |
| `POST /bank` | Generate (or fetch cached) question bank for a profile |
| `POST /sessions` | Start an interview, returns the first question |
| `POST /sessions/{id}/answer` | Score an answer, return the next question |
| `GET /sessions/{id}` | Live session state and per-domain scores |
| `GET /sessions/{id}/report` | Final report |
| `GET /health` | Mode, models, budgets, cumulative token usage |

## Cross-referencing

The resume and the GitHub evidence are reconciled, and mismatches become
interview material rather than a rejection signal — a claimed primary skill with
no trace in any repo is exactly the thing worth asking about. The prompt is
explicit that private and professional work is invisible on GitHub, so a missing
repo is weak evidence, and discrepancies are phrased as something to ask about.

## Not built yet

- **Mobile client.** React Native shell, TTS/STT, Bluetooth audio routing. The
  audio mode's server half exists (`SpeechDirective` per turn); nothing speaks.
- **Streaming.** Responses are whole; streaming the first sentence while the
  rest generates would take a further ~300 ms off perceived latency.
- **Multi-worker storage.** `SessionStore` is per-process. Swap for Redis.
- **Local on-device model.** The spec's ONNX path. Offline mode is heuristic,
  not a small LLM.
- **Score calibration.** Thresholds (85/70/45) are reasoned, not validated
  against real interview outcomes.
