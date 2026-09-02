#!/usr/bin/env python
"""Run a full interview in the terminal.

    python scripts/interview.py --resume data/sample_resume.md --github octocat
    python scripts/interview.py --resume data/sample_resume.md --mode audio

Works with no API key — the engine falls back to its heuristic scorer and
template question bank, and says so.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.ingest import resume as resume_mod  # noqa: E402
from app.ingest.profile import build_profile  # noqa: E402
from app.interview import bank as bank_mod, engine, report as report_mod  # noqa: E402
from app.models import Mode, Verdict  # noqa: E402

BADGE = {
    Verdict.excellent: "\033[92m✅ EXCELLENT\033[0m",
    Verdict.good: "\033[92m✅ GOOD\033[0m",
    Verdict.partial: "\033[93m⚠️  PARTIAL\033[0m",
    Verdict.wrong: "\033[91m❌ WRONG\033[0m",
    Verdict.off_topic: "\033[91m❌ OFF-TOPIC\033[0m",
}
DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run a mock interview in the terminal.")
    parser.add_argument("--resume", type=Path, help="Path to a PDF, .md, or .txt resume")
    parser.add_argument("--github", help="GitHub username to analyse")
    parser.add_argument("--mode", choices=["fast", "audio"], default="fast")
    parser.add_argument("--questions", type=int, default=None, help="Session length")
    args = parser.parse_args()

    if not args.resume and not args.github:
        parser.error("give --resume, --github, or both")

    settings = get_settings()
    if args.questions:
        settings.session_length = args.questions

    if settings.offline:
        print(
            f"{DIM}No Anthropic credentials found — running in offline mode: "
            f"template question bank, heuristic scoring.{RESET}\n"
        )
    else:
        print(f"{DIM}Models: {settings.fast_model} (online) / {settings.deep_model} (offline){RESET}\n")

    print("Indexing candidate…")
    t0 = time.perf_counter()
    profile = await build_profile(
        resume_text=resume_mod.read_resume(args.resume) if args.resume else None,
        github_login=args.github,
    )
    question_bank = await bank_mod.build_bank(profile)
    print(
        f"{DIM}Indexed in {time.perf_counter() - t0:.1f}s — "
        f"{profile.name}, {profile.role.value}/{profile.seniority.value}, "
        f"{len(profile.projects)} projects, {len(question_bank.questions)} questions ready.{RESET}\n"
    )
    if profile.discrepancies:
        print(f"{DIM}Flagged for probing:{RESET}")
        for d in profile.discrepancies[:3]:
            print(f"{DIM}  • [{d.severity}] {d.claim}{RESET}")
        print()

    session, question = await engine.start_session(
        profile, mode=Mode(args.mode), question_bank=question_bank
    )
    print("─" * 72)

    n = 1
    while True:
        print(f"\n{BOLD}Q{n}{RESET} {DIM}[{question.domain.value} · difficulty {question.difficulty}]{RESET}")
        print(f"{question.text}\n")
        try:
            answer = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nEnding early.")
            break
        if not answer:
            print(f"{DIM}(empty — skipping){RESET}")
            answer = "(no answer)"
        if answer.lower() in {"quit", "exit"}:
            break

        result = await engine.submit_answer(session, answer)
        ev = result.evaluation

        print(f"\n  {BADGE[ev.verdict]} {BOLD}{ev.score}/100{RESET} "
              f"{DIM}({result.elapsed_ms}ms{', heuristic only' if ev.heuristic_only else ''}){RESET}")
        print(f"  {ev.feedback}")
        if ev.bonuses:
            print(f"  {DIM}+ {'; '.join(ev.bonuses)}{RESET}")
        if ev.penalties:
            print(f"  {DIM}− {'; '.join(ev.penalties)}{RESET}")
        if ev.missed_points:
            print(f"  {DIM}Missed: {'; '.join(ev.missed_points[:3])}{RESET}")
        if result.model_answer:
            print(f"  {DIM}Model answer: {result.model_answer}{RESET}")
        if result.speech:
            print(f"  {DIM}[speak @ {result.speech.wpm}wpm, {result.speech.tone}: "
                  f"\"{result.speech.lead_in}\"]{RESET}")

        if result.done or result.next_question is None:
            break
        question = result.next_question
        n += 1

    print("\n" + "─" * 72)
    print(f"\n{BOLD}Report{RESET}\n")
    report = await report_mod.build_report(session)
    print(f"  Overall: {BOLD}{report.overall}/100{RESET}  ({report.duration_s:.0f}s)")
    for d in sorted(report.by_domain, key=lambda x: -x.mean_score):
        bar = "█" * round(d.mean_score / 5)
        print(f"  {d.domain.value:<20} {d.mean_score:>5.1f}  {DIM}{bar}{RESET} ({d.questions}q)")
    print(f"\n  {BOLD}Strengths{RESET}")
    for s in report.strengths:
        print(f"    ✓ {s}")
    print(f"\n  {BOLD}Growth areas{RESET}")
    for g in report.growth_areas:
        print(f"    → {g}")
    print(f"\n  {BOLD}Next steps{RESET}")
    for s in report.next_steps:
        print(f"    • {s}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
