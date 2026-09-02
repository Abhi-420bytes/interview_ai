"""Runtime configuration for the interview engine.

Model split (see README): a small, fast model carries the latency-critical
online path, a large model does offline pre-computation where seconds don't
matter. `offline` mode drops the LLM entirely and runs on heuristics only,
which is what makes the engine testable without an API key.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="INTERVIEW_", extra="ignore"
    )

    # --- Models -----------------------------------------------------------
    # Online path: every millisecond is visible to the candidate.
    fast_model: str = "claude-haiku-4-5"
    # Offline path: resume/GitHub analysis, question-bank generation.
    deep_model: str = "claude-opus-5"

    # --- Latency budgets (seconds) ---------------------------------------
    # The engine returns the heuristic verdict if the LLM misses these.
    eval_budget_s: float = 1.6
    followup_budget_s: float = 1.2

    # --- Token caps -------------------------------------------------------
    eval_max_tokens: int = 700
    followup_max_tokens: int = 300
    precompute_max_tokens: int = 16000

    # --- Behaviour --------------------------------------------------------
    # No API key / no network: heuristics only. Auto-detected in get_settings.
    offline: bool = False
    github_token: str | None = None
    data_dir: str = "data"

    # Number of questions in a full session.
    session_length: int = 12
    warmup_questions: int = 3


_settings: Settings | None = None


def _has_credentials() -> bool:
    """Mirror the SDK's credential resolution closely enough to decide offline mode.

    The SDK falls back to an `ant auth login` profile under ~/.config/anthropic
    when no env var is set, so the presence of that directory counts.
    """
    import os
    from pathlib import Path

    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")).joinpath(
        "anthropic"
    ).is_dir()


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        import os

        s = Settings()
        if not s.offline and not _has_credentials():
            s.offline = True
        _settings = s
    return _settings


def reset_settings() -> None:
    """Test hook."""
    global _settings
    _settings = None
