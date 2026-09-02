"""Thin async wrapper around the Anthropic SDK.

Two things live here that the rest of the app should not have to think about:

1. **Latency budgets.** The online path must answer in a couple of seconds.
   `structured(..., budget_s=...)` returns `None` rather than blocking past its
   budget, and callers fall back to their heuristic result.
2. **Offline mode.** With no credentials the wrapper short-circuits to `None`
   so the engine still runs end to end on heuristics alone.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from app.config import get_settings

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_client: anthropic.AsyncAnthropic | None = None


def get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        # Zero-arg constructor: resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
        # or an `ant auth login` profile.
        _client = anthropic.AsyncAnthropic(max_retries=1)
    return _client


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


_usage_log: list[Usage] = []


def usage_totals() -> Usage:
    total = Usage()
    for u in _usage_log:
        total.input_tokens += u.input_tokens
        total.output_tokens += u.output_tokens
        total.cache_read_input_tokens += u.cache_read_input_tokens
        total.cache_creation_input_tokens += u.cache_creation_input_tokens
    return total


def _record(response) -> None:
    u = getattr(response, "usage", None)
    if u is None:
        return
    _usage_log.append(
        Usage(
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )
    )


def _system_blocks(system: str, cache: bool) -> list[dict]:
    """A single cached system block.

    Caching is a prefix match, so the *stable* half of the prompt (task
    instructions + candidate profile) belongs here and everything volatile
    (this question, this answer) goes in the user turn.
    """
    block: dict = {"type": "text", "text": system}
    if cache:
        block["cache_control"] = {"type": "ephemeral"}
    return [block]


async def structured(
    *,
    schema: type[T],
    system: str,
    user: str,
    model: str,
    max_tokens: int,
    budget_s: float | None = None,
    effort: str | None = None,
    cache_system: bool = True,
) -> T | None:
    """Get a validated `schema` instance back, or `None` if it can't be had in time.

    Returning `None` instead of raising is deliberate: every caller on the
    online path has a heuristic fallback, and a slow model must degrade the
    answer quality, never the interview.
    """
    if get_settings().offline:
        return None

    async def _call() -> T | None:
        client = get_client()
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "system": _system_blocks(system, cache_system),
            "messages": [{"role": "user", "content": user}],
        }
        if effort:
            kwargs["output_config"] = {"effort": effort}
        try:
            response = await client.messages.parse(output_format=schema, **kwargs)
            _record(response)
            return response.parsed_output
        except (anthropic.BadRequestError, AttributeError, TypeError) as e:
            # Structured outputs unsupported for this model/SDK combination:
            # fall back to asking for raw JSON and validating it ourselves.
            log.debug("structured output unavailable (%s); using JSON fallback", e)
            return await _json_fallback(client, schema, kwargs)

    try:
        if budget_s is None:
            return await _call()
        return await asyncio.wait_for(_call(), timeout=budget_s)
    except asyncio.TimeoutError:
        log.info("model call exceeded %.2fs budget; using heuristic", budget_s)
        return None
    except anthropic.APIError as e:
        log.warning("model call failed: %s", e)
        return None


async def _json_fallback(client, schema: type[T], kwargs: dict) -> T | None:
    kwargs = dict(kwargs)
    kwargs["system"] = list(kwargs["system"]) + [
        {
            "type": "text",
            "text": (
                "Reply with a single JSON object matching this schema and nothing "
                "else — no prose, no code fence:\n"
                + json.dumps(schema.model_json_schema())
            ),
        }
    ]
    response = await client.messages.create(**kwargs)
    _record(response)
    text = next((b.text for b in response.content if b.type == "text"), "")
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        return schema.model_validate_json(text)
    except ValidationError as e:
        log.warning("JSON fallback failed validation: %s", e)
        return None


async def text(
    *,
    system: str,
    user: str,
    model: str,
    max_tokens: int,
    budget_s: float | None = None,
    cache_system: bool = True,
) -> str | None:
    """Plain-text completion with the same budget/offline semantics."""
    if get_settings().offline:
        return None

    async def _call() -> str:
        response = await get_client().messages.create(
            model=model,
            max_tokens=max_tokens,
            system=_system_blocks(system, cache_system),
            messages=[{"role": "user", "content": user}],
        )
        _record(response)
        return next((b.text for b in response.content if b.type == "text"), "")

    try:
        if budget_s is None:
            return await _call()
        return await asyncio.wait_for(_call(), timeout=budget_s)
    except asyncio.TimeoutError:
        return None
    except anthropic.APIError as e:
        log.warning("model call failed: %s", e)
        return None
