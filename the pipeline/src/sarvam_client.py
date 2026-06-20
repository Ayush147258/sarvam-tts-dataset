"""
Thin wrapper around the official `sarvamai` SDK. Centralizes:
  - client instantiation, using settings.sarvam_api_key from src/config.py
  - retry-with-exponential-backoff on rate-limit/5xx errors
  - a single get_client() function every other module imports, so there's
    one place to change auth/retry/timeout behavior for the whole project

CRITICAL SDK GOTCHA — verified against the real installed package, not
assumed: the sarvamai SDK has NO `.create()` method on chat completions
or speech-to-text. The call pattern is the method itself, called
directly:
    client.chat.completions(messages=[...], model="sarvam-105b")
    client.speech_to_text.transcribe(file=..., model="saaras:v3", ...)
NOT client.chat.completions.create(...) — that will raise AttributeError.
This exact gotcha is documented in Sarvam's own `sarvamai/skills` GitHub
repo, in a SKILL.md written specifically to catch mistakes AI coding
agents commonly make with this SDK (install via `npx skills add
sarvamai/skills` if using an AI coding tool to write pipeline code).

Error handling — verified against the actual installed sarvamai==0.1.28
source rather than guessed: the SDK raises a real, named exception
hierarchy under `sarvamai.errors`, all subclassing `ApiError` (itself a
plain Exception with a `.status_code` attribute set from the HTTP
response). The relevant ones for retry purposes:
    TooManyRequestsError   -> status_code 429 (rate limit)
    InternalServerError    -> status_code 500
    ServiceUnavailableError-> status_code 503
This module catches `ApiError` broadly and inspects `.status_code` rather
than catching each named class individually, so it stays correct even if
the SDK adds more 5xx-mapped error classes in a future version.

Usage:
    from src.sarvam_client import get_client, call_with_retry

    client = get_client()
    result = call_with_retry(
        lambda: client.chat.completions(messages=[...], model="sarvam-105b")
    )
"""

from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

from rich.console import Console
from sarvamai import SarvamAI
from sarvamai.core.api_error import ApiError

from src.config import require_sarvam_api_key

console = Console()

T = TypeVar("T")

# Hand-rolled retry rather than `tenacity`: tenacity is not installed by
# default in this project's dependency list (see BUILD_GUIDE.md Prompt 1),
# and pulling it in for a handful of retry calls isn't worth a new
# dependency. If the project later needs more elaborate retry policies,
# swapping this for tenacity's @retry decorator is a contained change —
# every caller already goes through call_with_retry() below.
MAX_RETRY_ATTEMPTS = 4
BASE_BACKOFF_SEC = 1.0  # exponential: ~1s, 2s, 4s, 8s, plus jitter

# HTTP status codes worth retrying. 429 = rate limited, 500/502/503/504 =
# server-side transient errors. 4xx codes other than 429 (bad request,
# auth failure, not found, etc.) are NOT retried — retrying a malformed
# request just repeats the same failure four times slower.
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

_client: SarvamAI | None = None


def get_client() -> SarvamAI:
    """
    Return a shared SarvamAI client instance, constructed from
    settings.sarvam_api_key (src/config.py). Lazily instantiated and
    cached at module level so repeated calls don't recreate the client
    (and its underlying HTTP connection pool) needlessly.
    """
    global _client
    if _client is None:
        _client = SarvamAI(api_subscription_key=require_sarvam_api_key())
        console.log("[green]✓[/green] Sarvam client initialized")
    return _client


def call_with_retry(
    fn: Callable[[], T],
    max_attempts: int = MAX_RETRY_ATTEMPTS,
    base_backoff_sec: float = BASE_BACKOFF_SEC,
    label: str = "sarvam_api_call",
) -> T:
    """
    Call `fn()` (a zero-argument callable wrapping a Sarvam SDK call),
    retrying with exponential backoff + jitter if it raises an ApiError
    whose status_code is in RETRYABLE_STATUS_CODES.

    Non-retryable ApiErrors (bad request, auth failure, not found, etc.)
    and any non-ApiError exception are raised immediately on the first
    attempt — retrying those just wastes time on a failure that will
    never succeed.

    Usage:
        result = call_with_retry(
            lambda: client.chat.completions(messages=[...], model="sarvam-105b")
        )
    """
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except ApiError as e:
            last_error = e
            status = getattr(e, "status_code", None)

            if status not in RETRYABLE_STATUS_CODES:
                console.log(
                    f"[red]✗[/red] [{label}] Non-retryable API error "
                    f"(status_code={status}): {e}"
                )
                raise

            if attempt < max_attempts:
                backoff = base_backoff_sec * (2 ** (attempt - 1))
                backoff += random.uniform(0, backoff * 0.25)  # jitter
                console.log(
                    f"[yellow]⚠[/yellow] [{label}] Retryable API error "
                    f"(status_code={status}) on attempt {attempt}/"
                    f"{max_attempts}: {e}. Retrying in {backoff:.1f}s..."
                )
                time.sleep(backoff)
            else:
                console.log(
                    f"[red]✗[/red] [{label}] Failed after {max_attempts} "
                    f"attempts (status_code={status}): {e}"
                )
                raise

    # Unreachable in practice (the loop always returns or raises), but
    # keeps type checkers happy and guards against a future refactor bug.
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"[{label}] call_with_retry exited with no result and no error")


if __name__ == "__main__":
    # Quick manual smoke test — see BUILD_GUIDE.md Prompt 8 verify step.
    # This only confirms the client constructs without error; it does
    # NOT make a real API call (no API credits spent).
    client = get_client()
    console.log(f"[bold green]Smoke test passed.[/bold green] Client: {client}")
