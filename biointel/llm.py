"""Thin async wrapper around the Claude Messages API with structured outputs.

* ``client.beta.messages.parse(output_format=<PydanticModel>)`` -> validated object
* thinking is adaptive by default on current models; depth is controlled with ``effort``
* server-side refusal fallbacks are enabled (biomedical text can trip safety classifiers)
* refusal / truncation raise typed errors so the orchestrator can degrade one module, not the run
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

from .config import Settings

T = TypeVar("T", bound=BaseModel)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    pass


class LLMRefusal(LLMError):
    pass


class LLMTruncated(LLMError):
    pass


class LLMAccountError(LLMError):
    """Billing / authentication problem: every call will fail, so the run must stop."""


def _account_error(exc) -> LLMAccountError | None:
    import anthropic

    msg = (getattr(exc, "message", "") or str(exc)).lower()
    if isinstance(exc, anthropic.AuthenticationError):
        return LLMAccountError("The Anthropic API key was rejected. Check ANTHROPIC_API_KEY in .streamlit/secrets.toml "
                               "(local) or the app's Settings -> Secrets (Streamlit Cloud).")
    if isinstance(exc, anthropic.PermissionDeniedError):
        return LLMAccountError("This Anthropic API key doesn't have access to the configured models.")
    if "credit balance" in msg or "billing" in msg:
        return LLMAccountError("Your Anthropic account has no API credits left. Add credits at platform.claude.com "
                               "-> Plans & Billing, then generate the briefing again.")
    return None


class LLM(Protocol):
    usage: dict[str, int]

    async def parse(
        self, *, model: str, system: str, user: str, output_format: type[T], effort: str, max_tokens: int | None = None
    ) -> T: ...


class ClaudeLLM:
    def __init__(self, settings: Settings) -> None:
        if not settings.anthropic_api_key:
            raise LLMError(
                "ANTHROPIC_API_KEY is not configured. Add it to .streamlit/secrets.toml (local) "
                "or the app's Settings -> Secrets (Streamlit Cloud)."
            )
        from anthropic import AsyncAnthropic

        self.settings = settings
        self.client = AsyncAnthropic(api_key=settings.anthropic_api_key, max_retries=2, timeout=150.0)
        self.usage: dict[str, int] = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        self.fatal: LLMAccountError | None = None  # set on the first billing/auth failure

    async def preflight(self) -> None:
        """One tiny request to surface billing / auth problems before any real work starts."""
        import anthropic

        try:
            await self.client.messages.create(
                model=self.settings.agent_model, max_tokens=16, messages=[{"role": "user", "content": "Reply OK"}],
                output_config={"effort": "low"},
            )
        except anthropic.APIStatusError as exc:
            if err := _account_error(exc):
                self.fatal = err
                raise err from exc
            raise LLMError(f"Claude API check failed ({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Could not reach the Claude API: {exc}") from exc

    async def parse(
        self, *, model: str, system: str, user: str, output_format: type[T], effort: str, max_tokens: int | None = None
    ) -> T:
        import anthropic

        if self.fatal:  # don't keep calling an account that cannot serve requests
            raise self.fatal
        try:
            resp = await self.client.beta.messages.parse(
                model=model,
                max_tokens=max_tokens or self.settings.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=output_format,
                output_config={"effort": effort},
                fallbacks="default",
                betas=[FALLBACK_BETA],
            )
        except (anthropic.BadRequestError, anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            if err := _account_error(exc):
                self.fatal = err
                raise err from exc
            raise LLMError(f"Claude rejected the request ({model}): {exc.message}") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Claude rate limit reached - try again shortly.") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Claude API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Could not reach the Claude API: {exc}") from exc

        self.usage["calls"] += 1
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.usage["input_tokens"] += int(getattr(usage, "input_tokens", 0) or 0)
            self.usage["output_tokens"] += int(getattr(usage, "output_tokens", 0) or 0)

        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise LLMRefusal(f"Claude declined this section (category: {category or 'unspecified'}).")
        if resp.stop_reason == "max_tokens":
            raise LLMTruncated("Claude's output hit the max_tokens limit before the section was complete.")
        parsed = resp.parsed_output
        if parsed is None:
            raise LLMError("Claude returned no structured output.")
        return parsed
