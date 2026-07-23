"""LLM access with per-run cost accounting.

Every agent goes through `Budget` rather than touching ChatOpenAI directly,
because the spend cap is only enforceable if all traffic passes one chokepoint.

Why an accumulator instead of writing straight into `state["token_usage"]`:
researchers run as parallel branches, and a node mutating a shared channel list
in place either races with its siblings or double-counts against the `operator.add`
reducer. So a node collects its own spend and *returns* it; the reducer merges.
"""

from __future__ import annotations

from typing import TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from ..config import get_settings
from ..state.schema import RunState, TokenUsage

T = TypeVar("T", bound=BaseModel)

#: USD per 1M tokens, (input, output). Used for the per-run cap; update alongside
#: OpenAI's pricing page. An unknown model is treated as free rather than
#: guessed at — the cap is a safety net, not billing.
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
}


class CostCapExceeded(RuntimeError):
    """The run hit its LLM spend cap and must stop."""


class SearchCapExceeded(RuntimeError):
    """The run hit its web-search call cap and must stop."""


def _model(name: str, temperature: float) -> ChatOpenAI:
    settings = get_settings()
    extra = {"base_url": settings.openai_base_url} if settings.openai_base_url else {}
    return ChatOpenAI(
        model=name,
        temperature=temperature,
        api_key=settings.openai_api_key,
        timeout=90,
        max_retries=2,
        **extra,
    )


def price(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """USD cost of one call. Unknown models cost 0 — the cap is a net, not billing."""
    rates = _PRICING.get(model)
    if not rates:
        return 0.0
    return (prompt_tokens * rates[0] + completion_tokens * rates[1]) / 1_000_000


def total_cost(state: RunState) -> float:
    """Total USD spent on LLM calls so far in this run."""
    return sum(u.cost_usd for u in state.get("token_usage", []))


class Budget:
    """A node's LLM channel: enforces the run cap, records what it spends.

    Create one per node, make calls through it, and return `.usage` from the node
    so the reducer folds the spend back into the run.
    """

    def __init__(self, state: RunState, node: str) -> None:
        self._prior = total_cost(state)
        self._node = node
        self.usage: list[TokenUsage] = []

    @property
    def spent(self) -> float:
        """USD spent by the whole run including this node so far."""
        return self._prior + sum(u.cost_usd for u in self.usage)

    def _check_cap(self) -> None:
        cap = get_settings().max_run_cost_usd
        if self.spent >= cap:
            raise CostCapExceeded(
                f"Run has spent ${self.spent:.3f} of its ${cap:.2f} LLM budget."
            )

    def _record(self, model: str, meta: dict) -> None:
        usage = meta.get("token_usage") or meta.get("usage") or {}
        prompt = int(usage.get("prompt_tokens", 0))
        completion = int(usage.get("completion_tokens", 0))
        self.usage.append(
            TokenUsage(
                node=self._node,
                model=model,
                prompt_tokens=prompt,
                completion_tokens=completion,
                cost_usd=price(model, prompt, completion),
            )
        )

    def text(
        self, system: str, user: str, *, cheap: bool = False, temperature: float = 0.0
    ) -> str:
        """Send one prompt and return the text reply, billing it to this node."""
        self._check_cap()
        settings = get_settings()
        name = settings.cheap_model if cheap else settings.reasoning_model

        reply = _model(name, temperature).invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        self._record(name, reply.response_metadata or {})
        return str(reply.content)

    def structured(
        self,
        system: str,
        user: str,
        schema: type[T],
        *,
        cheap: bool = False,
        temperature: float = 0.0,
    ) -> T:
        """Send one prompt and get back a validated `schema` instance.

        `include_raw` is on so usage metadata survives — the structured-output
        wrapper otherwise hands back only the parsed model and the run's spend
        would silently under-count.
        """
        self._check_cap()
        settings = get_settings()
        name = settings.cheap_model if cheap else settings.reasoning_model

        bound = _model(name, temperature).with_structured_output(schema, include_raw=True)
        out = bound.invoke([SystemMessage(content=system), HumanMessage(content=user)])

        raw = out.get("raw") if isinstance(out, dict) else None
        if raw is not None:
            self._record(name, getattr(raw, "response_metadata", {}) or {})

        parsed = out.get("parsed") if isinstance(out, dict) else out
        if parsed is None:
            raise ValueError(f"{self._node}: model did not return valid {schema.__name__}")
        return parsed  # type: ignore[return-value]
