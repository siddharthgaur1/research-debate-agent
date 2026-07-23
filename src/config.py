"""Env-driven settings, validated at import time.

Every secret, budget and path the system needs is declared here. Missing required
vars raise at startup rather than surfacing as an AttributeError six nodes deep
into a debate.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Required fields have no default and fail loudly."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    openai_api_key: str = Field(..., description="LLM API key (OpenAI or compatible).")
    # Any OpenAI-compatible endpoint. Empty = OpenAI; set it to run chat on a free
    # tier (Groq: https://api.groq.com/openai/v1, then reasoning_model=
    # llama-3.3-70b-versatile) or a local Ollama (http://localhost:11434/v1).
    openai_base_url: str = Field(default="", description="OpenAI-compatible base URL; empty = OpenAI.")
    # Optional: the runner degrades to an in-memory checkpointer if Redis is
    # unreachable, so a single API key is enough to run.
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis URL for streaming and checkpoints.")
    db_path: Path = Field(default=Path("data/runs.db"), description="SQLite file holding run history.")
    runs_dir: Path = Field(
        default=Path("runs"), description="Root for per-run artifacts (PDFs)."
    )

    # --- search ---
    search_provider: Literal["tavily", "serpapi"] = Field(
        default="tavily", description="Which search backend to use."
    )
    tavily_api_key: str | None = Field(default=None, description="Tavily API key.")
    serpapi_key: str | None = Field(default=None, description="SerpAPI key.")

    # --- models ---
    reasoning_model: str = Field(
        default="gpt-4o", description="Model for debate and arbitration."
    )
    cheap_model: str = Field(
        default="gpt-4o-mini", description="Model for summarisation."
    )
    embedding_model: str = Field(
        default="text-embedding-3-small", description="Model for source dedup."
    )

    # --- chroma ---
    chroma_host: str | None = Field(
        default=None, description="Chroma server host. Unset uses a local file store."
    )
    chroma_port: int = Field(default=8000, description="Chroma server port.")
    chroma_dir: Path = Field(
        default=Path("chroma_db"), description="Local Chroma path when no host is set."
    )
    dedup_threshold: float = Field(
        default=0.92,
        description="Cosine similarity at or above which two sources are the same.",
    )

    # --- budgets ---
    max_run_cost_usd: float = Field(
        default=2.0, description="Hard cap on LLM spend per run."
    )
    max_searches_per_run: int = Field(
        default=20, description="Hard cap on web-search calls per run."
    )
    results_per_subtask: int = Field(
        default=5, description="Search results requested per research subtask."
    )
    max_debate_rounds: int = Field(
        default=2, description="Advocate/Critic exchanges before the Bias Checker."
    )
    fetch_char_budget: int = Field(
        default=12_000, description="Chars of page text kept per source."
    )

    # --- uncertainty ---
    contested_margin: float = Field(
        default=0.15,
        description="Advocate/critic support within this margin means contested.",
    )
    uncertainty_ratio: float = Field(
        default=0.34,
        description="Fraction of contested claims that trips uncertainty mode.",
    )

    @model_validator(mode="after")
    def _check(self) -> "Settings":
        if not self.openai_api_key.strip():
            raise ValueError("OPENAI_API_KEY is set but empty.")
        key = self.tavily_api_key if self.search_provider == "tavily" else self.serpapi_key
        if not key or not key.strip():
            need = "TAVILY_API_KEY" if self.search_provider == "tavily" else "SERPAPI_KEY"
            raise ValueError(
                f"SEARCH_PROVIDER={self.search_provider} requires {need} to be set."
            )
        return self

    def run_dir(self, run_id: str) -> Path:
        """Return (and create) the artifact directory for a run."""
        d = self.runs_dir / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process. Raises ValidationError if a var is missing."""
    return Settings()
