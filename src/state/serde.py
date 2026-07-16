"""JSON round-tripping for RunState.

SQLite stores each transition as JSON, and a run must be replayable, so decoding
has to rebuild the Pydantic payloads rather than leaving them as bare dicts.
The field->model map below is the authority for that; a new typed field in
RunState needs an entry here or it round-trips as a plain dict.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from .schema import (
    AgentMessage,
    BiasReport,
    Claim,
    DebateTurn,
    RunState,
    RunStatus,
    Source,
    Subtask,
    TokenUsage,
    Verdict,
)

# field -> model, for fields holding a single Pydantic payload
_SCALAR_MODELS: dict[str, type[BaseModel]] = {
    "bias_report": BiasReport,
    "verdict": Verdict,
}

# field -> model, for fields holding a list of Pydantic payloads
_LIST_MODELS: dict[str, type[BaseModel]] = {
    "subtasks": Subtask,
    "sources": Source,
    "raw_sources": Source,
    "claims": Claim,
    "debate_transcript": DebateTurn,
    "messages": AgentMessage,
    "token_usage": TokenUsage,
}

_ENUMS: dict[str, type] = {"status": RunStatus}


def _default(obj: Any) -> Any:
    """Fallback encoder for types json doesn't know."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    raise TypeError(f"Cannot serialize {type(obj).__name__}")


def state_to_json(state: RunState) -> str:
    """Serialize a RunState to a JSON string."""
    return json.dumps(dict(state), default=_default)


def state_from_json(raw: str) -> RunState:
    """Rebuild a RunState, restoring Pydantic payloads and enums."""
    data: dict[str, Any] = json.loads(raw)

    for field, model in _SCALAR_MODELS.items():
        if data.get(field) is not None:
            data[field] = model.model_validate(data[field])

    for field, model in _LIST_MODELS.items():
        if data.get(field):
            data[field] = [model.model_validate(x) for x in data[field]]

    for field, enum in _ENUMS.items():
        if data.get(field) is not None:
            data[field] = enum(data[field])

    return RunState(**data)  # type: ignore[typeddict-item]
