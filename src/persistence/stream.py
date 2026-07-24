"""Live debate turns over Redis.

Two structures per run, because a viewer who opens the page mid-debate should
still see how the argument started:

  run:{id}:turns    a list  — every turn so far, for replay on connect
  run:{id}:channel  pub/sub — turns as they happen, for the live tail

The SSE endpoint drains the list, then tails the channel. Publishing is
best-effort: a debate must not die because Redis blinked. Losing a turn costs a
UI update; raising here would cost the run.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import redis

from ..config import get_settings
from ..state.schema import DebateTurn

#: Turns outlive the run by an hour so a late viewer can still replay it.
_TTL_SECONDS = 3600

#: Sentinel published when a run reaches a terminal state, so clients stop tailing.
DONE = "__done__"


def _client() -> redis.Redis:
    return redis.from_url(get_settings().redis_url, decode_responses=True)


def _keys(run_id: str) -> tuple[str, str]:
    return f"run:{run_id}:turns", f"run:{run_id}:channel"


def publish_turn(run_id: str, turn: DebateTurn) -> None:
    """Append a turn to the replay list and fan it out live. Never raises."""
    list_key, channel = _keys(run_id)
    payload = turn.model_dump_json()
    try:
        client = _client()
        pipe = client.pipeline()
        pipe.rpush(list_key, payload)
        pipe.expire(list_key, _TTL_SECONDS)
        pipe.publish(channel, payload)
        pipe.execute()
    except redis.RedisError:
        pass


def publish_done(run_id: str) -> None:
    """Tell tailing clients the run is over. Never raises."""
    _, channel = _keys(run_id)
    try:
        _client().publish(channel, DONE)
    except redis.RedisError:
        pass


def past_turns(run_id: str) -> list[dict]:
    """Every turn recorded so far. Empty list if Redis is unreachable."""
    list_key, _ = _keys(run_id)
    try:
        return [json.loads(raw) for raw in _client().lrange(list_key, 0, -1)]
    except redis.RedisError:
        return []


def tail(run_id: str, timeout: float = 1.0) -> Iterator[str]:
    """Yield raw turn payloads as they are published, ending on the DONE sentinel.

    Yields nothing and returns cleanly if Redis is unreachable, so the API can
    still serve the replay it already has.
    """
    _, channel = _keys(run_id)
    try:
        pubsub = _client().pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe(channel)
    except redis.RedisError:
        return

    try:
        while True:
            message = pubsub.get_message(timeout=timeout)
            if message is None:
                yield ""  # keep-alive: lets the caller notice a dropped client
                continue
            data = message.get("data")
            if data == DONE:
                return
            if data:
                yield str(data)
    except redis.RedisError:
        return
    finally:
        try:
            pubsub.close()
        except redis.RedisError:
            pass
