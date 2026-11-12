"""GPT-Live primary WebSocket protocol for the Twilio phone bridge."""

import asyncio
import base64
import json
import os
from bisect import bisect_left
from contextlib import asynccontextmanager

import websockets

from phone_dialogue import GREETING, ROLES, build_phone_dialogue


MODEL = os.getenv("OPENAI_PHONE_MODEL", "").strip() or "gpt-live-1"
RATE = 24000
URL = "wss://api.openai.com/v1/live/sessions"


AUDIO_FORMATS = {
    "pcm": {"type": "audio/pcm", "rate": RATE},
    "pcmu": {"type": "audio/pcmu", "rate": 8000},
}


def phone_session(ai_role, initiated_by, player_role, history, codec="pcm"):
    instructions, context = build_phone_dialogue(ai_role, initiated_by, player_role, history)
    voice = os.getenv(f"OPENAI_PHONE_VOICE_{ai_role.upper()}", "").strip() or ROLES[ai_role].voice
    if voice.startswith("voice_"):
        voice = {"id": voice}
    items = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": context}]}]
    if initiated_by == "admin":
        items.append({"type": "message", "role": "developer", "content": [{"type": "input_text", "text": GREETING}]})
    return {
        "model": MODEL,
        "instructions": instructions,
        "input": items,
        "audio": {"format": AUDIO_FORMATS[codec], "output": {"voice": voice}},
        "delegation": {"type": "client"},
        "store": False,
    }


@asynccontextmanager
async def connect(api_key, config):
    async with websockets.connect(
        URL, extra_headers={"Authorization": f"Bearer {api_key}"},
        open_timeout=10, close_timeout=3, max_size=2 * 1024 * 1024,
    ) as upstream:
        await send(upstream, "session.start", session=config)
        first = json.loads(await asyncio.wait_for(upstream.recv(), timeout=10))
        if first.get("type") != "session.started":
            raise RuntimeError("GPT-Live did not confirm the session")
        yield upstream


async def send(upstream, kind, **fields):
    await upstream.send(json.dumps({"type": kind, **fields}, ensure_ascii=False))


async def send_audio(upstream, audio):
    if audio:
        await send(upstream, "session.input_audio.append", audio=base64.b64encode(audio).decode("ascii"))


def transcript_turns(deltas):
    """Group each channel independently, then interleave the timestamped turns."""
    turns = []
    seen = set()
    channels = {}
    intervals = {"user": [], "assistant": []}
    for role, event in deltas:
        start, end = event.get("start_ms"), event.get("end_ms")
        if event.get("delta") and isinstance(start, int) and isinstance(end, int):
            intervals[role].append((start, end))
    for track in intervals.values():
        track.sort()
    for role, event in deltas:
        event_id = event.get("event_id")
        fragment = event.get("delta")
        if (event_id and event_id in seen) or not isinstance(fragment, str) or not fragment:
            continue
        if event_id:
            seen.add(event_id)
        start, end = event.get("start_ms"), event.get("end_ms")
        previous = channels.get(role)
        if previous and isinstance(start, int) and isinstance(previous.get("end_ms"), int):
            previous_end = previous["end_ms"]
            other = intervals["assistant" if role == "user" else "user"]
            first = bisect_left(other, (previous_end,))
            other_spoke_between = (
                start > previous_end and first < len(other) and other[first][1] <= start
            )
            if start - previous_end > 500 or other_spoke_between:
                previous = None
        if previous is None:
            previous = {"role": role, "text": ""}
            if isinstance(start, int):
                previous["start_ms"] = start
            channels[role] = previous
            turns.append(previous)
        previous["text"] += fragment
        if isinstance(end, int):
            previous["end_ms"] = max(end, previous.get("end_ms", 0))
    turns.sort(key=lambda turn: turn.get("start_ms", float("inf")))
    return [dict(turn, text=turn["text"].strip()) for turn in turns if turn["text"].strip()]
