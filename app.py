# -*- coding: utf-8 -*-
"""
voice_journalist_service.py

Service voix SYNIAL : Twilio (Media Streams) <-> GPT-Live

- /api/prepare_call : (protégé) prépare le moteur vocal, puis lance un appel Twilio sortant
- /twilio/stream    : WebSocket Twilio Media Streams (bidirectionnel) + bridge audio temps réel

⚠️ IMPORTANT :
- Twilio <Stream> exige wss:// et n'accepte pas de query string -> passer call_id via <Parameter>
- Si tu lances uvicorn avec plusieurs workers, la préchauffe "in-memory" ne marchera pas (un seul worker !).

Install :
pip install fastapi uvicorn twilio google-genai
"""

import os
import re
import json
import time
import html
import base64
import uuid
import asyncio
import logging
import sqlite3
import hmac
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import audioop  # stdlib (déprécié dans le futur, mais OK en 3.10-3.12)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response, HTTPException

from twilio.rest import Client as TwilioClient
import requests
import websockets
import openai_live_phone
from phone_postcall import CallRecording, PostCallQueue, deliver, gemini_client



# ============================================================
# 1) CONFIG
# ============================================================

# --- Twilio ---
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "").strip()

# Sécurité : API key pour déclencher l'appel depuis ton admin
VOICE_ADMIN_API_KEY = os.getenv("VOICE_ADMIN_API_KEY", "").strip()

# URL publique où Twilio peut appeler ce service (https)
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")

TWILIO_SAY_BEFORE_STREAM = os.getenv("TWILIO_SAY_BEFORE_STREAM", "").strip()
TWILIO_SAY_VOICE = os.getenv("TWILIO_SAY_VOICE", "alice").strip()
TWILIO_SAY_LANG = os.getenv("TWILIO_SAY_LANG", "fr-FR").strip()


TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "").strip() or "gemini-3.8-flash"

MAIN_APP_BASE_URL = os.getenv("MAIN_APP_BASE_URL", "").strip().rstrip("/")  # ex: https://synial.onrender.com
INTERNAL_APP_TOKEN = os.getenv("INTERNAL_APP_TOKEN", "").strip()           # = INTERNAL_VOICE_TOKEN côté Flask

# --- Casque Quest standalone ---
# Secret distinct de la clé d'administration : il n'est jamais renvoyé ni journalisé.
QUEST_DEVICE_TOKEN = os.getenv("QUEST_DEVICE_TOKEN", "").strip()
QUEST_DEVICE_TIMEOUT_SECONDS = max(20, int(os.getenv("QUEST_DEVICE_TIMEOUT_SECONDS", "45")))
QUEST_LONG_POLL_SECONDS = min(25, max(5, int(os.getenv("QUEST_LONG_POLL_SECONDS", "20"))))
QUEST_DEVICE_ID = "quest-main"

# OpenAI project credentials remain on this trusted service. The Quest only
# authenticates to the relay with its existing device token.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_LIVE_WEBSOCKET_URL = "wss://api.openai.com/v1/live/sessions"
VOICE_CALL_PROVIDER = os.getenv("VOICE_CALL_PROVIDER", "openai").strip().lower()

PROJECT_ID = (os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT") or "").strip()
LOCATION = (os.getenv("GOOGLE_CLOUD_LOCATION") or os.getenv("GOOGLE_CLOUD_REGION") or "us-central1").strip()
POSTCALL_LOCATION = os.getenv("VOICE_POSTCALL_GOOGLE_LOCATION", "").strip() or (
    "us" if TRANSCRIBE_MODEL == "gemini-3.8-flash" and LOCATION == "us-central1" else LOCATION
)

ENABLE_TRANSCRIPTIONS = os.getenv("ENABLE_TRANSCRIPTIONS", "true").lower() in ("1", "true", "yes", "y")
OPENAI_PHONE_CODEC = os.getenv("OPENAI_PHONE_CODEC", "pcm").strip().lower()
TWILIO_RATE_HZ = 8000
TWILIO_FRAME_BYTES = 160  # 20 ms
OUT_QUEUE_MAX_FRAMES = max(5, int(os.getenv("OUT_QUEUE_MAX_FRAMES", "25")))
OPENAI_PHONE_PLAYBACK_LEAD_SECONDS = min(.2, max(.02, int(os.getenv("OPENAI_PHONE_PLAYBACK_LEAD_MS", "60")) / 1000))
OPENAI_PHONE_CLOSE_TIMEOUT = 15
VOICE_RECORD_MAX_SECONDS = max(1, int(os.getenv("VOICE_RECORD_MAX_SECONDS", "1800")))
VOICE_POSTCALL_DIR = Path(os.getenv("VOICE_POSTCALL_DIR") or Path.home() / ".synial-phone-postcall")
PREPARED_SESSION_TTL_SECONDS = int(os.getenv("PREPARED_SESSION_TTL_SECONDS", "90"))

# --- Security hardening (optionnel) ---
# Exemple: "+33,+32" pour autoriser seulement FR/BE
ALLOWED_TO_PREFIXES = [p.strip() for p in os.getenv("ALLOWED_TO_PREFIXES", "").split(",") if p.strip()]


# ============================================================
# 2) LOGGING
# ============================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("voice_journalist")


# ============================================================
# 3) Helpers
# ============================================================

def _require_env(name: str, value: str) -> None:
    if not value:
        raise RuntimeError(f"Missing required config: {name}")


def _xml_escape(s: str) -> str:
    return html.escape(s, quote=True)


def _to_wss_url(public_base_url: str, path: str) -> str:
    if not public_base_url:
        raise RuntimeError("PUBLIC_BASE_URL is required")
    base = public_base_url.rstrip("/")
    if base.startswith("https://"):
        ws_base = "wss://" + base[len("https://") :]
    elif base.startswith("http://"):
        ws_base = "ws://" + base[len("http://") :]
    elif base.startswith("wss://") or base.startswith("ws://"):
        ws_base = base
    else:
        ws_base = "wss://" + base
    if not path.startswith("/"):
        path = "/" + path
    return ws_base + path


def build_twiml_stream(stream_ws_url: str, custom_parameters: Optional[Dict[str, str]] = None) -> str:
    parts = ['<?xml version="1.0" encoding="UTF-8"?>', "<Response>"]

    # Optionnel: TTS Twilio avant le stream (défaut vide)
    if TWILIO_SAY_BEFORE_STREAM:
        parts.append(
            f'<Say voice="{_xml_escape(TWILIO_SAY_VOICE)}" language="{_xml_escape(TWILIO_SAY_LANG)}">'
            f"{_xml_escape(TWILIO_SAY_BEFORE_STREAM)}</Say>"
        )

    parts.append("<Connect>")
    parts.append(f'<Stream url="{_xml_escape(stream_ws_url)}">')

    if custom_parameters:
        for k, v in custom_parameters.items():
            if k is None or v is None:
                continue
            k = str(k)[:200]
            v = str(v)[:250]
            parts.append(f'<Parameter name="{_xml_escape(k)}" value="{_xml_escape(v)}" />')

    parts.append("</Stream>")
    parts.append("</Connect>")
    parts.append("</Response>")
    return "".join(parts)


def validate_e164(number: str) -> bool:
    return bool(re.fullmatch(r"\+[1-9]\d{7,14}", number.strip()))


def _twilio_client() -> TwilioClient:
    _require_env("TWILIO_ACCOUNT_SID", TWILIO_ACCOUNT_SID)
    _require_env("TWILIO_AUTH_TOKEN", TWILIO_AUTH_TOKEN)
    return TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


# ============================================================
# 4) Audio conversion
# ============================================================

class AudioConverter:
    def __init__(self, codec):
        self.codec = codec
        self.in_state = None
        self.out_state = None
        self.trailing_pcm_byte = b""

    def from_twilio(self, ulaw):
        if self.codec == "pcmu":
            return ulaw
        pcm, self.in_state = audioop.ratecv(
            audioop.ulaw2lin(ulaw, 2), 2, 1, TWILIO_RATE_HZ, openai_live_phone.RATE, self.in_state
        )
        return pcm

    def to_twilio(self, chunk):
        if self.codec == "pcm":
            pcm = self.trailing_pcm_byte + chunk
            whole = len(pcm) & ~1
            self.trailing_pcm_byte = pcm[whole:]
            if not whole:
                return
            pcm, self.out_state = audioop.ratecv(
                pcm[:whole], 2, 1, openai_live_phone.RATE, TWILIO_RATE_HZ, self.out_state
            )
            chunk = audioop.lin2ulaw(pcm, 2)
        # Twilio accepts arbitrary chunk sizes. Send the short tail immediately.
        for index in range(0, len(chunk), TWILIO_FRAME_BYTES):
            yield chunk[index:index + TWILIO_FRAME_BYTES]


def normalize_ai_role(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace(" ", "_")
    s = s.replace("é", "e").replace("è", "e").replace("ê", "e").replace("à", "a").replace("ç", "c")
    return s or "journaliste"


@dataclass
class PreparedCall:
    call_id: str
    created_at: float
    expires_at: float

    player_name: str
    upstream_cm: Any
    upstream: Any
    codec: str
    initiated_by: str = "admin"
    ai_role: str = "journaliste"
    stream_token: str = field(default="", repr=False)

    twilio_call_sid: Optional[str] = None
    twilio_stream_sid: Optional[str] = None

    state: str = "ready"  # ready -> calling -> in_call -> ended
    cleanup_task: Optional[asyncio.Task] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    number_session: int = 0


PREPARED: Dict[str, PreparedCall] = {}
PREPARED_LOCK = asyncio.Lock()


# ============================================================
# 6b) Quest standalone store (un casque, une commande maximum)
# ============================================================

@dataclass
class QuestDeviceState:
    device_id: str
    last_seen: float = 0.0
    waiting: bool = False
    pending_launch_id: Optional[str] = None
    active_launch_id: Optional[str] = None


@dataclass
class QuestLaunchState:
    launch_id: str
    device_id: str
    command: Dict[str, Any]
    state: str = "pending"  # pending -> claimed -> started -> completed
    created_at: float = field(default_factory=time.time)
    claimed_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    transcript_forwarded: bool = False
    transcript_forwarding: bool = False


QUEST_DEVICES: Dict[str, QuestDeviceState] = {
    QUEST_DEVICE_ID: QuestDeviceState(device_id=QUEST_DEVICE_ID),
}
QUEST_LAUNCHES: Dict[str, QuestLaunchState] = {}
QUEST_LOCK = asyncio.Lock()
QUEST_OPENAI_RELAY_LOCK = asyncio.Lock()
QUEST_OPENAI_RELAY_ACTIVE = False


def _require_admin_api_key(request: Request) -> None:
    if not VOICE_ADMIN_API_KEY:
        raise HTTPException(status_code=500, detail="VOICE_ADMIN_API_KEY must be set")
    supplied = request.headers.get("X-API-Key", "").strip()
    if not hmac.compare_digest(supplied, VOICE_ADMIN_API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")


def _require_quest_token(request: Request) -> None:
    if not QUEST_DEVICE_TOKEN:
        raise HTTPException(status_code=503, detail="QUEST_DEVICE_TOKEN must be set")
    supplied = request.headers.get("X-Quest-Token", "").strip()
    if not hmac.compare_digest(supplied, QUEST_DEVICE_TOKEN):
        raise HTTPException(status_code=401, detail="Unauthorized")


def _quest_websocket_is_authorized(websocket: WebSocket) -> bool:
    """Authenticate without ever putting the device token in a URL or log."""
    if not QUEST_DEVICE_TOKEN:
        return False
    supplied = websocket.headers.get("X-Quest-Token", "").strip()
    return hmac.compare_digest(supplied, QUEST_DEVICE_TOKEN)


def _quest_device_or_404(device_id: str) -> QuestDeviceState:
    device = QUEST_DEVICES.get(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Casque Quest inconnu.")
    return device


def _quest_is_available(device: QuestDeviceState, now: Optional[float] = None) -> bool:
    current = time.time() if now is None else now
    fresh = bool(device.last_seen and current - device.last_seen <= QUEST_DEVICE_TIMEOUT_SECONDS)
    return fresh and device.waiting and not device.pending_launch_id and not device.active_launch_id


def _clean_quest_text(value: Any, max_length: int, default: str = "") -> str:
    text_value = str(value or "").strip()
    if not text_value:
        return default
    return text_value[:max_length]


def _quest_public_status(device: QuestDeviceState) -> Dict[str, Any]:
    available = _quest_is_available(device)
    launch_id = device.active_launch_id or device.pending_launch_id
    launch = QUEST_LAUNCHES.get(launch_id or "")
    return {
        "ok": True,
        "device_id": device.device_id,
        "label": "Quest principal",
        "available": available,
        "status": "Disponible" if available else "Indisponible",
        "launch_id": launch_id,
        "launch_state": launch.state if launch else None,
    }


def _quest_release_previous_active_launch(device: QuestDeviceState) -> None:
    """A new explicit wait session supersedes an interview lost with the old app process."""
    if not device.active_launch_id:
        return
    previous = QUEST_LAUNCHES.get(device.active_launch_id)
    if previous and previous.state in {"claimed", "started"}:
        previous.state = "abandoned"
    device.active_launch_id = None


def _forward_quest_transcript_to_flask(payload: Dict[str, Any]) -> tuple[bool, str]:
    if not MAIN_APP_BASE_URL:
        return False, "MAIN_APP_BASE_URL is not configured"
    if not INTERNAL_APP_TOKEN:
        return False, "INTERNAL_APP_TOKEN is not configured"
    try:
        response = requests.post(
            f"{MAIN_APP_BASE_URL}/internal/quest/transcript",
            json=payload,
            headers={"X-Internal-Token": INTERNAL_APP_TOKEN},
            timeout=20,
        )
    except requests.RequestException as exc:
        return False, f"Main SYNIAL site unreachable: {type(exc).__name__}"
    if 200 <= response.status_code < 300:
        return True, ""
    try:
        detail = str((response.json() or {}).get("detail") or "")
    except Exception:
        detail = ""
    return False, detail or f"Main SYNIAL site returned HTTP {response.status_code}"


async def _close_prepared_call(call_id: str, reason: str) -> None:
    async with PREPARED_LOCK:
        pc = PREPARED.pop(call_id, None)

    if not pc:
        return

    async with pc.lock:
        pc.state = "ended"

    if pc.cleanup_task and pc.cleanup_task is not asyncio.current_task():
        try:
            pc.cleanup_task.cancel()
        except Exception:
            pass

    try:
        await pc.upstream_cm.__aexit__(None, None, None)
    except Exception as e:
        logger.warning("[%s] failed to close voice session (%s): %s", call_id, reason, type(e).__name__)


async def _cleanup_expired_call(call_id: str) -> None:
    try:
        await asyncio.sleep(PREPARED_SESSION_TTL_SECONDS)
        async with PREPARED_LOCK:
            pc = PREPARED.get(call_id)
        if not pc:
            return
        if time.time() >= pc.expires_at and pc.state in ("ready", "calling"):
            logger.info("[%s] cleanup TTL reached (state=%s) -> closing", call_id, pc.state)
            await _close_prepared_call(call_id, reason="ttl")
    except asyncio.CancelledError:
        return
    except Exception as e:
        logger.warning("[%s] cleanup task error: %s", call_id, e)


# ============================================================
# 7) Twilio <-> GPT-Live bridge
# ============================================================

@dataclass
class StreamContext:
    stream_sid: str
    call_sid: str
    started_at: float = field(default_factory=time.monotonic)
    inbound_end: int = 0
    played_end: int = 0
    sent_end: int = 0
    marks: dict = field(default_factory=dict)
    first_output_ms: Optional[int] = None
    max_mark_delay_ms: int = 0
    max_queue_frames: int = 0
    playback_uncertain: bool = False
    last_media_at: Optional[float] = None
    ended_at: float = 0.0
    optional_events: set = field(default_factory=set)

    def position(self):
        return max(0, round((time.monotonic() - self.started_at) * TWILIO_RATE_HZ))


async def openai_phone_receiver_loop(session, converter, queue, deltas, stopping, ctx):
    async for raw in session:
        event = json.loads(raw)
        kind = event.get("type")
        if kind == "session.closed":
            return True
        if kind == "error":
            error = event.get("error") or {}
            if (event.get("client_event_id") or error.get("client_event_id")) in ctx.optional_events:
                logger.warning("GPT-Live rejected an optional phone instruction")
                continue
            raise RuntimeError("GPT-Live stream error")
        ctx.optional_events.discard(event.get("client_event_id"))
        if kind in {"session.input_transcript.delta", "session.output_transcript.delta"}:
            if ENABLE_TRANSCRIPTIONS:
                role = "user" if kind == "session.input_transcript.delta" else "assistant"
                deltas.append((role, event))
        elif kind == "session.output_audio.delta" and not stopping.is_set():
            for frame in converter.to_twilio(base64.b64decode(event["delta"], validate=True)):
                if stopping.is_set():
                    ctx.playback_uncertain = True
                    break
                await asyncio.wait_for(queue.put(frame), timeout=1)
                ctx.max_queue_frames = max(ctx.max_queue_frames, queue.qsize())
        elif kind == "session.delegation.created" and not stopping.is_set():
            delegation = event.get("delegation") or {}
            if delegation.get("target") == "client" and delegation.get("id"):
                event_id = uuid.uuid4().hex
                ctx.optional_events.add(event_id)
                await asyncio.wait_for(openai_live_phone.send(
                    session, "session.thinking.append", event_id=event_id,
                    delegation_id=delegation["id"],
                    content="Le dossier de simulation est déjà disponible. Aucune source ni vérification externe. "
                            "Utilise les éléments fournis ou indique l'incertitude, en restant dans le rôle.",
                ), timeout=3)
    return False


async def twilio_sender_loop(websocket, ctx, queue, recording):
    sequence = 0
    lead = round(OPENAI_PHONE_PLAYBACK_LEAD_SECONDS * TWILIO_RATE_HZ)
    while True:
        frame = await queue.get()
        sent = False
        try:
            wait = (ctx.sent_end + len(frame) - lead - ctx.position()) / TWILIO_RATE_HZ
            if wait > 0:
                await asyncio.sleep(wait)
            start = max(ctx.sent_end, ctx.position())
            payload = base64.b64encode(frame).decode("ascii")
            await asyncio.wait_for(websocket.send_json({
                "event": "media", "streamSid": ctx.stream_sid, "media": {"payload": payload},
            }), timeout=3)
            sent = True
            ctx.sent_end = start + len(frame)
            if recording is not None:
                recording.add("assistant", start, frame)
            if ctx.first_output_ms is None:
                ctx.first_output_ms = start * 1000 // TWILIO_RATE_HZ
            sequence += 1
            if sequence == 1 or sequence % 5 == 0:
                name = str(sequence)
                ctx.marks[name] = ctx.sent_end
                if len(ctx.marks) > 100:
                    del ctx.marks[next(iter(ctx.marks))]
                await asyncio.wait_for(websocket.send_json({
                    "event": "mark", "streamSid": ctx.stream_sid, "mark": {"name": name},
                }), timeout=3)
        finally:
            if not sent:
                ctx.playback_uncertain = True
            queue.task_done()


async def twilio_receiver_loop(websocket, ctx, session, converter, recording, initiated_by):
    greeting_pending = initiated_by == "admin"
    while True:
        event = await websocket.receive_json()
        kind = event.get("event")
        if kind == "stop":
            return
        if kind == "mark":
            name = (event.get("mark") or {}).get("name")
            end = ctx.marks.pop(name, None)
            if end is not None:
                ctx.played_end = max(ctx.played_end, end)
                ctx.max_mark_delay_ms = max(ctx.max_mark_delay_ms, (ctx.position() - end) * 1000 // TWILIO_RATE_HZ)
        if kind != "media":
            continue
        media = event.get("media") or {}
        if media.get("track", "inbound") != "inbound" or not media.get("payload"):
            continue
        ulaw = base64.b64decode(media["payload"], validate=True)
        try:
            start = max(0, int(media["timestamp"])) * TWILIO_RATE_HZ // 1000
        except (TypeError, ValueError, KeyError):
            start = ctx.inbound_end
        ctx.inbound_end = max(ctx.inbound_end, start + len(ulaw))
        ctx.last_media_at = time.monotonic()
        if recording is not None:
            recording.add("user", start, ulaw)
        await asyncio.wait_for(openai_live_phone.send_audio(session, converter.from_twilio(ulaw)), timeout=3)
        if greeting_pending:
            greeting_pending = False
            event_id = uuid.uuid4().hex
            ctx.optional_events.add(event_id)
            await asyncio.wait_for(openai_live_phone.send(
                session, "session.commentary.append", event_id=event_id,
                delegation_id=None, content=openai_live_phone.GREETING,
            ), timeout=3)


def _twilio_hangup_call(call_sid: str) -> None:
    client = _twilio_client()
    client.calls(call_sid).update(status="completed")


# ============================================================
# 8) FastAPI app
# ============================================================

@asynccontextmanager
async def lifespan(app):
    app.state.phone_finalizers = set()
    app.state.phone_postcall = PostCallQueue(
        VOICE_POSTCALL_DIR / "calls.sqlite3", TRANSCRIBE_MODEL,
        lambda: gemini_client(PROJECT_ID, POSTCALL_LOCATION),
        lambda payload: deliver(MAIN_APP_BASE_URL, INTERNAL_APP_TOKEN, payload),
    )
    worker = asyncio.create_task(app.state.phone_postcall.run())
    try:
        yield
    finally:
        await asyncio.gather(*app.state.phone_finalizers, return_exceptions=True)
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"ok": True, "service": "voice_journalist", "model": openai_live_phone.MODEL}


@app.get("/health")
async def health():
    return {
        "ok": True,
        "project": PROJECT_ID,
        "location": LOCATION,
        "model": openai_live_phone.MODEL,
        "phone_provider": VOICE_CALL_PROVIDER,
        "phone_codec": OPENAI_PHONE_CODEC,
        "phone_provider_configured": bool(OPENAI_API_KEY and VOICE_CALL_PROVIDER == "openai" and OPENAI_PHONE_CODEC in openai_live_phone.AUDIO_FORMATS),
        "twilio_configured": bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER),
        "public_base_url": PUBLIC_BASE_URL,
        "prepared_sessions": len(PREPARED),
    }


@app.post("/api/quest/devices/{device_id}/heartbeat")
async def quest_device_heartbeat(device_id: str, request: Request):
    _require_quest_token(request)
    device = _quest_device_or_404(device_id)
    body = await request.json()
    wants_waiting = bool(body.get("waiting", True))
    async with QUEST_LOCK:
        if wants_waiting:
            _quest_release_previous_active_launch(device)
        device.last_seen = time.time()
        device.waiting = wants_waiting and not device.active_launch_id
        return _quest_public_status(device)


@app.get("/api/quest/devices/{device_id}/poll")
async def quest_device_poll(device_id: str, request: Request, timeout_seconds: int = QUEST_LONG_POLL_SECONDS):
    """Long-poll. Claiming is atomic and removes the one pending command immediately."""
    _require_quest_token(request)
    device = _quest_device_or_404(device_id)
    wait_seconds = min(QUEST_LONG_POLL_SECONDS, max(1, int(timeout_seconds)))
    deadline = time.monotonic() + wait_seconds

    while True:
        async with QUEST_LOCK:
            if device.active_launch_id:
                _quest_release_previous_active_launch(device)
            device.last_seen = time.time()
            if not device.active_launch_id:
                device.waiting = True
            launch_id = device.pending_launch_id
            launch = QUEST_LAUNCHES.get(launch_id or "")
            if launch and launch.state == "pending" and not device.active_launch_id:
                device.pending_launch_id = None
                device.active_launch_id = launch.launch_id
                device.waiting = False
                launch.state = "claimed"
                launch.claimed_at = time.time()
                return {"ok": True, "command": launch.command}

        if time.monotonic() >= deadline:
            return Response(status_code=204)
        await asyncio.sleep(0.25)


@app.post("/api/quest/devices/{device_id}/leave")
async def quest_device_leave(device_id: str, request: Request):
    _require_quest_token(request)
    device = _quest_device_or_404(device_id)
    async with QUEST_LOCK:
        device.last_seen = time.time()
        device.waiting = False
        if device.pending_launch_id:
            launch = QUEST_LAUNCHES.get(device.pending_launch_id)
            if launch and launch.state == "pending":
                launch.state = "cancelled"
            device.pending_launch_id = None
        return {"ok": True}


@app.websocket("/api/quest/openai/live")
async def quest_openai_live_relay(websocket: WebSocket):
    """Transparent full-duplex GPT-Live relay for the standalone Quest.

    The client sends the official Live JSON protocol unchanged. This service
    contributes only server-side OpenAI authentication and never inspects or
    logs transcript/audio payloads.
    """
    global QUEST_OPENAI_RELAY_ACTIVE

    if not _quest_websocket_is_authorized(websocket):
        await websocket.close(code=1008)
        return
    if not OPENAI_API_KEY:
        await websocket.accept()
        await websocket.send_json({
            "type": "error",
            "error": {"message": "OpenAI Live is not configured on the SYNIAL backend."},
        })
        await websocket.close(code=1011)
        return

    async with QUEST_OPENAI_RELAY_LOCK:
        if QUEST_OPENAI_RELAY_ACTIVE:
            await websocket.accept()
            await websocket.send_json({
                "type": "error",
                "error": {"message": "The Quest OpenAI Live relay is already in use."},
            })
            await websocket.close(code=1013)
            return
        QUEST_OPENAI_RELAY_ACTIVE = True

    client_messages = 0
    upstream_messages = 0
    try:
        await websocket.accept()
        async with websockets.connect(
            OPENAI_LIVE_WEBSOCKET_URL,
            extra_headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            origin=None,
            max_size=8 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
        ) as upstream:
            logger.info("[quest-openai] live relay connected")

            async def quest_to_openai() -> None:
                nonlocal client_messages
                while True:
                    message = await websocket.receive_text()
                    client_messages += 1
                    await upstream.send(message)

            async def openai_to_quest() -> None:
                nonlocal upstream_messages
                async for message in upstream:
                    upstream_messages += 1
                    if isinstance(message, bytes):
                        await websocket.send_bytes(message)
                    else:
                        await websocket.send_text(message)

            tasks = {
                asyncio.create_task(quest_to_openai()),
                asyncio.create_task(openai_to_quest()),
            }
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                task.result()
    except WebSocketDisconnect:
        pass
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as exc:
        logger.warning("[quest-openai] relay failed: %s", type(exc).__name__)
        try:
            await websocket.send_json({
                "type": "error",
                "error": {"message": "The OpenAI Live relay connection failed."},
            })
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        async with QUEST_OPENAI_RELAY_LOCK:
            QUEST_OPENAI_RELAY_ACTIVE = False
        logger.info(
            "[quest-openai] live relay closed client_messages=%d upstream_messages=%d",
            client_messages,
            upstream_messages,
        )


@app.get("/api/quest/devices/{device_id}/status")
async def quest_device_status(device_id: str, request: Request):
    _require_admin_api_key(request)
    device = _quest_device_or_404(device_id)
    async with QUEST_LOCK:
        return _quest_public_status(device)


@app.post("/api/quest/launch")
async def quest_launch(request: Request):
    _require_admin_api_key(request)
    body = await request.json()
    launch_id = _clean_quest_text(body.get("launch_id"), 64)
    device_id = _clean_quest_text(body.get("device_id"), 64)
    experience_type = _clean_quest_text(body.get("experience_type"), 48).lower()
    try:
        uuid.UUID(launch_id)
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(status_code=400, detail="launch_id invalide.")
    if device_id != QUEST_DEVICE_ID:
        raise HTTPException(status_code=404, detail="Casque Quest inconnu.")
    if experience_type != "journaliste":
        raise HTTPException(status_code=400, detail="Expérience Quest non prise en charge.")
    try:
        number_session = int(body.get("number_session") or 0)
    except (TypeError, ValueError):
        number_session = 0
    if number_session <= 0:
        raise HTTPException(status_code=400, detail="number_session invalide.")

    command = {
        "launch_id": launch_id,
        "number_session": number_session,
        "device_id": device_id,
        "experience_type": experience_type,
        "player_name": _clean_quest_text(body.get("player_name"), 160, "Joueur"),
        "player_role": _clean_quest_text(body.get("player_role"), 240),
        "ai_role": _clean_quest_text(body.get("ai_role"), 64, "journaliste"),
        "system_instruction": _clean_quest_text(body.get("system_instruction"), 30000),
        "scenario_context": _clean_quest_text(body.get("scenario_context"), 300000),
    }

    async with QUEST_LOCK:
        existing = QUEST_LAUNCHES.get(launch_id)
        if existing:
            if existing.device_id != device_id or existing.command != command:
                raise HTTPException(status_code=409, detail="launch_id déjà utilisé avec une autre commande.")
            return {
                "ok": True,
                "launch_id": launch_id,
                "state": existing.state,
                "message": "Simulation envoyée",
                "duplicate": True,
            }

        device = _quest_device_or_404(device_id)
        if not _quest_is_available(device):
            raise HTTPException(status_code=409, detail="Casque Quest introuvable ou non disponible.")

        launch = QuestLaunchState(launch_id=launch_id, device_id=device_id, command=command)
        QUEST_LAUNCHES[launch_id] = launch
        device.pending_launch_id = launch_id
        device.waiting = True

    logger.info("[quest] launch queued id=%s device=%s session=%s", launch_id, device_id, number_session)
    return Response(
        content=json.dumps({
            "ok": True,
            "launch_id": launch_id,
            "state": "pending",
            "message": "Simulation envoyée",
        }),
        status_code=202,
        media_type="application/json",
    )


@app.get("/api/quest/launch/{launch_id}")
async def quest_launch_status(launch_id: str, request: Request):
    _require_admin_api_key(request)
    async with QUEST_LOCK:
        launch = QUEST_LAUNCHES.get(launch_id)
        if not launch:
            raise HTTPException(status_code=404, detail="Simulation Quest inconnue.")
        return {
            "ok": True,
            "launch_id": launch.launch_id,
            "device_id": launch.device_id,
            "state": launch.state,
            "started": launch.state in {"started", "completed"},
            "completed": launch.state == "completed",
        }


@app.post("/api/quest/launch/{launch_id}/started")
async def quest_launch_started(launch_id: str, request: Request):
    _require_quest_token(request)
    body = await request.json()
    device_id = _clean_quest_text(body.get("device_id"), 64)
    async with QUEST_LOCK:
        launch = QUEST_LAUNCHES.get(launch_id)
        if not launch or launch.device_id != device_id:
            raise HTTPException(status_code=404, detail="Simulation Quest inconnue.")
        if launch.state == "completed":
            return {"ok": True, "launch_id": launch_id, "state": launch.state, "duplicate": True}
        if launch.state not in {"claimed", "started"}:
            raise HTTPException(status_code=409, detail="La simulation n'a pas été récupérée par ce casque.")
        launch.state = "started"
        launch.started_at = launch.started_at or time.time()
    logger.info("[quest] launch started id=%s device=%s", launch_id, device_id)
    return {"ok": True, "launch_id": launch_id, "state": "started"}


@app.post("/api/quest/launch/{launch_id}/transcript")
async def quest_launch_transcript(launch_id: str, request: Request):
    _require_quest_token(request)
    body = await request.json()
    device_id = _clean_quest_text(body.get("device_id"), 64)
    try:
        uuid.UUID(launch_id)
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(status_code=400, detail="launch_id invalide.")
    if device_id != QUEST_DEVICE_ID:
        raise HTTPException(status_code=404, detail="Casque Quest inconnu.")
    turns = body.get("turns")
    if not isinstance(turns, list):
        raise HTTPException(status_code=400, detail="Transcript invalide.")

    async with QUEST_LOCK:
        launch = QUEST_LAUNCHES.get(launch_id)
        if launch:
            if launch.device_id != device_id:
                raise HTTPException(status_code=404, detail="Simulation Quest inconnue.")
            if launch.transcript_forwarded:
                return {"ok": True, "ack": True, "duplicate": True, "launch_id": launch_id}
            if launch.transcript_forwarding:
                raise HTTPException(status_code=409, detail="Transcript en cours de traitement, réessayez.")
            if launch.state not in {"claimed", "started"}:
                raise HTTPException(status_code=409, detail="État de simulation incompatible avec le transcript.")
            if int(body.get("number_session") or 0) != int(launch.command["number_session"]):
                raise HTTPException(status_code=400, detail="number_session ne correspond pas à la simulation.")
            forward_payload = {
                "launch_id": launch_id,
                "number_session": launch.command["number_session"],
                "experience_type": launch.command["experience_type"],
                "player_name": launch.command["player_name"],
                "player_role": launch.command["player_role"],
                "ai_role": launch.command["ai_role"],
                "turns": turns[:500],
            }
            launch.transcript_forwarding = True
        else:
            # Un ACK peut avoir été perdu juste avant un redémarrage Render.
            # Le casque authentifié renvoie alors son payload local; Flask garde
            # l'idempotence durable grâce à la clé unique launch_id.
            try:
                number_session = int(body.get("number_session") or 0)
            except (TypeError, ValueError):
                number_session = 0
            if number_session <= 0:
                raise HTTPException(status_code=400, detail="number_session invalide.")
            experience_type = _clean_quest_text(body.get("experience_type"), 48).lower()
            if experience_type != "journaliste":
                raise HTTPException(status_code=400, detail="Expérience Quest non prise en charge.")
            forward_payload = {
                "launch_id": launch_id,
                "number_session": number_session,
                "experience_type": experience_type,
                "player_name": _clean_quest_text(body.get("player_name"), 160, "Joueur"),
                "player_role": _clean_quest_text(body.get("player_role"), 240),
                "ai_role": _clean_quest_text(body.get("ai_role"), 64, "journaliste"),
                "turns": turns[:500],
            }

    success, detail = await asyncio.to_thread(_forward_quest_transcript_to_flask, forward_payload)

    async with QUEST_LOCK:
        launch = QUEST_LAUNCHES.get(launch_id)
        if launch:
            launch.transcript_forwarding = False
        if not success:
            raise HTTPException(status_code=502, detail=detail)
        if launch:
            launch.transcript_forwarded = True
            launch.state = "completed"
            launch.completed_at = time.time()
        device = QUEST_DEVICES.get(device_id)
        if device and device.active_launch_id == launch_id:
            device.active_launch_id = None
            device.waiting = False

    logger.info("[quest] transcript acknowledged id=%s device=%s", launch_id, device_id)
    return {"ok": True, "ack": True, "launch_id": launch_id}


@app.post("/api/prepare_call")
async def api_prepare_call(request: Request):
    if not VOICE_ADMIN_API_KEY:
        raise HTTPException(status_code=500, detail="VOICE_ADMIN_API_KEY must be set")

    api_key = request.headers.get("X-API-Key", "").strip()
    if api_key != VOICE_ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        _require_env("PUBLIC_BASE_URL", PUBLIC_BASE_URL)
        if VOICE_CALL_PROVIDER != "openai":
            raise ValueError("VOICE_CALL_PROVIDER must be openai for SYNIAL phone calls")
        if OPENAI_PHONE_CODEC not in openai_live_phone.AUDIO_FORMATS:
            raise ValueError("OPENAI_PHONE_CODEC must be pcm or pcmu")
        _require_env("OPENAI_API_KEY", OPENAI_API_KEY)
        _require_env("TWILIO_FROM_NUMBER", TWILIO_FROM_NUMBER)
        _require_env("TWILIO_ACCOUNT_SID", TWILIO_ACCOUNT_SID)
        _require_env("TWILIO_AUTH_TOKEN", TWILIO_AUTH_TOKEN)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    body = await request.json()
    to_number = str(body.get("to", "")).strip()
    player_name = str(body.get("player_name", "Joueur")).strip() or "Joueur"
    number_session = int(body.get("number_session") or 0)
    history_text = str(body.get("history_text") or "").strip()
    player_role = str(body.get("player_role") or "").strip()
    initiated_by = str(body.get("initiated_by") or "admin").strip().lower()
    if initiated_by not in {"admin", "player"}:
        initiated_by = "admin"
    ai_role = normalize_ai_role(str(body.get("ai_role") or "journaliste"))
    if ai_role not in openai_live_phone.ROLES:
        ai_role = "journaliste"

    if not to_number or not validate_e164(to_number):
        raise HTTPException(status_code=400, detail="Invalid 'to' (expected E.164 like +336...) ")

    if ALLOWED_TO_PREFIXES and not any(to_number.startswith(p) for p in ALLOWED_TO_PREFIXES):
        raise HTTPException(status_code=403, detail="This destination number is not allowed")

    call_id = uuid.uuid4().hex
    t0 = time.time()

    try:
        config = openai_live_phone.phone_session(ai_role, initiated_by, player_role, history_text, OPENAI_PHONE_CODEC)
        cm = openai_live_phone.connect(OPENAI_API_KEY, config)
        session = await cm.__aenter__()
    except Exception as e:
        logger.error("[%s] voice connection failed provider=%s error_type=%s", call_id, VOICE_CALL_PROVIDER, type(e).__name__)
        raise HTTPException(status_code=502, detail="Le moteur vocal est indisponible. Aucun appel n'a été lancé.") from None

    pc = PreparedCall(
        number_session=number_session,
        call_id=call_id,
        created_at=t0,
        expires_at=t0 + PREPARED_SESSION_TTL_SECONDS,
        player_name=player_name,
        initiated_by=initiated_by,
        ai_role=ai_role,
        codec=OPENAI_PHONE_CODEC,
        stream_token=uuid.uuid4().hex,
        upstream_cm=cm,
        upstream=session,
        state="ready",
    )

    async with PREPARED_LOCK:
        PREPARED[call_id] = pc

    pc.cleanup_task = asyncio.create_task(_cleanup_expired_call(call_id))

    try:
        stream_ws_url = _to_wss_url(PUBLIC_BASE_URL, "/twilio/stream")
        twiml = build_twiml_stream(
            stream_ws_url,
            custom_parameters={
                "call_id": call_id,
                "stream_token": pc.stream_token,
                "ai_role": ai_role,
                "number_session": str(number_session),
                "initiated_by": initiated_by,
                "player_name": player_name[:80],
            }
        )

        def _do_call() -> str:
            client = _twilio_client()
            call = client.calls.create(
                to=to_number,
                from_=TWILIO_FROM_NUMBER,
                twiml=twiml,
                machine_detection="Enable",
                async_amd=True,
                async_amd_status_callback=f"{PUBLIC_BASE_URL}/twilio/amd_callback",
                async_amd_status_callback_method="POST",
            )
            return call.sid

        twilio_call_sid = await asyncio.to_thread(_do_call)

        async with pc.lock:
            pc.twilio_call_sid = twilio_call_sid
            if pc.state == "ready":
                pc.state = "calling"

        prep_ms = int((time.time() - t0) * 1000)
        logger.info("[%s] prepared %s + started Twilio callSid=%s in %sms", call_id, VOICE_CALL_PROVIDER, twilio_call_sid, prep_ms)

        return {"ok": True, "call_id": call_id, "call_sid": twilio_call_sid, "prep_ms": prep_ms}

    except Exception as e:
        logger.error("[%s] Twilio call creation failed: %s", call_id, type(e).__name__)
        await _close_prepared_call(call_id, reason="twilio_create_failed")
        raise HTTPException(status_code=502, detail="Twilio n'a pas pu lancer l'appel.") from None


@app.api_route("/twilio/voice", methods=["GET", "POST"])
async def twilio_voice():
    raise HTTPException(status_code=404, detail="Utilisez le lancement d’appel SYNIAL.")


@app.api_route("/twilio/amd_callback", methods=["GET", "POST"])
async def twilio_amd_callback(request: Request):
    """Twilio AMD (Answering Machine Detection) callback."""
    form = await request.form()
    call_sid = str(form.get("CallSid", ""))
    answered_by = str(form.get("AnsweredBy", ""))
    logger.info("[amd] callSid=%s answeredBy=%s", call_sid, answered_by)

    # Si répondeur -> raccrocher
    if answered_by in ("machine_start", "machine_end_beep", "machine_end_silence", "machine_end_other", "fax"):
        try:
            await asyncio.to_thread(_twilio_hangup_call, call_sid)
            logger.info("[amd] hangup machine call callSid=%s", call_sid)
        except Exception as e:
            logger.warning("[amd] hangup failed: %s", e)

    return Response(content="", status_code=200)

async def receive_twilio_start(websocket):
    while True:
        event = await websocket.receive_json()
        if event.get("event") == "start":
            return event


@app.websocket("/twilio/stream")
async def twilio_stream(websocket: WebSocket):
    await websocket.accept()
    call_id = "unattached"
    prepared = None
    ctx = None
    recording = None
    queue = asyncio.Queue(maxsize=OUT_QUEUE_MAX_FRAMES)
    stopping = asyncio.Event()
    deltas = []
    io_tasks = []
    live_receiver = None
    close_code = 1000

    try:
        event = await asyncio.wait_for(receive_twilio_start(websocket), timeout=12)
        start = event.get("start") or {}
        parameters = start.get("customParameters") or {}
        media_format = start.get("mediaFormat") or {}
        if media_format and (
            media_format.get("encoding"), media_format.get("sampleRate"), media_format.get("channels"),
        ) != ("audio/x-mulaw", TWILIO_RATE_HZ, 1):
            close_code = 1003
            return
        stream_sid = event.get("streamSid") or start.get("streamSid")
        call_sid = start.get("callSid") or event.get("callSid")
        candidate_id = str(parameters.get("call_id") or "")
        supplied_token = str(parameters.get("stream_token") or "").encode()
        async with PREPARED_LOCK:
            candidate = PREPARED.get(candidate_id)
            if candidate is not None and stream_sid and call_sid:
                async with candidate.lock:
                    if (candidate.state in {"ready", "calling"} and candidate.expires_at > time.time()
                            and hmac.compare_digest(candidate.stream_token.encode(), supplied_token)
                            and (not candidate.twilio_call_sid or candidate.twilio_call_sid == call_sid)):
                        candidate.twilio_call_sid = call_sid
                        candidate.twilio_stream_sid = stream_sid
                        candidate.state = "in_call"
                        prepared = candidate
        if prepared is None:
            close_code = 1008
            return
        call_id = prepared.call_id
        if prepared.cleanup_task:
            prepared.cleanup_task.cancel()
        ctx = StreamContext(stream_sid, call_sid)
        if ENABLE_TRANSCRIPTIONS and prepared.number_session:
            recording = CallRecording(VOICE_RECORD_MAX_SECONDS)
        converter = AudioConverter(prepared.codec)
        session = prepared.upstream
        io_tasks = [
            asyncio.create_task(twilio_receiver_loop(websocket, ctx, session, converter, recording, prepared.initiated_by)),
            asyncio.create_task(twilio_sender_loop(websocket, ctx, queue, recording)),
        ]
        live_receiver = asyncio.create_task(openai_phone_receiver_loop(session, converter, queue, deltas, stopping, ctx))
        done, _ = await asyncio.wait([*io_tasks, live_receiver], return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("[%s] phone stream ended error_type=%s", call_id, type(exc).__name__)
    finally:
        stopping.set()
        if ctx is not None:
            ctx.ended_at = time.monotonic()
        for task in io_tasks:
            task.cancel()
        if prepared is not None:
            state = websocket.app.state
            finalizer = asyncio.create_task(finish_phone_call(
                state.phone_postcall, prepared, ctx, recording, deltas, queue, io_tasks, live_receiver,
            ))
            state.phone_finalizers.add(finalizer)
            finalizer.add_done_callback(state.phone_finalizers.discard)
        with suppress(RuntimeError, WebSocketDisconnect):
            await websocket.close(code=close_code)


async def finish_phone_call(postcall, prepared, ctx, recording, deltas, queue, io_tasks, live_receiver):
    call_id = prepared.call_id
    finalized = False
    await asyncio.gather(*io_tasks, return_exceptions=True)
    try:
        if live_receiver is not None:
            if not live_receiver.done():
                await asyncio.wait_for(openai_live_phone.send(prepared.upstream, "session.close"), timeout=2)
            finalized = await asyncio.wait_for(live_receiver, timeout=OPENAI_PHONE_CLOSE_TIMEOUT)
    except Exception as exc:
        logger.warning("[%s] voice finalization incomplete error_type=%s", call_id, type(exc).__name__)
    finally:
        if live_receiver is not None and not live_receiver.done():
            live_receiver.cancel()
            await asyncio.gather(live_receiver, return_exceptions=True)
        await _close_prepared_call(call_id, reason="call_end")
    if ctx is not None:
        logger.info(
            "[%s] phone ended codec=%s first_audio_ms=%s max_queue_frames=%s max_mark_delay_ms=%s finalized=%s",
            call_id, prepared.codec, ctx.first_output_ms, ctx.max_queue_frames, ctx.max_mark_delay_ms, finalized,
        )
    if recording is not None and ctx is not None:
        try:
            await enqueue_phone_record(postcall, prepared, ctx, recording, deltas, finalized, queue.empty())
        except Exception as exc:
            logger.error("[%s] post-call capture failed error_type=%s", call_id, type(exc).__name__)


async def enqueue_phone_record(postcall, prepared, ctx, recording, deltas, finalized, queue_empty):
    turns = openai_live_phone.transcript_turns(deltas)
    # Use the recorded inbound timeline plus Twilio playback confirmations.
    # With unacknowledged frames, estimate the playable prefix at disconnect.
    end = max(ctx.inbound_end, ctx.played_end)
    if ctx.last_media_at is not None:
        elapsed_after_input = max(0, round((ctx.ended_at - ctx.last_media_at) * TWILIO_RATE_HZ))
        end = max(end, min(ctx.sent_end, ctx.inbound_end + elapsed_after_input))
    audible = await asyncio.to_thread(recording.audible_roles, end)
    transcript_roles = {turn["role"] for turn in turns}
    fallback = bool(audible and (
        not finalized or not audible.issubset(transcript_roles) or not queue_empty or ctx.playback_uncertain
        or ctx.sent_end > end + TWILIO_FRAME_BYTES
        or ("assistant" in transcript_roles and not ctx.sent_end)
    ))
    if not audible and not ctx.sent_end:
        turns = [turn for turn in turns if turn["role"] != "assistant"]
    payload = {
        "number_session": prepared.number_session, "call_sid": ctx.call_sid, "turns": turns,
        "player_name": prepared.player_name, "initiated_by": prepared.initiated_by, "ai_role": prepared.ai_role,
        "transcript_source": "live" if turns else "no_exchange",
    }
    if fallback and recording.truncated:
        payload["transcript_recording_truncated"] = True
        logger.warning("[%s] phone recovery recording reached its duration limit", prepared.call_id)
    for attempt in range(3):
        try:
            await postcall.enqueue(payload, recording if fallback else None, audible, end)
            return
        except (OSError, sqlite3.Error) as exc:
            if attempt == 2:
                logger.error("[%s] post-call persistence failed error_type=%s", prepared.call_id, type(exc).__name__)
            else:
                await asyncio.sleep(.25 * (attempt + 1))


if __name__ == "__main__":
    import uvicorn

    if not PUBLIC_BASE_URL:
        logger.warning("PUBLIC_BASE_URL is empty. Twilio streaming will NOT work until set.")

    logger.info("Starting voice service on 0.0.0.0:%s", int(os.getenv("PORT", "8001")))
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8001")), log_level="info")
