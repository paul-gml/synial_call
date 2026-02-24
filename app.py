# -*- coding: utf-8 -*-
"""
voice_journalist_service.py

Service voix "Journaliste" : Twilio (Media Streams) <-> Gemini Live (Vertex AI)

- /api/prepare_call : (protégé) préchauffe Gemini Live, puis lance un appel Twilio sortant
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
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import audioop  # stdlib (déprécié dans le futur, mais OK en 3.10-3.12)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response, HTTPException

from twilio.rest import Client as TwilioClient
from google import genai
from google.genai import types
import requests


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

# --- Google / Vertex AI Live ---
# Option: mettre un JSON de service account directement dans une variable.
GCP_CREDS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "").strip()
if GCP_CREDS_JSON and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    creds_path = "/tmp/google_creds.json"
    try:
        with open(creds_path, "w", encoding="utf-8") as f:
            f.write(GCP_CREDS_JSON)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
    except Exception:
        pass

MAIN_APP_BASE_URL = os.getenv("MAIN_APP_BASE_URL", "").strip().rstrip("/")  # ex: https://synial.onrender.com
INTERNAL_APP_TOKEN = os.getenv("INTERNAL_APP_TOKEN", "").strip()           # = INTERNAL_VOICE_TOKEN côté Flask

PROJECT_ID = (os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT") or "").strip()
LOCATION = (os.getenv("GOOGLE_CLOUD_LOCATION") or os.getenv("GOOGLE_CLOUD_REGION") or "us-central1").strip()

MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-live-2.5-flash-native-audio").strip()
VOICE_NAME = os.getenv("GEMINI_VOICE_NAME", "Kore").strip()

ENABLE_TRANSCRIPTIONS = os.getenv("ENABLE_TRANSCRIPTIONS", "true").lower() in ("1", "true", "yes", "y")

# --- Audio ---
TWILIO_RATE_HZ = 8000
GEMINI_IN_RATE_HZ = 16000
GEMINI_OUT_RATE_HZ_DEFAULT = 24000

# 20ms µ-law @ 8kHz => 160 bytes
TWILIO_FRAME_BYTES = 160
OUT_QUEUE_MAX_FRAMES = int(os.getenv("OUT_QUEUE_MAX_FRAMES", "250"))

# --- Latence / cleanup ---
PREPARED_SESSION_TTL_SECONDS = int(os.getenv("PREPARED_SESSION_TTL_SECONDS", "90"))

# --- Auto hangup ---
AUTO_HANGUP_ON_GOODBYE = os.getenv("AUTO_HANGUP_ON_GOODBYE", "true").lower() in ("1", "true", "yes", "y")
HANGUP_DELAY_SECONDS = float(os.getenv("HANGUP_DELAY_SECONDS", "2"))
GOODBYE_REGEX = re.compile(r"\b(au\s+revoir|bonne\s+journ[eé]e|bye|à\s+bient[ôo]t|ciao)\b", re.IGNORECASE)

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


def parse_rate_from_mime(mime_type: Optional[str]) -> Optional[int]:
    # Example: "audio/pcm;rate=24000"
    if not mime_type:
        return None
    mt = str(mime_type)
    if "rate=" not in mt:
        return None
    try:
        tail = mt.split("rate=", 1)[1]
        num = ""
        for ch in tail:
            if ch.isdigit():
                num += ch
            else:
                break
        return int(num) if num else None
    except Exception:
        return None


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
    """Streaming-safe conversions between Twilio mulaw(8k) and Gemini PCM."""

    def __init__(self) -> None:
        self._in_ratecv_state = None
        self._out_ratecv_state = None
        self._out_ulaw_buffer = bytearray()

    def reset(self) -> None:
        self._in_ratecv_state = None
        self._out_ratecv_state = None
        self._out_ulaw_buffer.clear()

    def twilio_ulaw8k_to_gemini_pcm16k(self, ulaw: bytes) -> bytes:
        pcm8k = audioop.ulaw2lin(ulaw, 2)
        pcm16k, self._in_ratecv_state = audioop.ratecv(
            pcm8k, 2, 1, TWILIO_RATE_HZ, GEMINI_IN_RATE_HZ, self._in_ratecv_state
        )
        return pcm16k

    def gemini_pcm_to_twilio_ulaw_frames(self, pcm: bytes, pcm_rate_hz: int) -> list[bytes]:
        pcm8k, self._out_ratecv_state = audioop.ratecv(
            pcm, 2, 1, pcm_rate_hz, TWILIO_RATE_HZ, self._out_ratecv_state
        )
        ulaw = audioop.lin2ulaw(pcm8k, 2)

        self._out_ulaw_buffer.extend(ulaw)
        frames: list[bytes] = []
        while len(self._out_ulaw_buffer) >= TWILIO_FRAME_BYTES:
            frames.append(bytes(self._out_ulaw_buffer[:TWILIO_FRAME_BYTES]))
            del self._out_ulaw_buffer[:TWILIO_FRAME_BYTES]
        return frames

    def flush_output(self) -> None:
        self._out_ulaw_buffer.clear()
        self._out_ratecv_state = None


async def ws_send_json(websocket: WebSocket, send_lock: asyncio.Lock, obj: Dict[str, Any]) -> None:
    payload = json.dumps(obj, separators=(",", ":"))
    async with send_lock:
        await websocket.send_text(payload)


async def drain_queue(q: asyncio.Queue) -> None:
    try:
        while True:
            q.get_nowait()
            q.task_done()
    except asyncio.QueueEmpty:
        return


# ============================================================
# 5) Gemini config (journaliste)
# ============================================================

BASE_SYSTEM_TEMPLATE = (
    "Tu es une journaliste d'actualité. "
    "Tu appelles {player_name} au téléphone pour lui poser des questions dans un serious game de gestion de crise. "
    "Tu es professionnelle, directe, et tu poses des questions courtes, une par une. "
    "RÉPONDS EN FRANÇAIS, de façon naturelle et concise (1 à 3 phrases). "
    "IMPORTANT: N'initie pas la conversation tant que tu n'as pas entendu l'interlocuteur parler (ex: 'Allô'). "
    "Si l'interlocuteur veut raccrocher, réponds 'Merci, au revoir.' "
)


def build_live_config(system_instruction_text: str) -> types.LiveConnectConfig:
    kwargs: Dict[str, Any] = dict(
        response_modalities=["AUDIO"],
        system_instruction=types.Content(
            role="system", parts=[types.Part.from_text(text=system_instruction_text)]
        ),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE_NAME)
            )
        ),
    )
    if ENABLE_TRANSCRIPTIONS:
        kwargs["input_audio_transcription"] = {}
        kwargs["output_audio_transcription"] = {}
    return types.LiveConnectConfig(**kwargs)


# ============================================================
# 6) Prepared sessions store (in-memory)
# ============================================================

@dataclass
class PreparedCall:
    call_id: str
    created_at: float
    expires_at: float

    to_number: str
    player_name: str
    system_instruction: str

    gemini_client: Any
    gemini_cm: Any
    gemini_session: Any

    twilio_call_sid: Optional[str] = None
    twilio_stream_sid: Optional[str] = None

    state: str = "ready"  # ready -> calling -> in_call -> ended
    cleanup_task: Optional[asyncio.Task] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    number_session: int = 0
    transcript_turns: list = field(default_factory=list)
    transcript_sent: bool = False
    last_user_tr: str = ""
    last_assistant_tr: str = ""


PREPARED: Dict[str, PreparedCall] = {}
PREPARED_LOCK = asyncio.Lock()


async def _close_prepared_call(call_id: str, reason: str) -> None:
    async with PREPARED_LOCK:
        pc = PREPARED.pop(call_id, None)

    if not pc:
        return

    async with pc.lock:
        pc.state = "ended"

    if pc.cleanup_task:
        try:
            pc.cleanup_task.cancel()
        except Exception:
            pass

    try:
        await pc.gemini_cm.__aexit__(None, None, None)
    except Exception as e:
        logger.warning("[%s] failed to close Gemini session (%s): %s", call_id, reason, e)


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
# 7) Twilio <-> Gemini bridge loops
# ============================================================

@dataclass
class StreamContext:
    stream_sid: Optional[str] = None
    call_sid: Optional[str] = None
    custom_parameters: Dict[str, str] = field(default_factory=dict)
    started: asyncio.Event = field(default_factory=asyncio.Event)


async def gemini_receiver_loop(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    session: Any,
    ctx: StreamContext,
    stop_event: asyncio.Event,
    converter: AudioConverter,
    out_frames_q: asyncio.Queue,
    call_id: str,
    prepared_call: Optional[PreparedCall]) -> None:
    twilio_call_sid = prepared_call.twilio_call_sid if prepared_call else None
    async def _hangup_later(delay_s: float) -> None:
        if not (AUTO_HANGUP_ON_GOODBYE and twilio_call_sid):
            return
        await asyncio.sleep(max(0.0, delay_s))
        try:
            await asyncio.to_thread(_twilio_hangup_call, twilio_call_sid)
            logger.info("[%s] auto-hangup done callSid=%s", call_id, twilio_call_sid)
        except Exception as e:
            logger.warning("[%s] auto-hangup failed: %s", call_id, e)

    hangup_scheduled = False

    try:
        while not stop_event.is_set():
            async for message in session.receive():
                if ENABLE_TRANSCRIPTIONS:
                    in_tr = getattr(message, "input_transcription", None)
                    if in_tr is not None and getattr(in_tr, "text", None):
                        logger.info("[%s][user] %s", call_id, in_tr.text)

                    out_tr = getattr(message, "output_transcription", None)
                    if prepared_call is not None:
                        if in_tr is not None and getattr(in_tr, "text", None):
                            txt = in_tr.text.strip()
                            if txt and txt != prepared_call.last_user_tr:
                                prepared_call.last_user_tr = txt
                                prepared_call.transcript_turns.append({"role": "user", "text": txt})

                        if out_tr is not None and getattr(out_tr, "text", None):
                            txt = out_tr.text.strip()
                            if txt and txt != prepared_call.last_assistant_tr:
                                prepared_call.last_assistant_tr = txt
                                prepared_call.transcript_turns.append({"role": "assistant", "text": txt})
                    if out_tr is not None and getattr(out_tr, "text", None):
                        out_text = out_tr.text
                        logger.info("[%s][assistant] %s", call_id, out_text)

                        if AUTO_HANGUP_ON_GOODBYE and (not hangup_scheduled) and GOODBYE_REGEX.search(out_text or ""):
                            hangup_scheduled = True
                            asyncio.create_task(_hangup_later(HANGUP_DELAY_SECONDS))

                server_content = getattr(message, "server_content", None)
                if not server_content:
                    continue

                if getattr(server_content, "interrupted", False):
                    if ctx.stream_sid:
                        await ws_send_json(websocket, send_lock, {"event": "clear", "streamSid": ctx.stream_sid})
                    converter.flush_output()
                    await drain_queue(out_frames_q)
                    continue

                model_turn = getattr(server_content, "model_turn", None)
                if not model_turn or not getattr(model_turn, "parts", None):
                    continue

                for part in model_turn.parts:
                    inline = getattr(part, "inline_data", None)
                    if not inline or not getattr(inline, "data", None):
                        continue

                    pcm_bytes = inline.data
                    pcm_rate = parse_rate_from_mime(getattr(inline, "mime_type", None)) or GEMINI_OUT_RATE_HZ_DEFAULT

                    frames = converter.gemini_pcm_to_twilio_ulaw_frames(pcm_bytes, pcm_rate_hz=pcm_rate)
                    for fr in frames:
                        if out_frames_q.full():
                            try:
                                out_frames_q.get_nowait()
                                out_frames_q.task_done()
                            except asyncio.QueueEmpty:
                                pass
                        await out_frames_q.put(fr)

    except Exception as e:
        logger.error("[%s][gemini] receiver loop error: %s", call_id, e)
        traceback.print_exc()
        stop_event.set()


async def twilio_sender_loop(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    ctx: StreamContext,
    stop_event: asyncio.Event,
    out_frames_q: asyncio.Queue,
    call_id: str,
) -> None:
    try:
        await ctx.started.wait()
        next_send_time = time.monotonic()

        while not stop_event.is_set():
            frame = await out_frames_q.get()

            if not ctx.stream_sid:
                out_frames_q.task_done()
                continue

            duration = len(frame) / float(TWILIO_RATE_HZ)
            now = time.monotonic()
            if next_send_time > now:
                await asyncio.sleep(next_send_time - now)

            payload_b64 = base64.b64encode(frame).decode("ascii")
            await ws_send_json(
                websocket,
                send_lock,
                {"event": "media", "streamSid": ctx.stream_sid, "media": {"payload": payload_b64}},
            )

            next_send_time += duration
            out_frames_q.task_done()

    except WebSocketDisconnect:
        logger.info("[%s][twilio] sender ws disconnected", call_id)
        stop_event.set()
    except Exception as e:
        logger.error("[%s][twilio] sender loop error: %s", call_id, e)
        traceback.print_exc()
        stop_event.set()


def _twilio_hangup_call(call_sid: str) -> None:
    client = _twilio_client()
    client.calls(call_sid).update(status="completed")


# ============================================================
# 8) FastAPI app
# ============================================================

app = FastAPI()


@app.get("/")
async def root():
    return {"ok": True, "service": "voice_journalist", "model": MODEL_ID}


@app.get("/health")
async def health():
    return {
        "ok": True,
        "project": PROJECT_ID,
        "location": LOCATION,
        "model": MODEL_ID,
        "twilio_configured": bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER),
        "public_base_url": PUBLIC_BASE_URL,
        "prepared_sessions": len(PREPARED),
    }


@app.post("/api/prepare_call")
async def api_prepare_call(request: Request):
    if not VOICE_ADMIN_API_KEY:
        raise HTTPException(status_code=500, detail="VOICE_ADMIN_API_KEY must be set")

    api_key = request.headers.get("X-API-Key", "").strip()
    if api_key != VOICE_ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        _require_env("PUBLIC_BASE_URL", PUBLIC_BASE_URL)
        _require_env("GOOGLE_CLOUD_PROJECT", PROJECT_ID)
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
    if len(history_text) > 8000:
        history_text = history_text[-8000:]

    if not to_number or not validate_e164(to_number):
        raise HTTPException(status_code=400, detail="Invalid 'to' (expected E.164 like +336...) ")

    if ALLOWED_TO_PREFIXES and not any(to_number.startswith(p) for p in ALLOWED_TO_PREFIXES):
        raise HTTPException(status_code=403, detail="This destination number is not allowed")

    base = BASE_SYSTEM_TEMPLATE.format(player_name=player_name)
    if player_role:
        base += "\n\nINFO INTERLOCUTEUR:\n- Poste / fonction pendant la crise : " + player_role

    if history_text:
        system_instruction = (
            base
            + "\n\n--- HISTORIQUE RECENT (chat) ---\n"
            + history_text
            + "\n--- FIN HISTORIQUE ---\n"
            + "Consigne: utilise cet historique pour contextualiser tes questions."
        )
    else:
        system_instruction = base
    config = build_live_config(system_instruction)

    call_id = uuid.uuid4().hex
    t0 = time.time()

    try:
        gemini_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
        cm = gemini_client.aio.live.connect(model=MODEL_ID, config=config)
        session = await cm.__aenter__()
    except Exception as e:
        logger.error("[%s] failed to connect Gemini Live: %s", call_id, e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Gemini connect failed: {e}")

    pc = PreparedCall(
        number_session=number_session,
        call_id=call_id,
        created_at=t0,
        expires_at=t0 + PREPARED_SESSION_TTL_SECONDS,
        to_number=to_number,
        player_name=player_name,
        system_instruction=system_instruction,
        gemini_client=gemini_client,
        gemini_cm=cm,
        gemini_session=session,
        state="ready",
    )

    async with PREPARED_LOCK:
        PREPARED[call_id] = pc

    pc.cleanup_task = asyncio.create_task(_cleanup_expired_call(call_id))

    try:
        stream_ws_url = _to_wss_url(PUBLIC_BASE_URL, "/twilio/stream")
        twiml = build_twiml_stream(
        stream_ws_url,
        custom_parameters={"call_id": call_id, "role": "journalist", "number_session": str(number_session)})

        def _do_call() -> str:
            client = _twilio_client()
            call = client.calls.create(to=to_number, from_=TWILIO_FROM_NUMBER, twiml=twiml)
            return call.sid

        twilio_call_sid = await asyncio.to_thread(_do_call)

        async with pc.lock:
            pc.twilio_call_sid = twilio_call_sid
            pc.state = "calling"

        prep_ms = int((time.time() - t0) * 1000)
        logger.info("[%s] prepared Gemini + started Twilio callSid=%s in %sms", call_id, twilio_call_sid, prep_ms)

        return {"ok": True, "call_id": call_id, "call_sid": twilio_call_sid, "prep_ms": prep_ms}

    except Exception as e:
        logger.error("[%s] failed to create Twilio call: %s", call_id, e)
        traceback.print_exc()
        await _close_prepared_call(call_id, reason="twilio_create_failed")
        raise HTTPException(status_code=500, detail=f"Twilio call failed: {e}")


@app.api_route("/twilio/voice", methods=["GET", "POST"])
async def twilio_voice(request: Request):
    if not PUBLIC_BASE_URL:
        raise HTTPException(status_code=500, detail="PUBLIC_BASE_URL must be set")
    stream_ws_url = _to_wss_url(PUBLIC_BASE_URL, "/twilio/stream")
    twiml = build_twiml_stream(stream_ws_url, custom_parameters={"role": "journalist"})
    return Response(content=twiml, media_type="application/xml; charset=utf-8")


@app.websocket("/twilio/stream")
async def twilio_stream(websocket: WebSocket):
    await websocket.accept()
    send_lock = asyncio.Lock()

    ctx = StreamContext()
    stop_event = asyncio.Event()
    converter = AudioConverter()
    out_frames_q: asyncio.Queue = asyncio.Queue(maxsize=OUT_QUEUE_MAX_FRAMES)

    call_id = "unknown"
    prepared: Optional[PreparedCall] = None

    try:
        # 1) Wait for start to get call_id
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "connected":
                continue

            if event == "start":
                start = msg.get("start", {}) or {}
                ctx.stream_sid = msg.get("streamSid") or start.get("streamSid")
                ctx.call_sid = start.get("callSid") or msg.get("callSid")
                ctx.custom_parameters = start.get("customParameters") or {}
                call_id = str(ctx.custom_parameters.get("call_id") or "unknown")
                ctx.started.set()
                logger.info("[%s][twilio] start callSid=%s streamSid=%s", call_id, ctx.call_sid, ctx.stream_sid)
                break

        # 2) Get prepared (warm) session
        async with PREPARED_LOCK:
            prepared = PREPARED.get(call_id)

        if prepared is None:
            logger.warning("[%s] no prepared session found -> fallback connect (latency likely)", call_id)

            # ✅ Récupérer number_session depuis les customParameters Twilio
            try:
                fallback_number_session = int(ctx.custom_parameters.get("number_session") or 0)
            except Exception:
                fallback_number_session = 0

            system_instruction = BASE_SYSTEM_TEMPLATE.format(player_name="Joueur")
            config = build_live_config(system_instruction)
            gemini_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
            cm = gemini_client.aio.live.connect(model=MODEL_ID, config=config)
            session = await cm.__aenter__()

            prepared = PreparedCall(
                call_id=call_id,
                created_at=time.time(),
                expires_at=time.time() + PREPARED_SESSION_TTL_SECONDS,
                to_number="",
                player_name="Joueur",
                system_instruction=system_instruction,
                gemini_client=gemini_client,
                gemini_cm=cm,
                gemini_session=session,
                twilio_call_sid=ctx.call_sid,
                twilio_stream_sid=ctx.stream_sid,
                state="in_call",
                number_session=fallback_number_session,  # ✅ important
            )
        else:
            if prepared.cleanup_task:
                try:
                    prepared.cleanup_task.cancel()
                except Exception:
                    pass

            async with prepared.lock:
                prepared.twilio_stream_sid = ctx.stream_sid
                if ctx.call_sid:
                    prepared.twilio_call_sid = ctx.call_sid
                prepared.state = "in_call"

        session = prepared.gemini_session

        async def _twilio_receiver_after_start():
            try:
                while not stop_event.is_set():
                    text = await websocket.receive_text()
                    msg2 = json.loads(text)
                    ev = msg2.get("event")

                    if ev == "media":
                        media = msg2.get("media") or {}
                        track = media.get("track")
                        if track and track != "inbound":
                            continue
                        payload_b64 = media.get("payload")
                        if not payload_b64:
                            continue
                        ulaw = base64.b64decode(payload_b64)
                        pcm16k = converter.twilio_ulaw8k_to_gemini_pcm16k(ulaw)
                        await session.send_realtime_input(
                            audio=types.Blob(data=pcm16k, mime_type=f"audio/pcm;rate={GEMINI_IN_RATE_HZ}")
                        )
                        continue

                    if ev == "stop":
                        logger.info("[%s][twilio] stop", call_id)
                        stop_event.set()
                        return
            except WebSocketDisconnect:
                logger.info("[%s][twilio] websocket disconnected", call_id)
                stop_event.set()
            except Exception as e:
                logger.error("[%s][twilio] receiver-after-start error: %s", call_id, e)
                traceback.print_exc()
                stop_event.set()

        t_in = asyncio.create_task(_twilio_receiver_after_start())
        t_out = asyncio.create_task(
            gemini_receiver_loop(
                websocket, send_lock, session, ctx, stop_event, converter, out_frames_q, call_id, prepared
            )
        )
        t_send = asyncio.create_task(twilio_sender_loop(websocket, send_lock, ctx, stop_event, out_frames_q, call_id))

        done, pending = await asyncio.wait({t_in, t_out, t_send}, return_when=asyncio.FIRST_EXCEPTION)

        for task in done:
            exc = task.exception()
            if exc:
                raise exc

        for task in pending:
            task.cancel()

    except WebSocketDisconnect:
        logger.info("[%s][twilio] websocket disconnected (top)", call_id)
    except Exception as e:
        logger.error("[%s] twilio_stream error: %s", call_id, e)
        traceback.print_exc()
    finally:
        stop_event.set()
        converter.reset()
        try:
            if prepared and (not prepared.transcript_sent) and prepared.number_session and prepared.transcript_turns:
                await asyncio.to_thread(
                    post_transcript_to_flask,
                    prepared.number_session,
                    prepared.twilio_call_sid or (ctx.call_sid or ""),
                    prepared.transcript_turns
                )
                prepared.transcript_sent = True
        except Exception as e:
            logger.warning("[%s] transcript post failed: %s", call_id, e)
        if prepared is not None:
            async with PREPARED_LOCK:
                in_store = PREPARED.get(call_id) is prepared
            if in_store:
                await _close_prepared_call(call_id, reason="call_end")
            else:
                try:
                    await prepared.gemini_cm.__aexit__(None, None, None)
                except Exception:
                    pass

        try:
            await websocket.close()
        except Exception:
            pass


def post_transcript_to_flask(number_session: int, call_sid: str, turns: list) -> None:
    if not (MAIN_APP_BASE_URL and INTERNAL_APP_TOKEN and number_session):
        return

    url = f"{MAIN_APP_BASE_URL}/internal/voice/transcript"
    payload = {"number_session": number_session, "call_sid": call_sid, "turns": turns}

    # retry simple
    for i in range(3):
        try:
            r = requests.post(url, json=payload, headers={"X-Internal-Token": INTERNAL_APP_TOKEN}, timeout=20)
            if 200 <= r.status_code < 300:
                return
        except Exception:
            pass
        time.sleep(0.8 * (i + 1))

if __name__ == "__main__":
    import uvicorn

    if not PUBLIC_BASE_URL:
        logger.warning("PUBLIC_BASE_URL is empty. Twilio streaming will NOT work until set.")

    logger.info("Starting voice service on 0.0.0.0:%s", int(os.getenv("PORT", "8001")))
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8001")), log_level="info")