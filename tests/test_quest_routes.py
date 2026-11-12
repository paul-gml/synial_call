import importlib.util
import asyncio
import sys
import types as python_types
import uuid
from threading import Event

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


def _install_optional_dependency_stubs():
    if importlib.util.find_spec("audioop") is None:
        sys.modules["audioop"] = python_types.ModuleType("audioop")

    if importlib.util.find_spec("twilio") is None:
        twilio = python_types.ModuleType("twilio")
        rest = python_types.ModuleType("twilio.rest")
        rest.Client = object
        twilio.rest = rest
        sys.modules["twilio"] = twilio
        sys.modules["twilio.rest"] = rest

    try:
        import google as google_package
    except ImportError:
        google_package = python_types.ModuleType("google")
        sys.modules["google"] = google_package
    if importlib.util.find_spec("google.genai") is None:
        genai = python_types.ModuleType("google.genai")
        genai_types = python_types.ModuleType("google.genai.types")
        genai_types.LiveConnectConfig = object
        genai.types = genai_types
        genai.Client = object
        google_package.genai = genai
        sys.modules["google.genai"] = genai
        sys.modules["google.genai.types"] = genai_types


_install_optional_dependency_stubs()

import app as service  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_quest_state(monkeypatch):
    monkeypatch.setattr(service, "VOICE_ADMIN_API_KEY", "admin-test-key")
    monkeypatch.setattr(service, "QUEST_DEVICE_TOKEN", "quest-test-token-that-is-long-enough")
    service.QUEST_LAUNCHES.clear()
    service.QUEST_DEVICES.clear()
    service.QUEST_DEVICES[service.QUEST_DEVICE_ID] = service.QuestDeviceState(device_id=service.QUEST_DEVICE_ID)
    service.QUEST_OPENAI_RELAY_ACTIVE = False
    monkeypatch.setattr(service, "_forward_quest_transcript_to_flask", lambda _payload: (True, ""))
    yield


class _FakeOpenAIWebSocket:
    def __init__(self):
        self.received = []
        self.responses = asyncio.Queue()

    async def send(self, message):
        self.received.append(message)
        await self.responses.put('{"type":"session.started","session":{"id":"test-live"}}')

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.responses.get()


class _FakeOpenAIConnection:
    def __init__(self, upstream):
        self.upstream = upstream

    async def __aenter__(self):
        return self.upstream

    async def __aexit__(self, exc_type, exc, traceback):
        self.upstream.closed.set()
        return False


def test_quest_openai_relay_requires_auth_and_forwards_without_exposing_server_key(monkeypatch):
    client = TestClient(service.app)
    with pytest.raises(WebSocketDisconnect) as rejected:
        with client.websocket_connect("/api/quest/openai/live"):
            pass
    assert rejected.value.code == 1008

    upstream = _FakeOpenAIWebSocket()
    upstream.closed = Event()
    captured_connect = {}

    def fake_connect(url, **kwargs):
        captured_connect["url"] = url
        captured_connect["headers"] = kwargs.get("extra_headers")
        return _FakeOpenAIConnection(upstream)

    monkeypatch.setattr(service, "OPENAI_API_KEY", "server-only-openai-test-key")
    monkeypatch.setattr(service.websockets, "connect", fake_connect)
    with client.websocket_connect(
        "/api/quest/openai/live",
        headers={"X-Quest-Token": service.QUEST_DEVICE_TOKEN},
    ) as websocket:
        websocket.send_text('{"type":"session.start"}')
        assert '"session.started"' in websocket.receive_text()
        websocket.close()
        assert upstream.closed.wait(2)

    assert upstream.received == ['{"type":"session.start"}']
    assert captured_connect["url"] == service.OPENAI_LIVE_WEBSOCKET_URL
    assert captured_connect["headers"]["Authorization"] == "Bearer server-only-openai-test-key"
    assert service.QUEST_OPENAI_RELAY_ACTIVE is False


def test_quest_launch_is_claimed_once_and_transcript_is_idempotent():
    client = TestClient(service.app)
    quest_headers = {"X-Quest-Token": service.QUEST_DEVICE_TOKEN}
    admin_headers = {"X-API-Key": service.VOICE_ADMIN_API_KEY}

    unavailable = client.get("/api/quest/devices/quest-main/status", headers=admin_headers)
    assert unavailable.status_code == 200
    assert unavailable.json()["available"] is False

    heartbeat = client.post(
        "/api/quest/devices/quest-main/heartbeat",
        headers=quest_headers,
        json={"waiting": True},
    )
    assert heartbeat.status_code == 200
    assert heartbeat.json()["available"] is True

    launch_id = str(uuid.uuid4())
    command = {
        "launch_id": launch_id,
        "number_session": 1,
        "device_id": "quest-main",
        "experience_type": "journaliste",
        "player_name": "Camille Durand",
        "player_role": "Responsable communication",
        "ai_role": "journaliste",
        "system_instruction": "Instruction de test",
        "scenario_context": "Contexte de test",
    }
    queued = client.post("/api/quest/launch", headers=admin_headers, json=command)
    assert queued.status_code == 202
    duplicate_http_retry = client.post("/api/quest/launch", headers=admin_headers, json=command)
    assert duplicate_http_retry.status_code == 200
    assert duplicate_http_retry.json()["duplicate"] is True

    competing = dict(command, launch_id=str(uuid.uuid4()))
    rejected = client.post("/api/quest/launch", headers=admin_headers, json=competing)
    assert rejected.status_code == 409
    assert "non disponible" in rejected.json()["detail"]

    claimed = client.get(
        "/api/quest/devices/quest-main/poll?timeout_seconds=1",
        headers=quest_headers,
    )
    assert claimed.status_code == 200
    assert claimed.json()["command"]["launch_id"] == launch_id
    assert service.QUEST_DEVICES["quest-main"].pending_launch_id is None

    started = client.post(
        f"/api/quest/launch/{launch_id}/started",
        headers=quest_headers,
        json={"device_id": "quest-main"},
    )
    assert started.status_code == 200
    assert started.json()["state"] == "started"

    transcript = {
        "device_id": "quest-main",
        "number_session": 1,
        "turns": [
            {"role": "assistant", "text": "Première question."},
            {"role": "user", "text": "Première réponse."},
        ],
    }
    ack = client.post(f"/api/quest/launch/{launch_id}/transcript", headers=quest_headers, json=transcript)
    assert ack.status_code == 200
    assert ack.json()["ack"] is True
    duplicate_ack = client.post(f"/api/quest/launch/{launch_id}/transcript", headers=quest_headers, json=transcript)
    assert duplicate_ack.status_code == 200
    assert duplicate_ack.json()["duplicate"] is True
    assert service.QUEST_LAUNCHES[launch_id].state == "completed"


def test_saved_transcript_is_acknowledged_after_signal_call_restart(monkeypatch):
    forwarded = []
    monkeypatch.setattr(service, "_forward_quest_transcript_to_flask", lambda payload: (forwarded.append(payload) is None, ""))
    launch_id = str(uuid.uuid4())
    payload = {
        "launch_id": launch_id,
        "device_id": "quest-main",
        "number_session": 3,
        "experience_type": "journaliste",
        "player_name": "Alex Martin",
        "player_role": "Directeur de crise",
        "ai_role": "journaliste",
        "turns": [{"role": "user", "text": "Réponse conservée hors ligne."}],
    }

    response = TestClient(service.app).post(
        f"/api/quest/launch/{launch_id}/transcript",
        headers={"X-Quest-Token": service.QUEST_DEVICE_TOKEN},
        json=payload,
    )

    assert response.status_code == 200
    assert response.json()["ack"] is True
    assert forwarded[0]["launch_id"] == launch_id
    assert forwarded[0]["number_session"] == 3
