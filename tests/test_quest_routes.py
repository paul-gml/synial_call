import importlib.util
import os
import sys
import types as python_types
import uuid

import pytest
from fastapi.testclient import TestClient


os.environ["VOICE_ADMIN_API_KEY"] = "admin-test-key"
os.environ["QUEST_DEVICE_TOKEN"] = "quest-test-token-that-is-long-enough"


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
    service.QUEST_LAUNCHES.clear()
    service.QUEST_DEVICES.clear()
    service.QUEST_DEVICES[service.QUEST_DEVICE_ID] = service.QuestDeviceState(device_id=service.QUEST_DEVICE_ID)
    monkeypatch.setattr(service, "_forward_quest_transcript_to_flask", lambda _payload: (True, ""))
    yield


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
