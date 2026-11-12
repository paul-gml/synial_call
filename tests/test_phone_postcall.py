import asyncio
import audioop
import io
import json
import sqlite3
import wave
from contextlib import closing, nullcontext
from types import SimpleNamespace

import pytest

import app as service
import phone_postcall as postcall
from openai_live_phone import transcript_turns


def test_transcript_keeps_independent_overlapping_turns_and_deduplicates():
    def delta(role, start, end, text, event_id):
        return role, dict(start_ms=start, end_ms=end, delta=text, event_id=event_id)

    backchannel = delta("user", 170, 230, "Oui.", "u1")
    assert transcript_turns([
        delta("assistant", 100, 200, "Je", "a1"),
        backchannel,
        delta("assistant", 200, 350, " vous écoute.", "a2"),
        backchannel,
        delta("assistant", 1500, 1900, "Je vous rappelle.", "a3"),
    ]) == [
        dict(role="assistant", start_ms=100, end_ms=350, text="Je vous écoute."),
        dict(role="user", start_ms=170, end_ms=230, text="Oui."),
        dict(role="assistant", start_ms=1500, end_ms=1900, text="Je vous rappelle."),
    ]


def test_partial_live_transcript_recovers_both_tracks_then_retries_delivery(tmp_path):
    samples = []
    deliveries = []
    summaries = []
    signal = audioop.lin2ulaw(b"\x00\x10" * 160, 2)

    def generate(**kwargs):
        assert kwargs["model"] == "gemini-3.8-flash"
        contents = kwargs["contents"]
        if isinstance(contents, str):
            summaries.append(json.loads(contents))
            return SimpleNamespace(text=json.dumps({field: "valeur simulée" for field in postcall.SUMMARY_FIELDS}))
        with wave.open(io.BytesIO(contents[1].inline_data.data), "rb") as audio:
            assert audio.getnchannels() == 1 and audio.getframerate() == 8000
            frames = audio.readframes(audio.getnframes())
        first = next(index for index in range(0, len(frames), 2) if frames[index:index+2] != b"\0\0")
        samples.append(first // 2)
        start_ms = first // 16
        return SimpleNamespace(text=json.dumps([
            {"text": f"Piste {len(samples)}", "start_ms": start_ms, "end_ms": start_ms + 20},
        ]))

    def callback(payload):
        deliveries.append(payload)
        if len(deliveries) == 1:
            raise TimeoutError()

    path = tmp_path / "calls.sqlite3"

    def make_queue():
        return postcall.PostCallQueue(
            path, "gemini-3.8-flash",
            lambda: nullcontext(SimpleNamespace(models=SimpleNamespace(generate_content=generate))),
            callback,
        )

    recording = postcall.CallRecording(2)
    recording.add("user", 160, signal)
    recording.add("assistant", 240, signal)
    ctx = service.StreamContext(stream_sid="MZ", call_sid="CA", inbound_end=400, played_end=400, sent_end=400)
    prepared = SimpleNamespace(number_session=1, player_name="Camille", initiated_by="player",
                               ai_role="prefet", call_id="job-test")
    spool = make_queue()
    asyncio.run(service.enqueue_phone_record(
        spool, prepared, ctx, recording,
        [("assistant", dict(start_ms=30, end_ms=50, delta="Seulement une piste."))], True, True,
    ))
    assert spool.process_one()
    with closing(sqlite3.connect(path)) as db, db:
        blob, error = db.execute("SELECT recording, last_error FROM jobs").fetchone()
        assert blob is None and error == "delivery:TimeoutError"
        db.execute("UPDATE jobs SET retry_at=0")
    assert make_queue().process_one()
    assert not make_queue().process_one()
    assert samples == [160, 240]
    assert len(summaries) == 1 and len(deliveries) == 2
    assert deliveries[0] == deliveries[1]
    assert deliveries[1]["transcript_source"] == "recording"
    assert deliveries[1]["turns"] == [
        {"role": "user", "text": "Piste 1", "start_ms": 20, "end_ms": 40},
        {"role": "assistant", "text": "Piste 2", "start_ms": 30, "end_ms": 50},
    ]


@pytest.mark.parametrize("reply", [{"ok": True}, {"ok": True, "ack": True, "call_sid": "wrong"}])
def test_delivery_requires_a_receipt_for_the_same_call(monkeypatch, reply):
    monkeypatch.setattr(postcall.requests, "post", lambda *_args, **_kwargs: SimpleNamespace(
        status_code=200, json=lambda: reply,
    ))
    with pytest.raises(ValueError, match="Missing voice receipt"):
        postcall.deliver("https://app.invalid", "mock", {"call_sid": "CA"})


def test_short_silent_call_is_delivered_without_gemini(tmp_path):
    payloads = []

    def no_gemini():
        pytest.fail("A silent call should not invoke Gemini")

    spool = postcall.PostCallQueue(tmp_path / "calls.sqlite3", "gemini-3.8-flash", no_gemini, payloads.append)
    asyncio.run(spool.enqueue({"call_sid": "CA", "turns": []}, None, set(), 0))
    assert spool.process_one()
    assert payloads[0]["turns"] == [] and "Aucun échange" in payloads[0]["summary"]["key_points"]
