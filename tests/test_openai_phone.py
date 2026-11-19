import asyncio
import base64
import json
from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import openai_live_phone as live
import phone_postcall


class FakeLiveSocket:
    def __init__(self, codec="pcm"):
        self.events = asyncio.Queue()
        self.sent = []
        self.closed = False
        self.output = bytes([0xB1]) * 160 if codec == "pcmu" else bytes(960)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        self.closed = True

    async def send(self, raw):
        event = json.loads(raw)
        self.sent.append(event)
        kind = event['type']
        if kind == 'session.start':
            await self.put('session.started', session={'audio': event['session']['audio']})
        elif kind == 'session.input_audio.append':
            if sum(sent["type"] == kind for sent in self.sent) == 1:
                await self.put('session.output_audio.delta', delta=base64.b64encode(self.output).decode())
                await self.put('session.input_transcript.delta', start_ms=0, end_ms=200, delta='Bonjour.')
                await self.put('session.output_transcript.delta', start_ms=100, end_ms=250, delta='Bonjour,')
        elif kind == 'session.close':
            await self.put('session.output_transcript.delta', start_ms=200, end_ms=400, delta=' ici Sarah.')
            await self.put('session.closed')

    async def put(self, kind, **fields):
        await self.events.put(json.dumps({'type': kind, **fields}))

    async def recv(self):
        return await self.events.get()

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.recv()


@pytest.fixture
def voice_settings(monkeypatch, tmp_path):
    import app as service

    for name, value in {
        'VOICE_CALL_PROVIDER': 'openai', 'VOICE_ADMIN_API_KEY': 'test-admin',
        'OPENAI_API_KEY': 'test-provider', 'PUBLIC_BASE_URL': 'https://voice.invalid',
        'TWILIO_FROM_NUMBER': '+33155550101', 'TWILIO_ACCOUNT_SID': 'test',
        'TWILIO_AUTH_TOKEN': 'test', 'ALLOWED_TO_PREFIXES': [], 'ENABLE_TRANSCRIPTIONS': True,
        'VOICE_POSTCALL_DIR': tmp_path, 'MAIN_APP_BASE_URL': 'https://app.invalid',
        'INTERNAL_APP_TOKEN': 'test-callback', 'OPENAI_PHONE_CODEC': 'pcm', 'TWILIO_SAY_BEFORE_STREAM': '',
        'INBOUND_RING_MIN': 0, 'INBOUND_RING_MAX': 0,
    }.items():
        monkeypatch.setattr(service, name, value)
    service.PREPARED.clear()
    return service


def test_signed_inbound_registers_dialled_role_before_connecting_without_added_ring(voice_settings, monkeypatch):
    from twilio.request_validator import RequestValidator

    service = voice_settings
    upstream = FakeLiveSocket()
    number = "+33939240123"
    sid = "CA" + "1" * 32
    monkeypatch.setattr(service, "TWILIO_ACTOR_NUMBERS", {"prefet": number})
    monkeypatch.setattr(live.websockets, "connect", lambda *_args, **_kwargs: upstream)
    requests = []

    def resolve(action, payload):
        requests.append((action, payload))
        if action == "call_started":
            return {"ok": True}
        return {"run_id": "run", "player_id": "player", "number_session": 2, "player_name": "Sophie",
                "to": payload["from"], "briefing": {"points_importants_de_ce_fil": [
                    {"auteur": "Sophie", "destinataire": "Le préfet", "extrait_original": "Engagement SMS à 15 heures."},
                ]}}

    monkeypatch.setattr(service, "frontend_request", resolve)
    data = {"AccountSid": "test", "To": number, "From": "+33655550101", "CallSid": sid}
    signature = RequestValidator("test").compute_signature("https://voice.invalid/twilio/voice", data)
    async def capture(*_args):
        return None
    monkeypatch.setattr(service, "enqueue_phone_record", capture)
    with TestClient(service.app) as client:
        assert client.post("/twilio/voice", data=data).status_code == 403
        response = client.post("/twilio/voice", data=data, headers={"X-Twilio-Signature": signature})
        root = ElementTree.fromstring(response.text)
        assert root[0].tag == "Connect"
        assert root.find("Pause") is None
        assert requests[0][0] == "resolve" and requests[0][1]["actor"] == "prefet"
        assert requests[1][0] == "call_started" and requests[1][1]["direction"] == "in"
        config = upstream.sent[0]["session"]
        assert "Jean-Marc Delaunay" in config["instructions"]
        assert "Engagement SMS à 15 heures." in config["input"][0]["content"][0]["text"]
        assert "Allô" in config["input"][1]["content"][0]["text"]
        assert [event["type"] for event in upstream.sent] == [
            "session.start", "session.input_audio.append", "session.commentary.append",
        ]
        params = {p.get("name"): p.get("value") for p in root.iter("Parameter")}
        ws_sig = RequestValidator("test").compute_signature("wss://voice.invalid/twilio/stream", {})
        with client.websocket_connect("/twilio/stream", headers={"X-Twilio-Signature": ws_sig}) as ws:
            ws.send_json({"event": "start", "start": {"streamSid": "MZ-test", "callSid": sid, "customParameters": params}})
            assert ws.receive_json()["event"] == "media"
            assert ws.receive_json()["event"] == "mark"
            ws.send_json({"event": "stop"})
    assert [name for name, _ in requests] == ["resolve", "call_started"]
    openings = [event for event in upstream.sent if event["type"] == "session.commentary.append"]
    assert len(openings) == 1 and "Allô" in openings[0]["content"]


def test_sms_uses_own_number_and_checks_single_use_dispatch_before_provider(voice_settings, monkeypatch):
    service = voice_settings
    number = "+33939240123"
    posted = []
    monkeypatch.setattr(service, "TWILIO_ACTOR_NUMBERS", {"colonel_pompiers": number})
    monkeypatch.setattr(service, "frontend_request", lambda action, payload: (
        {"ok": True} if not posted and action == "authorize_sms" and payload["from"] == number else None
    ))
    monkeypatch.setattr(service, "_twilio_client", lambda: SimpleNamespace(messages=SimpleNamespace(
        create=lambda **kwargs: posted.append(kwargs) or SimpleNamespace(sid="SM" + "3" * 32, status="queued")
    )))
    data = {"actor": "colonel_pompiers", "to": "+33655550101", "body": "Confirmez l’accès nord.",
            "job_id": "4" * 32}
    with TestClient(service.app) as client:
        headers = {"X-API-Key": "test-admin"}
        assert client.post("/api/send_sms", json=data, headers=headers).status_code == 200
        assert client.post("/api/send_sms", json=data, headers=headers).status_code == 409
    assert len(posted) == 1 and posted[0]["from_"] == number


def test_managed_outbound_context_keeps_recent_decisions_and_attribution_in_a_short_briefing():
    long_text = "Détail ancien de l'exercice. " * 100
    previous_sms = {"auteur": "Marc", "destinataire": "Sarah Lenoir", "propos": "Le portail nord était fermé."}
    data = {
        "contexte_initial_connu": "Une alerte est reçue. " + long_text,
        "informations_communicables_de_ce_personnage": long_text,
        "interlocuteur": {"nom": "Sophie", "fonction": "Direction"},
        "motif": "Confirmer l'accès au site",
        "journal_de_la_cellule": {
            "événements_et_conséquences_publiés": [{"extrait": long_text}] * 10
            + [{"extrait": "Le portail a été ouvert à 14 h 15."}],
            "décisions_soumises_par_la_cellule": [long_text] * 10 + ["La cellule décide d'envoyer les gendarmes."],
        },
        "échanges_avec_les_autres_joueurs": [{"extraits": [previous_sms], "scenario_time": "14 h"}] * 20,
        "notes_animation": "Informations futures confidentielles.",
    }
    config = live.phone_session("journaliste", "admin", "Direction", "Dossier hors partie", briefing=data)
    context = config["input"][0]["content"][0]["text"]
    assert "Sarah Lenoir" in config["instructions"]
    assert "ton intérêt s'entend dès la première question" in config["instructions"]
    assert "Garde le même timbre" in config["instructions"]
    assert "accélère la relance" in config["instructions"]
    assert "d'envoyer les gendarmes" in context and "exécution à confirmer" in context
    assert "Le portail a été ouvert à 14 h 15." in context
    assert "14 h · Marc → Sarah Lenoir : Le portail nord était fermé." in context
    assert not any(value in context for value in ("Dossier hors partie", "Informations futures confidentielles"))
    assert len(context) < 3800
    inbound = live.phone_session("journaliste", "inbound", "Direction", "", briefing=data)
    assert "ton intérêt s'entend dès la première question" not in inbound["instructions"]
    assert "Garde le même timbre" in inbound["instructions"]


def test_call_audio_measures_reply_gap_without_transcripts_and_ignores_unplayed_audio():
    recording = phone_postcall.CallRecording(16)
    voice = b"\xb1" * 160
    def speech(role, start_ms, duration_ms):
        for t in range(start_ms, start_ms + duration_ms, 20):
            received = (t - 20) * 8 if role == "assistant" else None
            recording.add(role, t * 8, voice, received_sample=received)
    speech("assistant", 100, 400)
    speech("user", 1000, 1000)
    speech("assistant", 7000, 600)
    speech("user", 8000, 400)
    speech("assistant", 8900, 300)
    speech("user", 10000, 400)
    speech("assistant", 10600, 300)
    assert recording.audible_reply_timings_ms(9200 * 8) == [
        {"gap": 5000, "before_audio_arrival": 4980, "relay": 20},
        {"gap": 500, "before_audio_arrival": 480, "relay": 20},
    ]


def test_burst_of_digital_silence_cannot_hold_the_next_voice_packet(voice_settings):
    service = voice_settings
    pause = b"\xff\x7f" * (4400 * 4)
    speech = b"\xb1" * 1600
    low_level_audio = b"\xfe" * 160

    async def packets():
        for chunk in (pause, pause, speech, low_level_audio):
            yield json.dumps({"type": "session.output_audio.delta", "delta": base64.b64encode(chunk).decode()})
        yield json.dumps({"type": "session.closed"})

    async def exercise():
        ctx = service.StreamContext("stream", "call")
        queue = asyncio.Queue(maxsize=service.MAX_PENDING_PHONE_FRAMES)
        assert await asyncio.wait_for(service.openai_phone_receiver_loop(
            packets(), service.AudioConverter("pcmu"), queue, [], asyncio.Event(), ctx,
        ), timeout=.5)
        # No Twilio consumer is running. A bounded, paced reader would stall on the first silence packet.
        assert ctx.skipped_silence_samples == 69440
        items = [queue.get_nowait() for _ in range(queue.qsize())]
        kept = b"".join(frame for frame, _ in items)
        assert kept == pause[:960] + speech + low_level_audio
        assert all(timestamp < 4000 for _, timestamp in items)

    asyncio.run(exercise())


def test_full_playback_buffer_reports_an_overflow_instead_of_waiting_on_twilio(voice_settings):
    service = voice_settings

    async def packets():
        yield json.dumps({"type": "session.output_audio.delta", "delta": base64.b64encode(b"\xb1" * 320).decode()})

    async def exercise():
        ctx = service.StreamContext("stream", "call")
        with pytest.raises(BufferError):
            await asyncio.wait_for(service.openai_phone_receiver_loop(
                packets(), service.AudioConverter("pcmu"), asyncio.Queue(maxsize=1), [], asyncio.Event(), ctx,
            ), timeout=.5)
        assert ctx.playback_uncertain

    asyncio.run(exercise())


@pytest.mark.parametrize("codec", ["pcm", "pcmu"])
def test_outbound_call_bridges_openai_audio_and_posts_transcript(voice_settings, monkeypatch, codec):
    service = voice_settings
    upstream = FakeLiveSocket(codec)
    twilio = []
    transcripts = []
    delivered = Event()
    summary_requests = []
    monkeypatch.setattr(service, 'OPENAI_PHONE_CODEC', codec)

    def create_call(**kwargs):
        assert upstream.sent[0]['type'] == 'session.start'
        twilio.append(kwargs)
        return SimpleNamespace(sid='CA-test')

    def connect(url, **kwargs):
        assert url == live.URL and kwargs['extra_headers'] == {'Authorization': 'Bearer test-provider'}
        return upstream

    monkeypatch.setattr(live.websockets, 'connect', connect)
    monkeypatch.setattr(service, '_twilio_client', lambda: SimpleNamespace(calls=SimpleNamespace(create=create_call)))

    def summarize(**kwargs):
        assert upstream.closed
        summary_requests.append(kwargs)
        return SimpleNamespace(text=json.dumps({field: 'mock' for field in phone_postcall.SUMMARY_FIELDS}))

    def post(url, *, json, headers, **kwargs):
        assert url == 'https://app.invalid/internal/voice/transcript'
        assert headers == {'X-Internal-Token': 'test-callback'}
        transcripts.append(json)
        delivered.set()
        return SimpleNamespace(status_code=200, json=lambda: {'ok': True, 'ack': True, 'call_sid': 'CA-test'})

    monkeypatch.setattr(service, 'gemini_client', lambda *_: nullcontext(SimpleNamespace(
        models=SimpleNamespace(generate_content=summarize),
    )))
    monkeypatch.setattr(phone_postcall.requests, 'post', post)
    history = ('[CONTEXTE INITIAL DE LA SIMULATION]\nSecret futur\n[FIN DU CONTEXTE INITIAL]\n\n'
               '[TOURS ET ÉVÉNEMENTS]\n\nia: Fait publié.\n\nAutre paragraphe.\n\nadmin: Consigne historique.')
    with TestClient(service.app) as client:
        response = client.post('/api/prepare_call', headers={'X-API-Key': 'test-admin'}, json={
            'to': '+33655550101', 'number_session': 18, 'ai_role': 'journaliste', 'history_text': history,
        })
        assert response.status_code == 200
        config = upstream.sent[0]['session']
        assert config['model'] == 'gpt-live-1'
        assert config['audio']['format'] == live.AUDIO_FORMATS[codec]
        text = config['input'][0]['content'][0]['text']
        assert 'Fait publié.' in text and 'Autre paragraphe.' in text
        assert 'Secret futur' in text.split('[DOSSIER DU SCÉNARIO', 1)[1]
        assert 'Consigne historique' not in text
        params = {p.attrib['name']: p.attrib['value'] for p in ElementTree.fromstring(twilio[0]['twiml']).iter('Parameter')}
        assert 'stream_token' not in response.json()
        start = {'event': 'start', 'start': {
            'streamSid': 'MZ-test', 'callSid': 'CA-test', 'customParameters': params,
            'mediaFormat': {'encoding': 'audio/x-mulaw', 'sampleRate': 8000, 'channels': 1},
        }}
        with pytest.raises(WebSocketDisconnect) as rejected:
            with client.websocket_connect('/twilio/stream') as ws:
                ws.send_json({'event': 'start', 'start': dict(start['start'], customParameters={'call_id': params['call_id']})})
                ws.receive_json()
        assert rejected.value.code == 1008
        with client.websocket_connect('/twilio/stream') as ws:
            ws.send_json(start)
            incoming = bytes(range(160))
            ws.send_json({'event': 'media', 'media': {'timestamp': '0', 'payload': base64.b64encode(incoming).decode()}})
            outbound = ws.receive_json()
            assert outbound['event'] == 'media' and outbound['streamSid'] == 'MZ-test'
            outbound_audio = base64.b64decode(outbound['media']['payload'])
            assert len(outbound_audio) == 160
            if codec == 'pcmu':
                assert outbound_audio == upstream.output
            # The input keeps flowing while Twilio has not acknowledged the assistant audio.
            ws.send_json({'event': 'media', 'media': {'timestamp': '20', 'payload': base64.b64encode(incoming).decode()}})
            mark = ws.receive_json()
            assert mark['event'] == 'mark'
            ws.send_json(mark)
            ws.send_json({'event': 'stop'})
            with pytest.raises(WebSocketDisconnect):
                ws.receive_json()
        assert delivered.wait(3)
    assert [e['type'] for e in upstream.sent] == [
        'session.start', 'session.input_audio.append', 'session.commentary.append',
        'session.input_audio.append', 'session.close',
    ]
    inbound_audio = base64.b64decode(upstream.sent[1]['audio'])
    if codec == 'pcmu':
        assert inbound_audio == incoming
    else:
        assert len(inbound_audio) == 956
    assert upstream.closed and not service.PREPARED
    assert len(transcripts) == 1
    assert len(summary_requests) == 1
    assert transcripts[0]['number_session'] == 18 and transcripts[0]['call_sid'] == 'CA-test'
    assert transcripts[0]['transcript_source'] == 'live'
    assert transcripts[0]['summary']['new_hypotheses'] == 'mock'
    assert transcripts[0]['turns'] == [
        {'role': 'user', 'text': 'Bonjour.', 'start_ms': 0, 'end_ms': 200},
        {'role': 'assistant', 'text': 'Bonjour, ici Sarah.', 'start_ms': 100, 'end_ms': 400},
    ]


def test_missing_openai_key_does_not_dial(voice_settings, monkeypatch):
    service = voice_settings
    monkeypatch.setattr(service, 'OPENAI_API_KEY', '')
    monkeypatch.setattr(service, '_twilio_client', lambda: pytest.fail('No call expected'))
    with TestClient(service.app) as client:
        response = client.post('/api/prepare_call', headers={'X-API-Key': 'test-admin'}, json={'to': '+33655550101'})
    assert response.status_code == 500
    assert 'OPENAI_API_KEY' in response.json()['detail']


def test_legacy_gemini_phone_setting_cannot_start_a_voice_call(voice_settings, monkeypatch):
    service = voice_settings
    monkeypatch.setattr(service, 'VOICE_CALL_PROVIDER', 'gemini')
    monkeypatch.setattr(service, '_twilio_client', lambda: pytest.fail('No call expected'))
    with TestClient(service.app) as client:
        response = client.post('/api/prepare_call', headers={'X-API-Key': 'test-admin'}, json={'to': '+33655550101'})
    assert response.status_code == 500
    assert 'VOICE_CALL_PROVIDER' in response.json()['detail']


def test_voice_override_supports_named_and_custom_voice(monkeypatch):
    monkeypatch.setenv('OPENAI_PHONE_VOICE_JOURNALISTE', 'marin')
    monkeypatch.setenv('OPENAI_PHONE_VOICE_PREFET', 'voice_mock')
    assert live.phone_session('journaliste', 'player', '', '')['audio']['output']['voice'] == 'marin'
    assert live.phone_session('prefet', 'player', '', '')['audio']['output']['voice'] == {'id': 'voice_mock'}


@pytest.mark.parametrize(('role', 'player_cue', 'admin_cue'), [
    ('journaliste', 'Je prépare mon direct, faites vite', 'C\'est toi qui cherches le scoop'),
    ('prefet', 'Je suis en réunion', 'Tu prends l\'initiative pour obtenir un bilan'),
    ('colonel_pompiers', 'Colonel Vasseur, je vous écoute.', 'Tu appelles parce qu\'un renseignement'),
])
def test_call_origin_selects_one_role_opening_without_losing_scenario(role, player_cue, admin_cue):
    dossier = 'Situation : accident sur la RD 973.\n\nJalon futur : conférence à 17h30.'
    history = ('SESSION 18 — HISTORIQUE FACTUEL\n\n[CONTEXTE INITIAL DE LA SIMULATION]\n\n'
               + dossier + '\n\n[FIN DU CONTEXTE INITIAL]\n\n[TOURS ET ÉVÉNEMENTS]')
    for origin, expected, excluded in [('player', player_cue, admin_cue), ('admin', admin_cue, player_cue)]:
        config = live.phone_session(role, origin, '', history)
        assert dossier in config['input'][0]['content'][0]['text']
        assert 'Avant le premier événement' in config['instructions']
        assert expected in config['instructions'] and excluded not in config['instructions']


def test_provider_rejection_closes_socket_without_dialling(voice_settings, monkeypatch):
    service = voice_settings

    class RefusedSocket(FakeLiveSocket):
        async def send(self, _raw):
            await self.put('error', error={'message': 'private upstream details'})

    upstream = RefusedSocket()
    monkeypatch.setattr(live.websockets, 'connect', lambda *_args, **_kwargs: upstream)
    monkeypatch.setattr(service, '_twilio_client', lambda: pytest.fail('No call expected'))
    with TestClient(service.app) as client:
        response = client.post('/api/prepare_call', headers={'X-API-Key': 'test-admin'}, json={'to': '+33655550101'})
    assert response.status_code == 502
    assert 'private upstream details' not in response.text
    assert upstream.closed and not service.PREPARED


@pytest.mark.parametrize("confirmed_format", [
    {"type": "audio/pcmu", "rate": 8000},
    {"type": "audio/pcm", "rate": 16000},
])
def test_incompatible_startup_audio_is_rejected_before_dialling(voice_settings, monkeypatch, confirmed_format):
    service = voice_settings

    class MismatchedSocket(FakeLiveSocket):
        async def send(self, _raw):
            await self.put("session.started", session={"audio": {"format": confirmed_format}})

    upstream = MismatchedSocket()
    monkeypatch.setattr(live.websockets, "connect", lambda *_args, **_kwargs: upstream)
    monkeypatch.setattr(service, "_twilio_client", lambda: pytest.fail("No call expected"))
    with TestClient(service.app) as client:
        response = client.post("/api/prepare_call", headers={"X-API-Key": "test-admin"},
                               json={"to": "+33655550101"})
    assert response.status_code == 502
    assert upstream.closed and not service.PREPARED


@pytest.mark.parametrize("codec", ["pcmu", "pcm"])
def test_audio_packet_boundaries_preserve_pitch_and_duration(voice_settings, codec):
    import math
    import struct

    service = voice_settings
    sample_rate = live.AUDIO_FORMATS[codec]["rate"]
    tone_hz = 220
    pcm = b"".join(struct.pack("<h", round(10000 * math.sin(2 * math.pi * tone_hz * n / sample_rate)))
                   for n in range(sample_rate))
    audio = service.audioop.lin2ulaw(pcm, 2) if codec == "pcmu" else pcm
    converter = service.AudioConverter(codec)
    packet_size = 407  # The PCM case carries an incomplete sample between packets.
    delivered = b"".join(frame for start in range(0, len(audio), packet_size)
                         for frame in converter.to_twilio(audio[start:start + packet_size]))
    if codec == "pcmu":
        assert delivered == audio
    assert len(delivered) == service.TWILIO_RATE_HZ
    decoded = service.audioop.ulaw2lin(delivered, 2)
    samples = struct.unpack(f"<{service.TWILIO_RATE_HZ}h", decoded)
    crossings = sum(previous <= 0 < current for previous, current in zip(samples, samples[1:]))
    assert crossings == tone_hz


def test_phone_audio_burst_stays_in_order_with_bounded_playback_lead(voice_settings, monkeypatch):
    service = voice_settings
    clock = 100.0
    sent = []

    async def sleep(seconds):
        nonlocal clock
        clock += seconds

    async def exercise():
        queue = asyncio.Queue()
        ctx = service.StreamContext(stream_sid='MZ-test', call_sid='CA-test', started_at=100)
        frames = [bytes([index]) * 160 for index in range(8)]
        for frame in frames:
            queue.put_nowait((frame, 0))

        async def send(message):
            if message['event'] == 'media':
                sent.append((clock, base64.b64decode(message['media']['payload'])))

        monkeypatch.setattr(service, 'time', SimpleNamespace(monotonic=lambda: clock))
        monkeypatch.setattr(service, 'OPENAI_PHONE_PLAYBACK_LEAD_SECONDS', .08)
        monkeypatch.setattr(service.asyncio, 'sleep', sleep)
        task = asyncio.create_task(service.twilio_sender_loop(SimpleNamespace(send_json=send), ctx, queue, None))
        await queue.join()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert [frame for _, frame in sent] == frames

    asyncio.run(exercise())
    assert [timestamp - 100 for timestamp, _ in sent] == pytest.approx([0, 0, 0, 0, .02, .04, .06, .08])
