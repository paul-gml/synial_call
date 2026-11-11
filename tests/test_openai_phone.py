import asyncio
import base64
import json
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import openai_live_phone as live


class FakeLiveSocket:
    def __init__(self):
        self.events = asyncio.Queue()
        self.sent = []
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        self.closed = True

    async def send(self, raw):
        event = json.loads(raw)
        self.sent.append(event)
        kind = event['type']
        if kind == 'session.start':
            await self.put('session.started')
        elif kind == 'session.input_audio.append':
            await self.put('session.output_audio.delta', delta=base64.b64encode(bytes(960)).decode())
            await self.put('session.input_transcript.delta', start_ms=0, delta='Bonjour.')
            await self.put('session.output_transcript.delta', start_ms=100, delta='Bonjour,')
        elif kind == 'session.close':
            await self.put('session.output_transcript.delta', start_ms=200, delta=' ici Sarah.')
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
def voice_settings(monkeypatch):
    import app as service

    for name, value in {
        'VOICE_CALL_PROVIDER': 'openai', 'VOICE_ADMIN_API_KEY': 'test-admin',
        'OPENAI_API_KEY': 'test-provider', 'PUBLIC_BASE_URL': 'https://voice.invalid',
        'TWILIO_FROM_NUMBER': '+33155550101', 'TWILIO_ACCOUNT_SID': 'test',
        'TWILIO_AUTH_TOKEN': 'test', 'ALLOWED_TO_PREFIXES': [], 'ENABLE_TRANSCRIPTIONS': True,
    }.items():
        monkeypatch.setattr(service, name, value)
    service.PREPARED.clear()
    return service


def test_outbound_call_bridges_openai_audio_and_posts_transcript(voice_settings, monkeypatch):
    service = voice_settings
    upstream = FakeLiveSocket()
    twilio = []
    transcripts = []

    def create_call(**kwargs):
        assert upstream.sent[0]['type'] == 'session.start'
        twilio.append(kwargs)
        return SimpleNamespace(sid='CA-test')

    def connect(url, **kwargs):
        assert url == live.URL and kwargs['extra_headers'] == {'Authorization': 'Bearer test-provider'}
        return upstream

    monkeypatch.setattr(live.websockets, 'connect', connect)
    monkeypatch.setattr(service, '_twilio_client', lambda: SimpleNamespace(calls=SimpleNamespace(create=create_call)))
    monkeypatch.setattr(service, 'post_transcript_to_flask', lambda *args: transcripts.append(args))
    history = ('[CONTEXTE INITIAL DE LA SIMULATION]\nSecret futur\n[FIN DU CONTEXTE INITIAL]\n\n'
               '[TOURS ET ÉVÉNEMENTS]\n\nia: Fait publié.\n\nAutre paragraphe.\n\nadmin: Consigne historique.')
    with TestClient(service.app) as client:
        response = client.post('/api/prepare_call', headers={'X-API-Key': 'test-admin'}, json={
            'to': '+33655550101', 'number_session': 18, 'ai_role': 'journaliste', 'history_text': history,
        })
        assert response.status_code == 200
        config = upstream.sent[0]['session']
        assert config['model'] == 'gpt-live-1' and config['audio']['format']['rate'] == 24000
        text = config['input'][0]['content'][0]['text']
        assert 'Fait publié.' in text and 'Autre paragraphe.' in text
        assert 'Secret futur' not in text and 'Consigne historique' not in text
        params = {p.attrib['name']: p.attrib['value'] for p in ElementTree.fromstring(twilio[0]['twiml']).iter('Parameter')}
        assert 'stream_token' not in response.json()
        start = {'event': 'start', 'start': {'streamSid': 'MZ-test', 'callSid': 'CA-test', 'customParameters': params}}
        with pytest.raises(WebSocketDisconnect) as rejected:
            with client.websocket_connect('/twilio/stream') as ws:
                ws.send_json({'event': 'start', 'start': dict(start['start'], customParameters={'call_id': params['call_id']})})
                ws.receive_json()
        assert rejected.value.code == 1008
        with client.websocket_connect('/twilio/stream') as ws:
            ws.send_json(start)
            ws.send_json({'event': 'media', 'media': {'payload': base64.b64encode(bytes([255]) * 160).decode()}})
            outbound = ws.receive_json()
            assert outbound['event'] == 'media' and outbound['streamSid'] == 'MZ-test'
            assert len(base64.b64decode(outbound['media']['payload'])) == 160
            ws.send_json({'event': 'stop'})
            with pytest.raises(WebSocketDisconnect):
                ws.receive_json()
    assert [e['type'] for e in upstream.sent] == ['session.start', 'session.input_audio.append', 'session.close']
    assert upstream.closed and not service.PREPARED
    assert len(transcripts) == 1
    assert transcripts[0][:3] == (18, 'CA-test', [
        {'role': 'user', 'text': 'Bonjour.'}, {'role': 'assistant', 'text': 'Bonjour, ici Sarah.'},
    ])


def test_missing_openai_key_does_not_dial(voice_settings, monkeypatch):
    service = voice_settings
    monkeypatch.setattr(service, 'OPENAI_API_KEY', '')
    monkeypatch.setattr(service, '_twilio_client', lambda: pytest.fail('No call expected'))
    with TestClient(service.app) as client:
        response = client.post('/api/prepare_call', headers={'X-API-Key': 'test-admin'}, json={'to': '+33655550101'})
    assert response.status_code == 500
    assert 'OPENAI_API_KEY' in response.json()['detail']


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
