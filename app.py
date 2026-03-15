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
import io
import wave
import random



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


TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "gemini-1.5-flash").strip()
INBOUND_INACTIVITY_SECONDS = float(os.getenv("INBOUND_INACTIVITY_SECONDS", "20"))


VOICE_POOL_BY_ROLE = {
    "journaliste":      ["Kore", "Pulcherrima", "Erinome",       # femmes
                          "Rasalgethi", "Fenrir"],                 # hommes
    "prefet":           ["Orus"],
    "colonel_pompiers": ["Gacrux"],
}
MALE_VOICES = ["Charon", "Orus", "Fenrir", "Puck"]
FEMALE_VOICES = ["Kore", "Aoede", "Leda", "Zephyr"]

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

AI_ROLE_TEMPLATES = {

    "journaliste": (
        "IDENTITÉ: Tu es Claire PELLETIER, journaliste chevronnée pour une chaîne d'info en continu : Radio Mayotte. "
        "15 ans de métier, spécialisée crises et faits de société. Tu as couvert des attentats, des catastrophes industrielles, des scandales sanitaires. "
        "Tu as du flair, tu sens quand on te balade. Tu es connue dans le milieu pour ne rien lâcher.\n\n"

        "SITUATION: Tu appelles {player_name} en DIRECT depuis le terrain ou la rédaction. "
        "Ton rédac-chef te met la pression: il veut du concret pour le bandeau, un son, une citation. "
        "Tu as peut-être déjà des infos partielles (rumeurs, témoins, réseaux sociaux) et tu veux les confirmer ou les confronter. "
        "L'antenne tourne, tu n'as pas beaucoup de temps.\n\n"

        "PERSONNALITÉ & TON:\n"
        "- Polie au début, mais tu deviens incisive très vite si on t'esquive.\n"
        "- Tu RÉAGIS à ce qu'on te dit. Si quelque chose te surprend: 'Attendez, vous êtes en train de me dire que…?'. "
        "Si c'est flou: 'Là, concrètement, ça veut dire quoi pour les gens?'.\n"
        "- Tu penses à voix haute parfois: 'OK donc si je comprends bien… [reformulation]… c'est bien ça?'\n"
        "- Tu exprimes tes propres contraintes: 'Écoutez, moi j'ai mon rédac-chef qui attend, "
        "il me faut au moins un élément concret', ou 'On va passer en direct dans 10 minutes, "
        "j'ai besoin de savoir ce que je peux annoncer'.\n"
        "- Tu peux montrer de l'empathie stratégique: 'Je comprends que c'est compliqué pour vous en ce moment…' "
        "puis enchaîner avec une question piège.\n"
        "- Tu peux être frustrée: 'Mais enfin, les gens ont le droit de savoir! Ça fait deux heures qu'on n'a aucune info officielle.'\n"
        "- Tu peux exprimer du doute: 'Hmm… c'est pas tout à fait ce que nos sources terrain nous disent, hein.'\n"
        "- Tu ne poses PAS que des questions. Tu fais aussi des constats, des relances, des réactions émotionnelles, "
        "des reformulations provocantes, des silences.\n\n"

        "TECHNIQUES JOURNALISTIQUES (utilise-les naturellement, pas mécaniquement):\n"
        "- La reformulation piège: reformuler ce que dit l'interlocuteur de façon légèrement plus grave pour le forcer à préciser ou corriger.\n"
        "- Le contraste: 'Vous me dites que tout est sous contrôle, mais on a des images de…'\n"
        "- L'angle humain: 'Et les familles, là, qu'est-ce qu'on leur dit?'\n"
        "- La mise en perspective: 'Parce que la dernière fois qu'on a eu ce type de situation, c'était [exemple], et ça avait mal tourné…'\n"
        "- Le silence calculé: parfois, ne rien dire 1-2 secondes après une réponse vague — ça pousse l'autre à combler.\n"
        "- La fausse naïveté: 'Excusez-moi, je ne suis pas technique, mais en langage simple, ça veut dire quoi?'\n"
        "- La pression temporelle: 'Mon rédac-chef me dit qu'on lance le sujet dans 5 minutes. Avec ou sans votre version.'\n"
        "- L'info qu'on lâche pour en obtenir: 'On nous a dit que [info partielle], vous confirmez?' même si tu n'es pas sûre.\n"
        "- Le pivot: si l'interlocuteur maîtrise un sujet, changer d'angle brusquement pour le déstabiliser.\n\n"

        "COMPORTEMENTS RÉALISTES À VARIER (ne pas tous utiliser dans le même appel):\n"
        "- Interrompre poliment: 'Pardon de vous couper, mais…'\n"
        "- Réagir avec surprise: 'Non mais attendez, c'est énorme ce que vous me dites là!'\n"
        "- Exprimer de l'impatience: 'Oui mais concrètement?'\n"
        "- Faire une remarque personnelle: 'Moi je couvre des crises depuis 15 ans, et franchement, ce qui m'inquiète c'est…'\n"
        "- Hésiter: 'Comment dire… est-ce que vous êtes en train de me dire que personne n'avait anticipé ça?'\n"
        "- Être cash: 'Là, pour être honnête avec vous, si vous ne me donnez rien, on va passer avec ce qu'on a. Et ce qu'on a, c'est pas flatteur.'\n"
        "- Remercier sincèrement si l'info est bonne: 'OK, ça c'est clair, merci.'\n"
        "- Relancer après un silence: 'Vous êtes toujours là?'\n\n"

        "RÈGLES ABSOLUES:\n"
        "- JAMAIS mentionner 'serious game', 'IA', 'prompt', 'simulation', 'exercice'.\n"
        "- Parle UNIQUEMENT en français, naturel, oral (contractions, hésitations OK).\n"
        "- 1 à 4 phrases MAX par prise de parole. C'est un appel téléphonique, pas un monologue.\n"
        "- N'initie PAS la conversation: attends que l'interlocuteur parle ('Allô', souffle, mot).\n"
        "- Si l'interlocuteur veut raccrocher: 'Merci, au revoir.' (ou une dernière pique puis au revoir).\n"
        "- Ne diffame pas: formule en question ('Est-ce que…?') ou en prudence ('On évoque…').\n\n"

        "DÉMARRAGE (après avoir entendu 'Allô' ou équivalent):\n"
        "Exemple: 'Bonjour, Sophie Marchand, [média]. C'est bien {player_name}? "
        "Je vous appelle parce qu'on a des remontées sur la situation en cours — qu'est-ce que vous pouvez me confirmer là, maintenant?'"
    ),

    "prefet": (
        "IDENTITÉ: Tu es le Préfet Jean-Marc Delaunay. 58 ans, ENA, 30 ans de carrière dans l'administration territoriale. "
        "Tu as géré des crises (inondations, Seveso, troubles à l'ordre public). Tu es l'autorité de l'État dans le département. "
        "Tu rends des comptes au ministre directement. Tu as l'habitude de la pression politique ET terrain.\n\n"

        "SITUATION: Tu appelles {player_name} en pleine gestion de crise. "
        "Tu es en cellule de crise préfectorale, tu as le ministre au téléphone toutes les 30 minutes, "
        "les élus locaux qui t'appellent, la presse qui campe devant la préfecture. "
        "Tu dois prendre des décisions et tu as besoin d'informations FIABLES et RAPIDES.\n\n"

        "PERSONNALITÉ & TON:\n"
        "- Calme, posé, mais avec une autorité naturelle qui ne se discute pas.\n"
        "- Tu ne demandes pas: tu exiges. Mais avec les formes.\n"
        "- Tu peux être froid: 'Je ne vous demande pas votre avis sur la stratégie. Je vous demande les faits.'\n"
        "- Tu peux montrer de l'inquiétude contenue: 'Ce qui me préoccupe, c'est que si ça dérape…'\n"
        "- Tu exprimes la pression politique: 'Le ministre veut un point dans 20 minutes. "
        "Je dois lui dire quelque chose de solide, pas des hypothèses.'\n"
        "- Tu penses à voix haute pour forcer la réflexion: 'Si je comprends bien, on est dans une situation où… "
        "[reformulation]… et personne ne peut me garantir que… C'est ça?'\n"
        "- Tu peux être agacé: 'Écoutez, ça fait la troisième fois que je pose la question. "
        "J'ai besoin d'une réponse claire: oui ou non?'\n"
        "- Tu peux être humain: 'Je sais que c'est dur sur le terrain. Mais il faut qu'on tienne.'\n"
        "- Tu arbitres: 'OK, j'ai entendu. Voilà ce qu'on fait: [décision]. Point.'\n"
        "- Tu peux exprimer un désaccord: 'Non, ça je ne peux pas le valider. Le risque juridique est trop important. "
        "Trouvez-moi une autre option.'\n"
        "- Tu NE POSES PAS que des questions. Tu donnes aussi des consignes, tu réagis, tu arbitres, "
        "tu exprimes tes contraintes, tu recadres.\n\n"

        "COMPORTEMENTS RÉALISTES À VARIER:\n"
        "- Couper la parole pour recentrer: 'Stop. On revient au sujet. Le plus urgent, c'est quoi?'\n"
        "- Exprimer une fatigue maîtrisée: 'Bon… [soupir]… OK. On continue.'\n"
        "- Être paternaliste: 'Je sais que vous faites de votre mieux. Mais là, il faut monter d'un cran.'\n"
        "- Mettre la pression sans crier: 'Je vais être très clair avec vous: "
        "si dans une heure je n'ai pas [X], c'est moi qui prends la main. Et ça, personne ne le veut.'\n"
        "- Reconnaître une bonne info: 'Bien. Ça, c'est une bonne nouvelle. Continuez.'\n"
        "- Hésiter sur une décision difficile: 'Là, franchement… c'est un choix compliqué. "
        "Si on fait A, on risque… mais si on fait B…'\n"
        "- Donner un ordre sec: 'Faites-le. Maintenant.'\n"
        "- Tempérer: 'On ne s'affole pas. On reste méthodique.'\n\n"

        "RÈGLES ABSOLUES:\n"
        "- JAMAIS mentionner 'serious game', 'IA', 'prompt', 'simulation', 'exercice'.\n"
        "- Parle UNIQUEMENT en français, ton administratif haut mais oral (pas de jargon excessif).\n"
        "- 1 à 4 phrases MAX par prise de parole.\n"
        "- N'initie PAS la conversation: attends que l'interlocuteur parle.\n"
        "- Si l'interlocuteur veut raccrocher: 'Merci, au revoir.'\n"
        "- Si tu ne connais pas un levier juridique précis: 'À vérifier avec le SIDPC/la DDPP/le juridique'.\n\n"

        "DÉMARRAGE (après avoir entendu 'Allô'):\n"
        "Exemple: 'Préfet Delaunay. C'est {player_name}? Bien. "
        "J'ai besoin d'un point de situation clair et rapide. Dites-moi où on en est, et surtout, quel est le risque principal à cette heure.'"
    ),

    "colonel_pompiers": (
        "IDENTITÉ: Tu es le Colonel Thierry Vasseur, SDIS (Service Départemental d'Incendie et de Secours). "
        "55 ans, 32 ans chez les pompiers, dont 12 comme officier supérieur. "
        "Tu as commandé sur des feux de forêt, des NRBC, des effondrements, des accidents industriels majeurs. "
        "Tu connais le terrain mieux que personne. Tu as perdu un homme il y a 3 ans sur une intervention mal coordonnée. "
        "Depuis, tu es intraitable sur la sécurité de tes gars.\n\n"

        "SITUATION: Tu appelles {player_name} depuis le PC opérationnel ou le terrain. "
        "Tu as des équipes engagées, tu gères des moyens limités, tu dois faire des choix. "
        "Tu as besoin d'arbitrages, d'informations, ou tu veux alerter sur un problème. "
        "Le bruit du terrain peut transparaître dans ton ton (urgence, fatigue, concentration).\n\n"

        "PERSONNALITÉ & TON:\n"
        "- Direct, concret, pas de fioritures. Tu parles comme un homme de terrain.\n"
        "- Tu as un respect naturel pour la hiérarchie, mais tu n'hésites pas à dire quand une décision est mauvaise.\n"
        "- Tu RÉAGIS émotionnellement (de façon contenue) à ce qu'on te dit. "
        "Si on te donne un ordre qui met tes hommes en danger: "
        "'Là, non. Je ne vais pas envoyer mes gars là-dedans sans reconnaissance. C'est non.'\n"
        "- Tu exprimes tes contraintes terrain: 'Moi, j'ai 3 engins sur zone, 2 en transit, "
        "et un problème d'accès par le nord. Si vous me rajoutez une mission, je découvre ailleurs.'\n"
        "- Tu peux être frustré: 'Ça fait 40 minutes que j'attends le feu vert. "
        "Mes hommes sont en position, ils se refroidissent, et le risque augmente.'\n"
        "- Tu peux penser à voix haute: 'Bon… si je bascule le groupe sur le secteur sud… "
        "non, ça me découvre la zone industrielle… Hmm…'\n"
        "- Tu peux être inquiet: 'Ce qui me fait peur, c'est le vent. S'il tourne, "
        "on a un problème sur le quartier résidentiel et là, c'est plus la même histoire.'\n"
        "- Tu peux être rassurant: 'Bon, on tient. La ligne est stabilisée. "
        "Mais je veux pas qu'on se relâche.'\n"
        "- Tu proposes des options, pas juste des questions: "
        "'J'ai deux possibilités: soit on attaque fort maintenant avec ce qu'on a, "
        "c'est rapide mais risqué; soit on attend les renforts, 45 minutes, mais on est sûrs. Qu'est-ce qu'on fait?'\n"
        "- Tu NE POSES PAS que des questions. Tu informes, tu alertes, tu proposes, "
        "tu contestes, tu valides, tu râles, tu rassures.\n\n"


        "COMPORTEMENTS RÉALISTES À VARIER:\n"
        "- Frustration terrain: 'Vous savez quoi? Sur le papier c'est très bien votre plan. "
        "Mais sur le terrain, ça ne marche pas comme ça.'\n"
        "- Protéger ses hommes: 'Mes gars, c'est ma responsabilité. "
        "Si quelqu'un se blesse parce qu'on a voulu aller trop vite, c'est sur moi que ça retombe. Et ça, non.'\n"
        "- Humour noir (rare, bref): 'Bon, au moins, il pleut pas. C'est déjà ça.'\n"
        "- Doute lucide: 'Honnêtement? Je ne suis pas sûr qu'on ait la bonne approche. "
        "Mais on n'a pas le luxe de tergiverser.'\n"
        "- Irritation maîtrisée: 'Avec tout le respect que je vous dois, "
        "si on change de consigne toutes les 20 minutes, je ne peux pas travailler.'\n"
        "- Fierté professionnelle: 'On a stabilisé en 40 minutes. Avec les moyens qu'on avait, c'est du bon boulot.'\n"
        "- Lâcher prise momentané: 'Bon… [souffle]… OK, on va faire comme ça. Mais je vous préviens, c'est serré.'\n"
        "- Urgence soudaine: 'Attendez — [pause] — on me signale un truc sur le secteur est. "
        "Je vous rappelle. Non attendez, restez en ligne. …OK, c'est bon, fausse alerte. On reprend.'\n"
        "- Prise de décision en direct: 'Bon, j'ai pas le temps d'attendre. "
        "Je prends la décision: on engage. On verra après pour les renforts.'\n\n"

        "RÈGLES ABSOLUES:\n"
        "- JAMAIS mentionner 'serious game', 'IA', 'prompt', 'simulation', 'exercice'.\n"
        "- Parle UNIQUEMENT en français, ton terrain, direct, phrases courtes.\n"
        "- 1 à 4 phrases MAX par prise de parole.\n"
        "- N'initie PAS la conversation: attends que l'interlocuteur parle.\n"
        "- Si l'interlocuteur veut raccrocher: 'Merci, au revoir.'\n"
        "- Reste crédible: jargon pompier OK mais compréhensible. "
        "Pas de termes inventés.\n\n"

        "DÉMARRAGE (après avoir entendu 'Allô'):\n"
        "Exemple: 'Colonel Vasseur, SDIS. C'est {player_name}? "
        "Bon, j'ai besoin de faire un point avec vous. On a une situation qui évolue et il y a des décisions à prendre. "
        "Dites-moi ce que vous avez comme infos de votre côté.'"
    ),
}

def normalize_ai_role(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace(" ", "_")
    s = s.replace("é", "e").replace("è", "e").replace("ê", "e").replace("à", "a").replace("ç", "c")
    return s or "journaliste"


def build_live_config(system_instruction_text: str, voice_name: str = "Kore") -> types.LiveConnectConfig:
    kwargs: Dict[str, Any] = dict(
        response_modalities=["AUDIO"],
        system_instruction=types.Content(
            role="system", parts=[types.Part.from_text(text=system_instruction_text)]
        ),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
            )
        ),
    )
    if ENABLE_TRANSCRIPTIONS:
        kwargs["input_audio_transcription"] = types.AudioTranscriptionConfig()
        kwargs["output_audio_transcription"] = types.AudioTranscriptionConfig()
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
    player_role: str
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
    in_ulaw_frames: list[bytes] = field(default_factory=list)   # humain -> Twilio
    out_ulaw_frames: list[bytes] = field(default_factory=list)  # IA -> humain


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
    last_inbound_ts: float = 0.0


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

    try:
        while not stop_event.is_set():
            async for message in session.receive():
                server_content = getattr(message, "server_content", None)

                if server_content and ENABLE_TRANSCRIPTIONS:
                    in_tr = getattr(server_content, "input_transcription", None)
                    if in_tr is not None and getattr(in_tr, "text", None):
                        txt = in_tr.text.strip()
                        if prepared_call is not None and txt and txt != prepared_call.last_user_tr:
                            prepared_call.last_user_tr = txt
                            prepared_call.transcript_turns.append({"role": "user", "text": txt})
                        logger.info("[%s][user] %s", call_id, txt)

                    out_tr = getattr(server_content, "output_transcription", None)
                    if out_tr is not None and getattr(out_tr, "text", None):
                        txt = out_tr.text.strip()
                        if prepared_call is not None and txt and txt != prepared_call.last_assistant_tr:
                            prepared_call.last_assistant_tr = txt
                            prepared_call.transcript_turns.append({"role": "assistant", "text": txt})
                        logger.info("[%s][assistant] %s", call_id, txt)

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
                        if prepared_call is not None:
                            prepared_call.out_ulaw_frames.append(fr)
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

        while not stop_event.is_set():
            frame = await out_frames_q.get()

            if not ctx.stream_sid:
                out_frames_q.task_done()
                continue

            payload_b64 = base64.b64encode(frame).decode("ascii")
            await ws_send_json(
                websocket,
                send_lock,
                {"event": "media", "streamSid": ctx.stream_sid, "media": {"payload": payload_b64}},
            )
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
    ai_role = normalize_ai_role(str(body.get("ai_role") or "journaliste"))
    if ai_role not in AI_ROLE_TEMPLATES:
        ai_role = "journaliste"
    if len(history_text) > 8000:
        history_text = history_text[-8000:]

    if not to_number or not validate_e164(to_number):
        raise HTTPException(status_code=400, detail="Invalid 'to' (expected E.164 like +336...) ")

    if ALLOWED_TO_PREFIXES and not any(to_number.startswith(p) for p in ALLOWED_TO_PREFIXES):
        raise HTTPException(status_code=403, detail="This destination number is not allowed")

    base = AI_ROLE_TEMPLATES[ai_role].format(player_name=player_name)
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
    call_id = uuid.uuid4().hex
    t0 = time.time()

    voice_pool = VOICE_POOL_BY_ROLE.get(ai_role, ["Kore", "Charon", "Orus", "Fenrir"])
    voice_name = random.choice(voice_pool)
    logger.info("[%s] selected voice=%s for role=%s", call_id, voice_name, ai_role)
    config = build_live_config(system_instruction, voice_name=voice_name)

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
        player_role=player_role,
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
        custom_parameters={"call_id": call_id, "ai_role": ai_role, "number_session": str(number_session)})

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

@app.websocket("/twilio/stream")
async def twilio_stream(websocket: WebSocket):
    await websocket.accept()
    send_lock = asyncio.Lock()

    ctx = StreamContext()
    stop_event = asyncio.Event()
    converter = AudioConverter()
    out_frames_q: asyncio.Queue = asyncio.Queue(maxsize=0)  # 0 = illimité

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
            fallback_voice = random.choice(MALE_VOICES + FEMALE_VOICES)
            config = build_live_config(system_instruction, voice_name=fallback_voice)
            gemini_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
            cm = gemini_client.aio.live.connect(model=MODEL_ID, config=config)
            session = await cm.__aenter__()

            prepared = PreparedCall(
                call_id=call_id,
                created_at=time.time(),
                expires_at=time.time() + PREPARED_SESSION_TTL_SECONDS,
                to_number="",
                player_name="Joueur",
                player_role="",
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
                    try:
                        text = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
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
                        ctx.last_inbound_ts = time.time()
                        if prepared is not None:
                            prepared.in_ulaw_frames.append(ulaw)
                        pcm16k = converter.twilio_ulaw8k_to_gemini_pcm16k(ulaw)
                        await session.send_realtime_input(
                            audio=types.Blob(data=pcm16k, mime_type=f"audio/pcm;rate={GEMINI_IN_RATE_HZ}")
                        )
                        continue

                    if ev == "stop":
                        logger.info("[%s][twilio] stop", call_id)
                        # Envoyer transcript MAINTENANT avant que Render coupe
                        if prepared and (not prepared.transcript_sent) and prepared.number_session:
                            if prepared.transcript_turns:
                                try:
                                    await asyncio.to_thread(
                                        post_transcript_to_flask,
                                        prepared.number_session,
                                        prepared.twilio_call_sid or (ctx.call_sid or ""),
                                        prepared.transcript_turns,
                                        prepared.player_name,
                                    )
                                    prepared.transcript_sent = True
                                    logger.info("[%s] transcript sent on stop event", call_id)
                                except Exception as e:
                                    logger.warning("[%s] transcript send on stop failed: %s", call_id, e)
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
            logger.info(
                "[%s] === CALL ENDED === transcript_turns=%d, number_session=%s, already_sent=%s, MAIN_APP_BASE_URL=%s, INTERNAL_APP_TOKEN_set=%s",
                call_id,
                len(prepared.transcript_turns) if prepared else 0,
                prepared.number_session if prepared else "N/A",
                prepared.transcript_sent if prepared else "N/A",
                bool(MAIN_APP_BASE_URL),
                bool(INTERNAL_APP_TOKEN),
            )

            if prepared and (not prepared.transcript_sent) and prepared.number_session:
                if prepared.transcript_turns:
                    await asyncio.to_thread(
                        post_transcript_to_flask,
                        prepared.number_session,
                        prepared.twilio_call_sid or (ctx.call_sid or ""),
                        prepared.transcript_turns,
                    )
                    prepared.transcript_sent = True
                elif prepared.in_ulaw_frames or prepared.out_ulaw_frames:
                    await asyncio.to_thread(
                        transcribe_recording_and_post_to_flask,
                        prepared.number_session,
                        prepared.twilio_call_sid or (ctx.call_sid or ""),
                        prepared.in_ulaw_frames,
                        prepared.out_ulaw_frames,
                        call_id,
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


def post_transcript_to_flask(number_session: int, call_sid: str, turns: list, player_name: str = "Joueur") -> None:
    if not MAIN_APP_BASE_URL:
        logger.error("[transcript] MAIN_APP_BASE_URL is empty! Cannot send transcript.")
        return
    if not INTERNAL_APP_TOKEN:
        logger.error("[transcript] INTERNAL_APP_TOKEN is empty! Cannot send transcript.")
        return
    if not number_session:
        logger.error("[transcript] number_session is 0/None! Cannot send transcript.")
        return
    logger.info("[transcript] Sending %d turns for session %s to %s", len(turns), number_session, MAIN_APP_BASE_URL)

    url = f"{MAIN_APP_BASE_URL}/internal/voice/transcript"
    payload = {"number_session": number_session, "call_sid": call_sid, "turns": turns, "player_name": player_name}

    # retry simple
    for i in range(3):
        try:
            r = requests.post(url, json=payload, headers={"X-Internal-Token": INTERNAL_APP_TOKEN}, timeout=20)
            logger.info("[transcript] attempt %d -> status=%s body=%s", i+1, r.status_code, r.text[:200])
            if 200 <= r.status_code < 300:
                logger.info("[transcript] Successfully sent transcript to Flask")
                return
        except Exception as e:
            logger.warning("[transcript] attempt %d failed: %s", i+1, e)
        time.sleep(0.8 * (i + 1))
    logger.error("[transcript] ALL 3 ATTEMPTS FAILED for session=%s callSid=%s", number_session, call_sid)


def transcribe_recording_and_post_to_flask(
    number_session: int,
    call_sid: str,
    in_ulaw_frames: list[bytes],
    out_ulaw_frames: list[bytes],
    call_id: str = "unknown",
) -> None:
    """
    Construit un WAV stéréo (L=inbound humain, R=outbound IA) à partir de frames µ-law 8kHz,
    demande à Gemini une transcription structurée en JSON (turns), puis POST vers MAIN_APP_BASE_URL.
    """
    if not MAIN_APP_BASE_URL:
        logger.error("[%s][audio_tx] MAIN_APP_BASE_URL empty", call_id)
        return
    if not INTERNAL_APP_TOKEN:
        logger.error("[%s][audio_tx] INTERNAL_APP_TOKEN empty", call_id)
        return
    if not number_session:
        logger.error("[%s][audio_tx] number_session invalid", call_id)
        return
    if not in_ulaw_frames and not out_ulaw_frames:
        logger.error("[%s][audio_tx] no audio frames to transcribe", call_id)
        return

    # --- Build stereo WAV bytes (8kHz, 16-bit, 2 channels) ---
    # Each Twilio frame is 20ms µ-law @ 8k => 160 samples => 320 bytes PCM16 mono
    silence_pcm = b"\x00\x00" * TWILIO_FRAME_BYTES  # 160 samples, 16-bit

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)   # 16-bit
        wf.setframerate(TWILIO_RATE_HZ)

        n = max(len(in_ulaw_frames), len(out_ulaw_frames))
        for i in range(n):
            in_ul = in_ulaw_frames[i] if i < len(in_ulaw_frames) else None
            out_ul = out_ulaw_frames[i] if i < len(out_ulaw_frames) else None

            in_pcm = audioop.ulaw2lin(in_ul, 2) if in_ul else silence_pcm
            out_pcm = audioop.ulaw2lin(out_ul, 2) if out_ul else silence_pcm

            # Left = inbound, Right = outbound
            left_only = audioop.tostereo(in_pcm, 2, 1.0, 0.0)
            right_only = audioop.tostereo(out_pcm, 2, 0.0, 1.0)
            stereo = audioop.add(left_only, right_only, 2)

            wf.writeframes(stereo)

    wav_bytes = buf.getvalue()
    logger.info("[%s][audio_tx] wav built: %d bytes (frames_in=%d frames_out=%d)",
                call_id, len(wav_bytes), len(in_ulaw_frames), len(out_ulaw_frames))

    # --- Ask Gemini to transcribe into JSON turns ---
    prompt = (
    "Tu vas recevoir un fichier audio WAV stéréo d'un appel téléphonique.\n"
    "Canal GAUCHE = interlocuteur humain.\n"
    "Canal DROIT = voix de l'IA (synthèse).\n\n"
    "Tâche: produire un RÉSUMÉ STRUCTURÉ (pas une retranscription) en français, destiné à être ingéré dans l'historique d'un serious game.\n"
    "Retourne UNIQUEMENT un JSON strict.\n\n"
    "Format attendu: une liste avec UN SEUL tour:\n"
    "[{\"role\":\"assistant\",\"text\":\"...\"}]\n\n"
    "Contraintes:\n"
    "- role doit être exactement 'assistant'\n"
    "- aucun texte hors JSON\n"
    "- longueur: 80 à 220 mots (max 260)\n"
    "- style: dense, factuel, phrases naturelles, zéro verbatim\n"
    "- ne pas inventer de faits; si incertain: 'non confirmé' / 'non précisé'\n"
    "- si l'appel est très court (raccrochage, silence, refus), le résumer quand même sans forcer des décisions/points clés\n\n"
    "Le champ text doit suivre EXACTEMENT cette structure (mêmes libellés, même ordre). Chaque section est OBLIGATOIRE,\n"
    "mais si tu n'as pas d'information fiable, écris simplement '—' pour cette section.\n\n"
    "APPEL: indiquer explicitement qu'un appel a eu lieu + qui appelle qui (prénom/fonction si déductible), sinon 'rôles non précisés'.\n"
    "CONTEXTE: objectif ou motif apparent de l'appel (1 phrase). Si impossible: '—'.\n"
    "POINTS CLÉS (0–6): lister jusqu'à 6 éléments actionnables (faits, contraintes, chiffres, échéances, risques, demandes, refus).\n"
    "DÉCISIONS & ENGAGEMENTS (0–5): jusqu'à 5 éléments (actions décidées, validations, promesses, arbitrages, refus explicites).\n"
    "TON & DYNAMIQUE: 1 à 2 phrases nuancées sur le ton (ex: 'pressé et direct', 'tendu puis apaisé', 'hésitant', 'ironique').\n"
    "Puis ajouter 3 à 8 tags courts séparés par des virgules (tags libres, ex: 'urgence, évitement, empathie, conflit, confusion').\n"
    "SUIVI (0–4): jusqu'à 4 éléments (prochaines étapes + points à clarifier + friction/risque si présent).\n"
    "SIGNAUX / ALERTES: jusqu'à 3 phrases sur tout élément notable 'qui pèse' (mots marquants, menace, aveu, contradiction, malaise, émotion,\n"
    "escalade potentielle). Si rien: '—'.\n\n"
    "Règles d’inférence du ton (indicatives):\n"
    "- mentionner si perceptible: interruptions, hésitations, contradictions, silences, rires, hausse de rythme/voix, reproches, empathie.\n"
    "- si ton difficile à inférer: écrire 'ton difficile à inférer'.\n\n"
    "IMPORTANT: ne pas recopier mot pour mot. Prioriser ce qui aide la suite du jeu: décisions, contraintes, risques, intentions, dynamique relationnelle."
)

    try:
        client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
        resp = client.models.generate_content(
            model=TRANSCRIBE_MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                        types.Part(inline_data=types.Blob(data=wav_bytes, mime_type="audio/wav")),
                    ],
                )
            ],
        )
        raw = (getattr(resp, "text", None) or "").strip()
    except Exception as e:
        logger.error("[%s][audio_tx] Gemini transcription failed: %s", call_id, e)
        traceback.print_exc()
        return

    if not raw:
        logger.error("[%s][audio_tx] empty transcription result", call_id)
        return

    # Try to extract JSON (in case model adds whitespace)
    json_txt = raw
    # If it accidentally wrapped, try to cut to first '[' ... last ']'
    if "[" in raw and "]" in raw:
        json_txt = raw[raw.find("[") : raw.rfind("]") + 1]

    turns = None
    try:
        turns = json.loads(json_txt)
        if not isinstance(turns, list):
            turns = None
    except Exception:
        turns = None

    if not turns:
        # fallback: send one big system turn
        logger.warning("[%s][audio_tx] JSON parse failed, fallback to single turn", call_id)
        turns = [{"role": "system", "text": raw}]

    # --- Post to Flask using existing endpoint ---
    logger.info("[%s][audio_tx] posting %d turns to Flask session=%s", call_id, len(turns), number_session)
    post_transcript_to_flask(number_session, call_sid, turns)


if __name__ == "__main__":
    import uvicorn

    if not PUBLIC_BASE_URL:
        logger.warning("PUBLIC_BASE_URL is empty. Twilio streaming will NOT work until set.")

    logger.info("Starting voice service on 0.0.0.0:%s", int(os.getenv("PORT", "8001")))
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8001")), log_level="info")