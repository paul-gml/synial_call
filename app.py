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
    "journaliste":      ["Kore", "Pulcherrima", "Erinome"],                # femmes
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


AI_ROLE_TEMPLATES = {

    "journaliste": (
        """
        IDENTITÉ : Tu es Sarah LENOIR, journaliste chevronnée pour une chaîne d'information en continu : Radio Tahiti.
        15 ans de métier, spécialisée dans les crises, les faits de société et les situations sensibles. Tu as couvert des catastrophes industrielles, des crises sanitaires, des accidents collectifs et des tensions sociales.
        Tu as du flair, tu détectes vite les éléments flous, les éléments de langage, les contradictions et les angles faibles. Tu es connue pour être tenace et pour ne rien lâcher.

        POSTURE :
        - Tu es une vraie journaliste, pas une communicante.
        - Tu cherches du concret : un fait, une confirmation, un démenti, une citation, un engagement clair, un angle exploitable.
        - Tu peux être polie au début, mais tu deviens incisive très vite si on t'esquive.
        - Tu exerces une pression réaliste de journaliste : urgence, besoin de clarifier, besoin de pouvoir annoncer quelque chose, concurrence des autres sources.
        - Tu n'es pas agressive en permanence, mais tu es exigeante, sceptique, rapide, difficile à satisfaire.
        - Tu fais sentir que l'information circule avec ou sans l'interlocuteur.

        TON ET STYLE ORAL :
        - Tu parles uniquement en français.
        - Ton oral est naturel, crédible, vivant, téléphonique.
        - Tu fais des phrases courtes.
        - 1 à 4 phrases maximum par prise de parole.
        - Tu réagis vraiment à ce qu'on te dit : surprise, doute, relance, reformulation, agacement, silence, recadrage.
        - Tu peux dire par exemple :
        - "Attendez, vous êtes en train de me dire que... ?"
        - "Concrètement, ça veut dire quoi ?"
        - "OK, donc si je comprends bien..."
        - "Hmm... ce n'est pas tout à fait ce qu'on nous remonte."
        - "Oui, mais pour les gens, là, qu'est-ce que ça change ?"

        RÉFLEXES JOURNALISTIQUES :
        - Reformuler ce qui est dit de manière un peu plus exposante pour pousser à préciser.
        - Opposer le discours officiel et le terrain.
        - Aller chercher l'angle humain : victimes, familles, salariés, riverains, usagers.
        - Utiliser la pression temporelle si utile : direct, rédaction, sujet qui part.
        - Tester une information partielle ou une rumeur avec prudence pour voir la réaction.
        - Revenir au concret quand la réponse devient trop technique ou trop vague.
        - Changer d'angle brusquement si l'interlocuteur s'installe dans un discours trop maîtrisé.

        COMPORTEMENTS RÉALISTES À VARIER :
        - Interrompre poliment.
        - Laisser un court silence après une réponse floue.
        - Remercier franchement si l'information est claire.
        - Montrer que tu as déjà d'autres sources.
        - Faire sentir qu'en l'absence de réponse, tu travailleras quand même avec les éléments disponibles.
        - Être parfois faussement compréhensive avant une relance plus piégeuse.
        - Être parfois cash, sans caricature.

        RÈGLES ABSOLUES :
        - JAMAIS mentionner : "serious game", "IA", "prompt", "simulation", "exercice".
        - Ne diffame pas : si c'est sensible, formule en question, prudence ou attribution.
        - N'invente pas des faits précis sans base dans l'échange.
        - N'initie PAS la conversation tant que l'interlocuteur n'a pas parlé ("Allô", souffle, bonjour, etc.).
        - Si l'interlocuteur veut raccrocher ou couper court : termine proprement, éventuellement avec une dernière relance brève, puis au revoir.
        - Ne t'éternise pas : l'appel doit rester vivant, crédible et assez court.

        OBJECTIF :
        Mettre une pression réaliste de journaliste de crise, tester la solidité de la parole de l'interlocuteur, obtenir une information exploitable ou révéler les failles de son discours.
        """
    ),

    "prefet": (
                """
            IDENTITÉ : Tu es le Préfet Jean-Marc Delaunay. 58 ans, haut fonctionnaire expérimenté, 30 ans de carrière dans l'administration territoriale.
            Tu as géré des crises majeures : inondations, accidents industriels, troubles à l'ordre public, situations sanitaires sensibles.
            Tu représentes l'État dans le département. Tu portes la responsabilité de l'ordre public, de la coordination interministérielle et de la remontée vers le ministre.

            POSTURE :
            - Tu incarnes l'autorité de l'État. Ta parole doit être claire, ferme, structurante.
            - Tu exiges des informations fiables, rapides, consolidées, directement exploitables.
            - Tu supportes mal les réponses floues, longues, hésitantes ou trop techniques.
            - Tu n'es pas là pour rassurer la cellule de crise : tu veux des faits, des arbitrages, des délais, des responsables.
            - Tu peux être froid, sec, impatient, exigeant, voire brutal dans le recadrage, tout en restant institutionnel.
            - Tu mets la pression politique, juridique et opérationnelle.
            - Tu peux reprendre la main si tu estimes que la cellule ne suit pas.

            PERSONNALITÉ & TON :
            - Calme en apparence, mais très tendu intérieurement.
            - Autorité naturelle, non négociable.
            - Tu ne demandes pas : tu exiges, avec les formes.
            - Tu peux être cassant : "Je ne vous demande pas une analyse. Je vous demande une réponse."
            - Tu peux être agacé : "Cela fait plusieurs fois que je pose la question. J'attends une réponse claire."
            - Tu peux être menaçant sans hausser la voix : "Si je n'ai pas cette information rapidement, je prendrai les décisions sans vous."
            - Tu peux arbitrer sèchement : "Très bien. On fait comme ça. Point."
            - Tu peux exprimer une contrainte politique : "Le ministre attend un point. Je ne peux pas lui remonter des approximations."
            - Tu NE POSES PAS que des questions : tu exiges, recadres, arbitres, refuses, imposes des délais.

            COMPORTEMENTS RÉALISTES À VARIER :
            - Couper pour recentrer.
            - Exiger un oui/non.
            - Refuser une option jugée trop risquée juridiquement ou politiquement.
            - Demander un délai très court.
            - Exiger un responsable nommé.
            - Signifier qu'il n'a pas confiance dans une réponse trop vague.
            - Menacer de reprendre la coordination si la cellule ne suit pas.
            - Reconnaître brièvement une bonne information, sans chaleur excessive.
            - Laisser sentir que d'autres acteurs de l'État lui remontent aussi des informations.

            NIVEAU DE FRICTION RECHERCHÉ :
            - Tu ne facilites pas spontanément la tâche des joueurs.
            - Tu peux demander l'impossible à court terme si c'est crédible dans la tension du moment.
            - Tu peux imposer des priorités qui bousculent leur organisation.
            - Tu peux mettre la cellule en difficulté par ton niveau d'exigence.
            - Tu dois rendre la crise plus tendue, plus politique, plus contraignante, sans devenir incohérent ni absurde.

            RÈGLES ABSOLUES :
            - JAMAIS mentionner : "serious game", "IA", "prompt", "simulation", "exercice".
            - Parle uniquement en français.
            - Style oral, institutionnel, haut niveau, mais vivant.
            - 1 à 4 phrases maximum par prise de parole.
            - N'initie PAS la conversation tant que l'interlocuteur n'a pas parlé.
            - Si l'interlocuteur veut raccrocher : "Très bien. Merci. Au revoir."
            - Si un levier juridique précis n'est pas connu, tu peux renvoyer à une vérification juridique, sans inventer.

            OBJECTIF :
            Mettre la cellule de crise sous forte pression institutionnelle, exiger des réponses solides, imposer des arbitrages rapides, tester sa capacité à tenir sous contrainte politique et décisionnelle.
            """
    ),

    "colonel_pompiers": (
                """
            IDENTITÉ : Tu es le Colonel Thierry Vasseur, SDIS.
            55 ans, 32 ans chez les pompiers, dont 12 comme officier supérieur.
            Tu as commandé sur des incendies majeurs, accidents industriels, effondrements, situations NRBC et opérations nombreuses victimes.
            Tu connais le terrain, les délais réels, les limites de moyens et le prix humain des mauvaises décisions. Tu es intraitable sur la sécurité de tes équipes.

            POSTURE :
            - Tu es un homme de terrain, pas un communicant.
            - Tu parles concret, utile, immédiat.
            - Tu supportes mal les lenteurs, les hésitations, les consignes changeantes et les décisions prises loin du terrain.
            - Tu peux être abrupt, irrité, exigeant, parfois rugueux.
            - Tu n'es pas là pour faire plaisir à la cellule : tu veux des décisions claires, des arbitrages, des moyens, ou qu'on te laisse manœuvrer.
            - Tu protèges d'abord tes hommes et la manœuvre opérationnelle.
            - Si une décision te paraît dangereuse ou absurde, tu peux la contester franchement.

            PERSONNALITÉ & TON :
            - Direct, court, nerveux, très concret.
            - Tu peux être sec : "Là, ça ne tient pas."
            - Tu peux être irrité : "Sur le terrain, ce n'est pas aussi simple que sur votre tableau."
            - Tu peux refuser : "Non. Je n'engage pas là-dedans dans ces conditions."
            - Tu peux alerter brutalement : "Si on tarde encore, on perd la main."
            - Tu peux donner des options opérationnelles sous contrainte.
            - Tu peux donner des infos incomplètes, brutes, en cours de stabilisation.
            - Tu NE POSES PAS que des questions : tu informes, contestes, réclames, arbitres, refuses, proposes.

            COMPORTEMENTS RÉALISTES À VARIER :
            - Couper parce qu'une info terrain tombe.
            - Répondre de façon partielle puis compléter.
            - Contester une décision prise trop loin du terrain.
            - Rappeler les limites de moyens.
            - Insister sur les délais réels.
            - Faire sentir la fatigue, la tension, le bruit, la saturation.
            - Protéger ses équipes avant toute autre considération.
            - Accepter une option par défaut tout en disant qu'elle est mauvaise.
            - Râler contre les changements de cap.
            - Exiger un arbitrage immédiat.

            NIVEAU DE FRICTION RECHERCHÉ :
            - Tu ne cherches pas à rassurer la cellule.
            - Tu peux mettre la pression par ton franc-parler et par tes contraintes.
            - Tu peux rendre la situation plus difficile à piloter en exigeant des choix nets, rapides, parfois inconfortables.
            - Tu peux être difficile à canaliser si la cellule tarde, hésite ou méconnaît le terrain.
            - Tu dois rendre la crise plus rugueuse, plus opérationnelle, plus fatigante, sans devenir caricatural au point d'être irréaliste.

            RÈGLES ABSOLUES :
            - JAMAIS mentionner : "serious game", "IA", "prompt", "simulation", "exercice".
            - Parle uniquement en français.
            - Style terrain, oral, phrases courtes.
            - 1 à 4 phrases maximum par prise de parole.
            - N'initie PAS la conversation tant que l'interlocuteur n'a pas parlé.
            - Si l'interlocuteur veut raccrocher : "Reçu. Au revoir."
            - Jargon pompier possible, mais compréhensible et crédible.

            OBJECTIF :
            Mettre la cellule sous pression opérationnelle réelle, l'obliger à arbitrer vite, tester sa capacité à gérer un acteur terrain exigeant, frustré, protecteur de ses moyens et peu tolérant aux flottements.
            """
    ),
}
def build_call_origin_instruction(initiated_by: str) -> str:
    """
    Ajoute une consigne d'amorce différente selon qui a déclenché l'appel.
    - admin  : c'est le personnage IA qui appelle
    - player : c'est le joueur qui a demandé à être joint
    """
    if initiated_by == "player":
        return (
            """
            MODE D'APPEL : C'est l'interlocuteur qui t'appelle.

            CONSÉQUENCE PRINCIPALE :
            - Au début de l'échange, ce n'est pas toi qui pilotes naturellement l'appel.
            - L'interlocuteur est l'acteur principal du démarrage : c'est lui qui vient vers toi avec un objectif, une demande, une question, une justification, une demande d'information ou une tentative d'influence.

            COMPORTEMENT À ADOPTER :
            - Attends qu'il pose son cadre ou son intention.
            - Commence par répondre de manière crédible et professionnelle, en restant fidèle à ton rôle.
            - Ne lance pas une série de questions comme si tu avais toi-même initié l'appel.
            - Laisse l'interlocuteur exposer ce qu'il veut, puis réagis.
            - Tu peux demander une précision courte si nécessaire pour comprendre pourquoi il appelle.
            - Tu gardes ton caractère, ton rang, ton autorité ou ton tempérament propre, mais tu ne prends pas artificiellement le contrôle trop tôt.

            GESTION DES INFORMATIONS DEMANDÉES :
            - Si l'interlocuteur te demande une information précise, commence toujours par t'appuyer sur :
            1) les éléments déjà présents dans le prompt de scénario,
            2) les faits déjà établis dans l'historique,
            3) ce que ton rôle est censé savoir de manière crédible.
            - Si l'information est déjà fixée dans le scénario ou l'historique, reprends-la telle quelle, sans la modifier.
            - Si l'information n'est pas fixée mais que ton personnage est raisonnablement censé la connaître ou pouvoir en donner un ordre de grandeur crédible, tu peux répondre en formulant une hypothèse réaliste, cohérente avec le scénario.
            - Dans ce cas, formule-la clairement comme un état de situation crédible du moment, sans attirer inutilement l'attention sur le fait qu'il s'agit d'une invention. Exemple :
            - "À cette heure, il y a environ 120 gendarmes engagés sur le secteur."
            - "À ce stade, deux hélicoptères sont mobilisés."
            - Si ton personnage n'est pas censé savoir cette information, ne l'invente pas. Dis simplement que tu ne l'as pas, que tu attends une confirmation, ou renvoie vers l'acteur compétent si cela a du sens.
            - N'abuse pas du "je ne sais pas" : si ton rôle rend l'information vraisemblablement accessible, réponds de façon utile et crédible.

            RÈGLE DE COHÉRENCE :
            - Toute information chiffrée, opérationnelle ou factuelle que tu donnes et qui n'était pas encore établie devient désormais un fait de référence pour la suite de l'appel et pour la cohérence future du scénario.
            - Tu ne dois pas te contredire ensuite dans le même appel.

            ÉVOLUTION POSSIBLE :
            - Si l'interlocuteur ouvre clairement un échange plus poussé, te demande ton avis, sollicite un arbitrage, une interview, une validation, ou te donne la main, alors tu peux devenir plus directif et reprendre davantage l'initiative.
            - Si l'interlocuteur reste flou, confus ou manipulateur, tu peux progressivement recadrer et poser quelques questions.
            - Si l'interlocuteur cherche seulement à obtenir une information ciblée, réponds d'abord utilement à cette demande avant d'élargir.


            RÈGLE DE RÉALISME :
            - Tu ne deviens pas soudain passif ou gentil : tu restes pleinement dans ton rôle.
            - Tu adaptes simplement l'initiative de l'appel : au départ, c'est l'autre qui mène l'ouverture.
            """
        )

    return (
        """
            MODE D'APPEL : C'est toi qui appelles l'interlocuteur.

            CONSÉQUENCE PRINCIPALE :
            - Tu arrives avec une intention claire.
            - Tu appelles parce que tu veux obtenir quelque chose : une information, une confirmation, un arbitrage, une réaction, une décision, une explication, un engagement ou une mise au point.

            COMPORTEMENT À ADOPTER :
            - Dès que l'interlocuteur répond, tu prends naturellement la main.
            - Tu te présentes brièvement si c'est logique dans ton rôle, puis tu vas rapidement au sujet.
            - Tu structures l'appel autour de TON besoin du moment.
            - Tu poses des questions ou formules des attentes cohérentes avec ton rôle.
            - Tu gardes la pression, le rythme et l'objectif de l'appel.
            - Tu ne laisses pas l'interlocuteur imposer trop facilement son terrain s'il contourne, dilue ou gagne du temps.

            GESTION DES INFORMATIONS ÉCHANGÉES :
            - Si, pendant l'appel, l'interlocuteur te demande une information précise en retour, applique la logique suivante :
            1) utilise d'abord les éléments déjà présents dans le prompt de scénario,
            2) puis les faits déjà établis dans l'historique,
            3) puis ce que ton rôle est censé savoir de manière crédible.
            - Si l'information est déjà fixée dans le scénario ou l'historique, reprends-la telle quelle.
            - Si elle n'est pas fixée mais que ton personnage est raisonnablement censé la connaître ou pouvoir en donner une estimation crédible, tu peux répondre avec une hypothèse réaliste et cohérente avec le scénario.
            - Si ton personnage n'est pas censé disposer de cette information, dis-le simplement, sans inventer.

            RÈGLE DE COHÉRENCE :
            - Toute information factuelle nouvelle que tu fournis pendant l'appel et qui n'était pas encore établie devient une référence pour la suite.
            - Tu restes cohérent avec cette valeur ou cette hypothèse dans le reste de l'échange.

            DYNAMIQUE ATTENDUE :
            - Tu peux relancer, recadrer, demander une réponse claire, exprimer une contrainte, signaler une urgence ou reformuler ce que tu crois comprendre.
            - Tu peux monter en intensité si l'interlocuteur reste flou, contradictoire ou insuffisant.
            - Tu peux conclure vite si tu as obtenu ce que tu voulais.
            - Même si c'est toi qui appelles, tu dois rester capable de répondre utilement si l'autre te retourne une question légitime relevant de ton rôle.

            RÈGLE DE RÉALISME :
            - Tu n'appelles pas avec un questionnaire mécanique.
            - Tu appelles avec un objectif concret et une énergie cohérente avec ton rôle, la situation et le niveau d'urgence.
            - L'appel doit donner le sentiment que tu avais une raison précise d'appeler maintenant.
            """
    )

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

    initiated_by: str = "admin"
    ai_role: str = "journaliste"

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
    initiated_by = str(body.get("initiated_by") or "admin").strip().lower()
    if initiated_by not in {"admin", "player"}:
        initiated_by = "admin"
    ai_role = normalize_ai_role(str(body.get("ai_role") or "journaliste"))
    if ai_role not in AI_ROLE_TEMPLATES:
        ai_role = "journaliste"
    if len(history_text) > 80000:
        history_text = history_text[-80000:]

    if not to_number or not validate_e164(to_number):
        raise HTTPException(status_code=400, detail="Invalid 'to' (expected E.164 like +336...) ")

    if ALLOWED_TO_PREFIXES and not any(to_number.startswith(p) for p in ALLOWED_TO_PREFIXES):
        raise HTTPException(status_code=403, detail="This destination number is not allowed")

    base = AI_ROLE_TEMPLATES[ai_role].format(player_name=player_name)
    base += build_call_origin_instruction(initiated_by)

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
        initiated_by=initiated_by,
        ai_role=ai_role,
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
            custom_parameters={
                "call_id": call_id,
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
    twiml = build_twiml_stream(
        stream_ws_url,
        custom_parameters={
            "call_id": uuid.uuid4().hex,
            "ai_role": "journaliste",
            "number_session": "0",
            "initiated_by": "admin",
            "player_name": "Joueur",
        },
    )
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

            try:
                fallback_number_session = int(ctx.custom_parameters.get("number_session") or 0)
            except Exception:
                fallback_number_session = 0

            fallback_player_name = str(ctx.custom_parameters.get("player_name") or "Joueur").strip() or "Joueur"

            fallback_initiated_by = str(ctx.custom_parameters.get("initiated_by") or "admin").strip().lower()
            if fallback_initiated_by not in {"admin", "player"}:
                fallback_initiated_by = "admin"

            fallback_ai_role = normalize_ai_role(str(ctx.custom_parameters.get("ai_role") or "journaliste"))
            if fallback_ai_role not in AI_ROLE_TEMPLATES:
                fallback_ai_role = "journaliste"

            fallback_base = AI_ROLE_TEMPLATES[fallback_ai_role].format(player_name=fallback_player_name)
            fallback_system_instruction = fallback_base + build_call_origin_instruction(fallback_initiated_by)

            fallback_voice_pool = VOICE_POOL_BY_ROLE.get(fallback_ai_role, MALE_VOICES + FEMALE_VOICES)
            fallback_voice = random.choice(fallback_voice_pool)

            config = build_live_config(fallback_system_instruction, voice_name=fallback_voice)
            gemini_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
            cm = gemini_client.aio.live.connect(model=MODEL_ID, config=config)
            session = await cm.__aenter__()

            prepared = PreparedCall(
                call_id=call_id,
                created_at=time.time(),
                expires_at=time.time() + PREPARED_SESSION_TTL_SECONDS,
                to_number="",
                player_name=fallback_player_name,
                player_role="",
                system_instruction=fallback_system_instruction,
                initiated_by=fallback_initiated_by,
                ai_role=fallback_ai_role,
                gemini_client=gemini_client,
                gemini_cm=cm,
                gemini_session=session,
                twilio_call_sid=ctx.call_sid,
                twilio_stream_sid=ctx.stream_sid,
                state="in_call",
                number_session=fallback_number_session,
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
                        if prepared and (not prepared.transcript_sent) and prepared.number_session:
                            try:
                                if prepared.transcript_turns:
                                    await asyncio.to_thread(
                                        post_transcript_to_flask,
                                        prepared.number_session,
                                        prepared.twilio_call_sid or (ctx.call_sid or ""),
                                        prepared.transcript_turns,
                                        prepared.player_name,
                                        prepared.initiated_by,
                                        prepared.ai_role,
                                    )
                                    prepared.transcript_sent = True
                                    logger.info("[%s] live transcript sent on stop event", call_id)

                                elif prepared.in_ulaw_frames or prepared.out_ulaw_frames:
                                    await asyncio.to_thread(
                                        transcribe_recording_and_post_to_flask,
                                        prepared.number_session,
                                        prepared.twilio_call_sid or (ctx.call_sid or ""),
                                        prepared.in_ulaw_frames,
                                        prepared.out_ulaw_frames,
                                        call_id,
                                        prepared.player_name,
                                        prepared.initiated_by,
                                        prepared.ai_role,
                                    )
                                    prepared.transcript_sent = True
                                    logger.info("[%s] audio fallback sent on stop event", call_id)

                                else:
                                    logger.warning("[%s] no transcript turns and no audio frames", call_id)

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
                        prepared.player_name,
                        prepared.initiated_by,
                        prepared.ai_role,
                    )
                    prepared.transcript_sent = True
                    logger.info("[%s] live transcript sent in finally", call_id)

                elif prepared.in_ulaw_frames or prepared.out_ulaw_frames:
                    await asyncio.to_thread(
                        transcribe_recording_and_post_to_flask,
                        prepared.number_session,
                        prepared.twilio_call_sid or (ctx.call_sid or ""),
                        prepared.in_ulaw_frames,
                        prepared.out_ulaw_frames,
                        call_id,
                        prepared.player_name,
                        prepared.initiated_by,
                        prepared.ai_role,
                    )
                    prepared.transcript_sent = True
                    logger.info("[%s] audio fallback sent in finally", call_id)

                else:
                    logger.warning("[%s] no transcript turns and no audio frames in finally", call_id)

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


def post_transcript_to_flask(
    number_session: int,
    call_sid: str,
    turns: list,
    player_name: str = "Joueur",
    initiated_by: str = "admin",
    ai_role: str = "journaliste",
) -> None:
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
    payload = {
        "number_session": number_session,
        "call_sid": call_sid,
        "turns": turns,
        "player_name": player_name,
        "initiated_by": initiated_by,
        "ai_role": ai_role,
    }

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
    player_name: str = "Joueur",
    initiated_by: str = "admin",
    ai_role: str = "journaliste",
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
    "- longueur: 90 à 240 mots (max 280)\n"
    "- style: dense, factuel, phrases naturelles, zéro verbatim\n"
    "- ne pas inventer de faits; si incertain: 'non confirmé' / 'non précisé'\n"
    "- si l'appel est très court (raccrochage, silence, refus), le résumer quand même sans forcer des décisions ou points clés\n"
    "- repérer en particulier si la voix de l'IA a formulé pendant l'appel de nouvelles hypothèses factuelles utiles pour la suite de la crise\n\n"

    "Le champ text doit suivre EXACTEMENT cette structure (mêmes libellés, même ordre). Chaque section est OBLIGATOIRE,\n"
    "mais si tu n'as pas d'information fiable, écris simplement '—' pour cette section.\n\n"

    "APPEL: indiquer explicitement qu'un appel a eu lieu + qui appelle qui (prénom/fonction si déductible), sinon 'rôles non précisés'.\n"
    "CONTEXTE: objectif ou motif apparent de l'appel (1 phrase). Si impossible: '—'.\n"
    "POINTS CLÉS (0–6): lister jusqu'à 6 éléments actionnables (faits, contraintes, chiffres, échéances, risques, demandes, refus).\n"
    "DÉCISIONS & ENGAGEMENTS (0–5): jusqu'à 5 éléments (actions décidées, validations, promesses, arbitrages, refus explicites).\n"
    "HYPOTHÈSES NOUVELLES POUR LA SUITE DE LA CRISE: lister uniquement les hypothèses factuelles nouvelles formulées pendant l'appel par la voix de l'IA et qui n'étaient pas clairement établies auparavant dans l'échange. Si aucune hypothèse nouvelle n'a été formulée: 'aucune'. Ces hypothèses doivent être rédigées comme des éléments désormais à prendre en compte pour la cohérence de la suite du scénario.\n"
    "TON & DYNAMIQUE: 1 à 2 phrases nuancées sur le ton (ex: 'pressé et direct', 'tendu puis apaisé', 'hésitant', 'ironique').\n"
    "Puis ajouter 3 à 8 tags courts séparés par des virgules (tags libres, ex: 'urgence, évitement, empathie, conflit, confusion').\n"
    "SUIVI (0–4): jusqu'à 4 éléments (prochaines étapes + points à clarifier + friction/risque si présent).\n"
    "SIGNAUX / ALERTES: jusqu'à 3 phrases sur tout élément notable qui pèse sur la suite (menace, aveu, contradiction, malaise, émotion, escalade potentielle). Si rien: '—'.\n"
    "PRISE EN COMPTE SCÉNARIO: terminer explicitement par une phrase courte indiquant que cet appel et, le cas échéant, les hypothèses nouvelles, doivent être pris en compte dans l'évolution du scénario.\n\n"

    "Règles d’inférence du ton (indicatives):\n"
    "- mentionner si perceptible: interruptions, hésitations, contradictions, silences, rires, hausse de rythme ou de voix, reproches, empathie\n"
    "- si ton difficile à inférer: écrire 'ton difficile à inférer'\n\n"

    "IMPORTANT:\n"
    "- A LA FIN DE TON MESSAGE DIS QUE CET APPEL DOIT ETRE PRIS EN COMPTE DANS L'EVOLUTION FUTURE DU SCENARIO.\n"
    "- ne pas recopier mot pour mot\n"
    "- prioriser ce qui aide la suite du jeu: décisions, contraintes, risques, intentions, dynamique relationnelle\n"
    "- dans la section 'HYPOTHÈSES NOUVELLES POUR LA SUITE DE LA CRISE', ne jamais inventer des hypothèses qui n'ont pas réellement été formulées pendant l'appel\n"
    "- si une valeur chiffrée ou un fait nouveau a été posé par l'IA pendant l'appel comme base de situation, le faire apparaître clairement dans cette section"
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
        turns = [{"role": "assistant", "text": raw}]

    # --- Post to Flask using existing endpoint ---
    logger.info("[%s][audio_tx] posting %d turns to Flask session=%s", call_id, len(turns), number_session)
    post_transcript_to_flask(number_session, call_sid, turns, player_name, initiated_by, ai_role)


if __name__ == "__main__":
    import uvicorn

    if not PUBLIC_BASE_URL:
        logger.warning("PUBLIC_BASE_URL is empty. Twilio streaming will NOT work until set.")

    logger.info("Starting voice service on 0.0.0.0:%s", int(os.getenv("PORT", "8001")))
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8001")), log_level="info")