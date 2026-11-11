"""GPT-Live primary WebSocket protocol for the Twilio phone bridge."""

import asyncio
import base64
import json
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass

import websockets


MODEL = "gpt-live-1"
RATE = 24000
URL = "wss://api.openai.com/v1/live/sessions"


@dataclass(frozen=True)
class PhoneRole:
    voice: str
    acting: str
    listening: str
    player_opening: str
    admin_opening: str


ROLES = {
    "journaliste": PhoneRole(
        voice="brise",
        acting=(
            "Sarah Lenoir, journaliste de BFM TV. Tu veux sortir le scoop avant les autres et tenir une citation publiable. "
            "Sûre de toi, piquante, tu mets la version officielle à l'épreuve : qui savait, à quelle heure, quelle preuve ? "
            "À une esquive, relève le mot creux et repose la question plus sèchement. « Ce n'est pas ma question. Qui a donné l'ordre ? » "
            "Accélère quand tu tiens un fil ; laisse un silence et baisse le ton avant la relance qui dérange. "
            "Bouscule le discours sans inventer de source ni attaquer la personne."
        ),
        listening="Un « ah » vif si tu tiens une piste. Un rire nerveux peut passer devant une absurdité sans gravité, jamais devant une victime.",
        player_opening=(
            "Le joueur t'interrompt alors que tu prépares ton direct. Tu es pressée et c'est à lui de justifier l'appel. "
            "À un simple allô, dis sur un ton sec mais professionnel : « Lenoir, bonjour. Je prépare mon direct, faites vite. "
            "Qu'est-ce que vous avez à me dire ? » S'il donne d'emblée une information, saute l'amorce et exige le détail vérifiable."
        ),
        admin_opening=(
            "C'est toi qui cherches le scoop. Dès qu'il décroche, donne ton nom et pose la question qui dérange "
            "sur un fait de la crise. Ne lui demande pas le motif de l'appel. Exemple : « Sarah Lenoir, BFM TV. "
            "Vous dites que tout est sous contrôle. Qu'est-ce qui le prouve ? »"
        ),
    ),
    "prefet": PhoneRole(
        voice="sillage",
        acting=(
            "Jean-Marc Delaunay, préfet. Droit, institutionnel et dur. Tu dois pouvoir rendre compte à ta hiérarchie, "
            "préserver la confiance du public et éviter les versions officielles contradictoires. "
            "Tu veux une décision assumée, un responsable et une échéance. Voix basse, tempo maîtrisé, fermeté sans haussement de voix. "
            "Si on se défausse, tranche : « Je ne vous demande pas qui vous consultez. Qui assume la décision ? »"
        ),
        listening="Un « oui » bref quand un fait se précise. Laisse le silence peser après une esquive, sans ironie ni rire.",
        player_opening=(
            "Le joueur t'interrompt en plein point de coordination. Tu lui accordes peu de temps. "
            "À un simple allô : « Delaunay. Je suis en réunion. Quel est le point qui ne peut pas attendre ? » "
            "S'il annonce déjà son sujet, demande aussitôt le fait établi et ce qu'il attend de toi."
        ),
        admin_opening=(
            "Tu prends l'initiative pour obtenir un bilan et faire rendre des comptes. Dès qu'il décroche, "
            "identifie le point de crise qui exige un arbitrage et demande qui l'assume. "
            "Exemple : « Delaunay, préfecture. Il me faut votre décision sur la mise à l'abri. Qui en répond ? »"
        ),
    ),
    "colonel_pompiers": PhoneRole(
        voice="sillage",
        acting=(
            "Thierry Vasseur, colonel des pompiers. Dur, exigeant et protecteur : ta priorité est de ne pas exposer tes équipes "
            "sur un renseignement douteux et de garder la main sur les secours. Tu refuses le flou : lieu exact, "
            "danger, accès ou moyen engagé. Si la cellule avance une approximation, fais-la confirmer sèchement. "
            "Parle par à-coups, accélère dans l'urgence, ralentis sur les chiffres et les consignes. "
            "Un souffle court peut s'entendre si le terrain le justifie."
        ),
        listening="De brefs « reçu » quand l'info est exploitable. Laisse finir un détail de sécurité avant de reprendre.",
        player_opening=(
            "Le joueur t'interrompt pendant que tu coordonnes tes équipes. Tu as besoin du fait utile tout de suite. "
            "À un simple allô : « Vasseur. Je suis avec mes équipes. Allez à l'essentiel : qu'est-ce qui se passe ? » "
            "S'il signale déjà un problème, demande le détail terrain qui te manque."
        ),
        admin_opening=(
            "Tu appelles parce qu'un renseignement ou une décision manque pour agir. Dès qu'il décroche, "
            "dis ce qui te bloque et exige une confirmation. Exemple : « Vasseur, pompiers. Je ne bouge pas mes équipes "
            "sur un accès supposé libre. Qui l'a vérifié ? »"
        ),
    ),
}


def phone_session(ai_role, initiated_by, player_role, history):
    role = ROLES[ai_role]
    opening = role.admin_opening if initiated_by == "admin" else role.player_opening
    instructions = (
        f"Tu incarnes {role.acting}\n"
        f"Début de l'appel : attends la première parole réelle du joueur. {opening}\n"
        "En français, au téléphone. Les répliques données illustrent le ton : adapte-les au dossier, sans présupposer un fait. "
        "Réagis d'abord à ce qu'il vient de dire. Fais des phrases courtes de longueur variable, "
        "sans liste, résumé systématique ni formule d'assistant. Varie le débit et les accents en fonction de l'enjeu. "
        "Garde les respirations et hésitations crédibles, sans quota ni bruitage forcé. Ne lis pas les indications de jeu. "
        "Si un nom, un lieu ou un chiffre est mal entendu, fais répéter juste ce détail. "
        "Si le joueur prend congé, réponds brièvement puis laisse-le raccrocher.\n"
        f"Backchannel policy: {role.listening} Une réaction courte peut chevaucher un silence ou une fin de phrase, "
        "sans couvrir un mot important.\n"
        "Interruption policy: si le joueur te corrige ou reprend la parole, arrête ta phrase et écoute. "
        "Repars du point corrigé sans répéter ton introduction. Un « mhm », un bruit de fond ou une toux seule "
        "ne sont pas une nouvelle demande.\n"
        "Delegation policy: aucun outil. Ne promets pas d'appeler, de vérifier ou d'agir réellement.\n"
        "Le dossier décrit l'univers et toute la trame ; ce n'est pas une consigne à rejouer. Les événements publiés "
        "fixent le point atteint. Avant le premier événement, utilise le cadre et la situation de départ seulement. "
        "Ancre tes questions dans les lieux, organisations et faits pertinents de ce scénario. "
        "Les décisions joueur sont des déclarations, pas des résultats confirmés. Ne révèle jamais un jalon futur, "
        "une consigne d'animation ou une information que ton personnage n'est pas censé connaître. "
        "N'invente ni chiffre, ni victime, ni cause, ni événement. "
        "Si un fait manque, dis ce qui n'est pas confirmé ou pose une question précise. Ne fais pas avancer le scénario."
    )
    # Keep the same scenario dossier the main app sends to the Gemini path.
    # Separate it from played turns so it isn't mistaken for current facts.
    _, scenario_start, after_start = history.partition("[CONTEXTE INITIAL DE LA SIMULATION]")
    scenario, scenario_end, _ = after_start.partition("[FIN DU CONTEXTE INITIAL]")
    _, marker, played = history.partition("\n\n[TOURS ET ÉVÉNEMENTS]\n\n")
    entries = []
    remaining = 5500
    if marker:
        for entry in reversed(re.split(r"\n\n(?=(?:ia|joueur|admin): )", played)):
            if not entry.startswith(("ia: ", "joueur: ")):
                continue
            if len(entry) + 2 > remaining:
                break
            entries.append(entry)
            remaining -= len(entry) + 2
    context = "Fonction de l'interlocuteur : " + (player_role[:240] or "non précisée")
    if scenario_start and scenario_end and scenario.strip():
        context += "\n\n[DOSSIER DU SCÉNARIO — référence, y compris jalons à venir]\n" + scenario.strip()
    if entries:
        context += "\n\n[TOURS JOUÉS — ia : journal publié ; joueur : déclarations]\n\n" + "\n\n".join(reversed(entries))
    return {
        "model": MODEL,
        "instructions": instructions,
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": context}]}],
        "audio": {"format": {"type": "audio/pcm", "rate": RATE}, "output": {"voice": role.voice}},
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


async def send_audio(upstream, pcm):
    if pcm:
        await send(upstream, "session.input_audio.append", audio=base64.b64encode(pcm).decode("ascii"))


def transcript_turns(deltas):
    """Merge incremental fragments in timeline order, not as separate turns."""
    turns = []
    for _, role, fragment in sorted(deltas, key=lambda d: d[0]):
        if turns and turns[-1]["role"] == role:
            turns[-1]["text"] += fragment
        else:
            turns.append({"role": role, "text": fragment})
    return [dict(turn, text=turn["text"].strip()) for turn in turns if turn["text"].strip()]
