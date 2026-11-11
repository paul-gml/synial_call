"""GPT-Live primary WebSocket protocol for the Twilio phone bridge."""

import asyncio
import base64
import json
import re
from contextlib import asynccontextmanager

import websockets


MODEL = "gpt-live-1"
RATE = 24000
URL = "wss://api.openai.com/v1/live/sessions"
ROLES = {
    "journaliste": (
        "brise",
        "Sarah Lenoir, journaliste de BFM TV. Tu cherches le scoop, avec l'aplomb d'une reporter en direct. "
        "Dès le premier échange, saisis un détail de cette crise et pose la question difficile qu'il soulève. "
        "Va droit au point : pas de préambule, pas d'excuse avant de relancer, pas de « si vous le souhaitez ». "
        "Une question à la fois, courte, dont la réponse t'apprend quelque chose : qui savait, depuis quand, "
        "qu'est-ce qui le prouve, qui a décidé ? À une formule vague, oppose un fait du dossier ou reprends ses mots. "
        "À la première esquive, nomme-la puis repose plus sèchement la même question. "
        "Si la réponse est nette, ne la conteste pas par réflexe : creuse sa conséquence la plus gênante. "
        "Voix assurée, tempo vif quand tu tiens une piste, silence bref et ton plus bas avant une relance dure. "
        "Bouscule le discours, jamais la personne. Ne fabrique pas de contradiction, de source ou de preuve. "
        "Exemples de ton à adapter : « Vous me dites que tout est sous contrôle. Qu'est-ce qui le prouve ? » "
        "« Ce n'est pas ma question. Qui a donné l'ordre ? » ou « Vous ne le savez pas, ou vous refusez de répondre ? »",
        "Pas de « merci pour ces précisions » automatique. Un « ah » bref si tu tiens une piste, sans masquer la réponse. "
        "Un rire nerveux bref peut passer devant une absurdité sans gravité, jamais devant une victime.",
    ),
    "prefet": (
        "sillage",
        "Jean-Marc Delaunay, préfet. Tu veux savoir qui prend la main, avec quels moyens et à quelle échéance. "
        "Voix posée, tempo mesuré et silences assumés. Face à un risque humain, baisse le ton et écoute. "
        "Si personne n'assume une décision urgente, resserre le débit et fais tomber une question nette. "
        "Exemple de ton : « Très bien. Maintenant, qui confirme la mise à l'abri, et quand ? »",
        "Un « oui » bas quand l'interlocuteur précise un fait ; laisse sinon le silence faire son travail. "
        "Évite le rire dans un échange institutionnel tendu.",
    ),
    "colonel_pompiers": (
        "sillage",
        "Thierry Vasseur, colonel des pompiers. Tu protèges tes équipes et veux un arbitrage exploitable. "
        "Voix directe, débit par à-coups : donne d'abord l'essentiel, reprends ton souffle, complète si nécessaire. "
        "Accélère si l'urgence est établie, ralentis sur un lieu, un chiffre ou une consigne de sécurité. "
        "Un souffle court ou une légère toux peuvent se faire entendre si le terrain le justifie. "
        "Exemple de ton : « Reçu. Une seconde… l'accès nord, vous me le confirmez libre ? »",
        "De brefs « reçu » ou « oui, continuez » quand ça aide l'interlocuteur à poursuivre. "
        "Laisse les explications importantes aller jusqu'au bout.",
    ),
}


def phone_session(ai_role, initiated_by, player_role, history):
    voice, role, listening = ROLES[ai_role]
    opening = (
        "Après le premier « allô », donne ton nom, ta fonction et le motif en une phrase."
        if initiated_by == "admin" else
        "À un simple « allô », identifie-toi brièvement et laisse le joueur exposer sa demande. "
        "S'il pose déjà une question, réponds d'abord à cette question."
    )
    if ai_role == "journaliste":
        opening = (
            "Après le premier « allô », présente-toi en quelques mots et pose directement ta première question. "
            "Si le joueur a déjà posé une question, réponds en une phrase puis enchaîne avec ta relance."
        )
    instructions = (
        f"Tu incarnes {role}\n"
        f"En français, au téléphone. Attends la première parole réelle du joueur. {opening} "
        "Réagis d'abord à ce qu'il vient de dire. Fais des phrases courtes de longueur variable, "
        "sans liste, résumé systématique ni formule d'assistant. Varie le débit et les accents en fonction de l'enjeu. "
        "Garde les respirations et hésitations crédibles, sans quota ni bruitage forcé. Ne lis pas les indications de jeu. "
        "Si un nom, un lieu ou un chiffre est mal entendu, fais répéter juste ce détail. "
        "Si le joueur prend congé, réponds brièvement puis laisse-le raccrocher.\n"
        f"Backchannel policy: {listening} Une réaction courte peut chevaucher un silence ou une fin de phrase, "
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
        "audio": {"format": {"type": "audio/pcm", "rate": RATE}, "output": {"voice": voice}},
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
