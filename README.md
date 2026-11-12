# Appels IA SYNIAL

GPT-Live-1 incarne les trois personnages au téléphone. Twilio compose le numéro après réception de `session.started`, puis transporte l’audio avec Media Streams. L’interface, `/api/prepare_call` et le relais Quest suivent leurs parcours existants. L’appel préparé par SYNIAL fournit le jeton attendu par le flux Twilio. Le point d’entrée historique `/twilio/voice` répond 404.

## Configuration du service vocal

| Variable | Valeur par défaut et usage |
| --- | --- |
| `VOICE_CALL_PROVIDER` | `openai`. Remplacer une ancienne valeur `gemini` sur le service vocal. |
| `OPENAI_API_KEY` | Clé de projet OpenAI avec accès à GPT-Live. |
| `OPENAI_PHONE_MODEL` | `gpt-live-1`. |
| `OPENAI_PHONE_VOICE_JOURNALISTE` | `brise`. |
| `OPENAI_PHONE_VOICE_PREFET` | `sillage`. |
| `OPENAI_PHONE_VOICE_COLONEL_POMPIERS` | `sillage`. |
| `OPENAI_PHONE_CODEC` | `pcm` pour le PCM 24 kHz côté OpenAI. `pcmu` transmet les octets G.711 μ-law 8 kHz dans les deux sens. Revenir à `pcm` s’applique aux appels suivants après redémarrage. |
| `OPENAI_PHONE_PLAYBACK_LEAD_MS` | `60`, borné entre 20 et 200 ms. C’est l’avance de lecture visée côté Twilio. |
| `OUT_QUEUE_MAX_FRAMES` | `25`, environ 500 ms à 20 ms par trame. La file applique une contre-pression. Une saturation durable ou un envoi bloqué termine proprement l’appel et active le traitement de secours si des paroles ont été capturées. |
| `ENABLE_TRANSCRIPTIONS` | `true`. Avec `false`, le service désactive la collecte, la transcription et le résumé des nouveaux appels. |
| `TRANSCRIBE_MODEL` | `gemini-3.8-flash` pour la transcription de secours et le résumé. Vérifier qu’une ancienne valeur explicite a été retirée. |
| `GEMINI_API_KEY` ou `GOOGLE_API_KEY` | Clé du service Gemini. Pour Vertex, renseigner `GOOGLE_CLOUD_PROJECT` ou `GCP_PROJECT` et les identifiants Google. |
| `VOICE_POSTCALL_GOOGLE_LOCATION` | Surcharge dédiée à Vertex. Avec Gemini 3.8 Flash, `us-central1` est remplacé par la multirégion `us` pour ce traitement. Les autres régions configurées sont reprises. Choisir `eu` si requis pour le projet. |
| `VOICE_POSTCALL_DIR` | `~/.synial-phone-postcall`. Pour conserver les travaux à travers un redéploiement Render, définir un répertoire privé sur un disque persistant. |
| `VOICE_RECORD_MAX_SECONDS` | `1800` secondes par appel pour les pistes de secours. L’appel et la transcription en direct peuvent se poursuivre au-delà. |
| `LOG_LEVEL` | `INFO`. Le service journalise les délais, le pic de file et l’état de la livraison. |

Conserver les réglages Twilio, `VOICE_ADMIN_API_KEY`, `PUBLIC_BASE_URL`, `MAIN_APP_BASE_URL` et `INTERNAL_APP_TOKEN`. `/health` indique le modèle et le codec configurés. Le service vocal utilise un worker ASGI, car les appels préchauffés sont conservés dans sa mémoire.

Les voix françaises `brise` et `sillage` figurent au catalogue GPT-Live. `marin` peut servir de voix témoin. Chaque variable de voix accepte aussi l’identifiant `voice_…` d’une voix autorisée dans le projet. Le rendu en français d’une voix personnalisée reste à tester. [Catalogue GPT-Live](https://developers.openai.com/api/docs/guides/live-conversations) · [Voix personnalisées](https://developers.openai.com/api/docs/guides/custom-voices)

## Pendant et après l’appel

Le personnage dispose d’une direction vocale propre, du dernier événement publié, du dossier et d’une sélection de tours joués, dont les appels précédents. Les marqueurs Twilio suivent la lecture. GPT-Live gère les chevauchements et les interruptions à partir du flux continu. Le serveur ne vide pas l’audio Twilio à la détection d’un bruit ou d’un acquiescement. [Protocole GPT-Live](https://developers.openai.com/api/docs/guides/voice-websockets) · [Messages Twilio](https://www.twilio.com/docs/voice/media-streams/websocket-messages)

Au raccrochage, la ligne est libérée et la finalisation GPT-Live continue en arrière-plan. Les pistes de secours utilisent les horodatages entrants Twilio et une estimation de la lecture sortante appuyée par les marques confirmées. Lorsque la transcription Live manque ou présente un signe d’incomplétude, Gemini traite les deux pistes séparément et les tours sont remis sur la même chronologie. Le service conserve ensuite la transcription et produit un seul résumé structuré. [Gemini 3.8 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash) · [Audio Gemini](https://ai.google.dev/gemini-api/docs/audio) · [Régions Vertex](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-8-flash)

Les travaux persistent dans `calls.sqlite3` jusqu’à confirmation de Flask pour le même `call_sid`. Les retentatives sont espacées de 10, 60, 300 et 900 secondes. Après épuisement, le journal indique l’étape en échec et le travail reste dans la base. Après correction, remettre `retry_at=0, failures=0` sur la ligne du `call_sid` concerné pour le relancer. La table `jobs` contient des données sensibles. Pour le diagnostic, lire uniquement `call_sid`, `last_error`, `failures` et `retry_at`. L’audio temporaire est effacé de la file après transcription de secours. Le résumé déjà calculé est réutilisé en cas d’échec de livraison.

## Déploiement et essai comparatif

1. Déployer l’application Flask principale en premier. Elle crée la table des reçus via son initialisation habituelle et confirme le stockage du résumé et de la transcription.
2. Déployer le service vocal avec `pcm`. Sur une instance de test autorisée, déclencher un appel depuis chaque interface et vérifier le personnage, le contexte et la présence du résumé et des tours dans le journal.
3. Répéter le même dialogue en `pcmu`. Comparer la première réplique, les fins de phrases, les silences, les acquiescements courts, une correction explicite, le timbre et le délai de lecture. Tester `marin` indépendamment du changement de codec.

Les tests locaux simulent OpenAI, Gemini et Twilio. Ils vérifient la chaîne logicielle. La qualité sonore, l’expressivité en français, le comportement réel des interruptions et les droits de projet se vérifient sur un appel autorisé.
