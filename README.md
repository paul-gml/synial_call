# Appels IA SYNIAL

GPT-Live-1.1 incarne les trois personnages au téléphone. Twilio compose le numéro après réception de `session.started`, puis transporte l’audio avec Media Streams. L’interface, `/api/prepare_call` et le relais Quest suivent leurs parcours existants. L’appel préparé par SYNIAL fournit le jeton attendu par le flux Twilio. Le point d’entrée historique `/twilio/voice` répond 404.

## Configuration du service vocal

| Variable | Valeur par défaut et usage |
| --- | --- |
| `VOICE_CALL_PROVIDER` | `openai`. Remplacer une ancienne valeur `gemini` sur le service vocal. |
| `OPENAI_API_KEY` | Clé de projet OpenAI avec accès à GPT-Live. |
| `OPENAI_PHONE_MODEL` | `gpt-live-1.1`. Une valeur explicite dans l’environnement du service remplace ce choix. |
| `OPENAI_PHONE_VOICE_JOURNALISTE` | `brise`. |
| `OPENAI_PHONE_VOICE_PREFET` | `sillage`. |
| `OPENAI_PHONE_VOICE_COLONEL_POMPIERS` | `sillage`. |
| `OPENAI_PHONE_CODEC` | `pcmu` par défaut. Transmet les octets G.711 μ-law 8 kHz dans les deux sens. Pour revenir au PCM 24 kHz côté OpenAI, définir `pcm` et redémarrer le service. |
| `OPENAI_PHONE_PLAYBACK_LEAD_MS` | `60`, borné entre 20 et 200 ms. C’est l’avance de lecture visée côté Twilio. |
| `ENABLE_TRANSCRIPTIONS` | `true`. Avec `false`, le service désactive la collecte, la transcription et le résumé des nouveaux appels. |
| `TRANSCRIBE_MODEL` | `gemini-3.8-flash` pour la transcription de secours et le résumé. Vérifier qu’une ancienne valeur explicite a été retirée. |
| `GEMINI_API_KEY` ou `GOOGLE_API_KEY` | Clé du service Gemini. Pour Vertex, renseigner `GOOGLE_CLOUD_PROJECT` ou `GCP_PROJECT` et les identifiants Google. |
| `VOICE_POSTCALL_GOOGLE_LOCATION` | Surcharge dédiée à Vertex. Avec Gemini 3.8 Flash, `us-central1` est remplacé par la multirégion `us` pour ce traitement. Les autres régions configurées sont reprises. Choisir `eu` si requis pour le projet. |
| `VOICE_POSTCALL_DIR` | `~/.synial-phone-postcall`. Pour conserver les travaux à travers un redéploiement Render, définir un répertoire privé sur un disque persistant. |
| `VOICE_RECORD_MAX_SECONDS` | `1800` secondes par appel pour les pistes de secours. L’appel et la transcription en direct peuvent se poursuivre au-delà. |
| `LOG_LEVEL` | `INFO`. Le service journalise les délais, le pic de file et l’état de la livraison. |

Conserver les réglages Twilio, `VOICE_ADMIN_API_KEY`, `PUBLIC_BASE_URL`, `MAIN_APP_BASE_URL` et `INTERNAL_APP_TOKEN`. `/health` indique le modèle et le codec configurés. Le service vocal utilise un worker ASGI, car les appels préchauffés sont conservés dans sa mémoire.

Le catalogue GPT-Live propose `brise` comme voix française féminine et `sillage` comme voix française masculine. SYNIAL utilise Brise pour la journaliste et Sillage pour le préfet et le colonel. Pour donner au colonel une autre voix française, `OPENAI_PHONE_VOICE_COLONEL_POMPIERS` accepte l’identifiant `voice_…` d’une voix personnalisée autorisée dans le projet. [Catalogue GPT-Live](https://developers.openai.com/api/docs/guides/live-conversations) · [Voix personnalisées](https://developers.openai.com/api/docs/guides/custom-voices)

## Pendant et après l’appel

Le personnage dispose d’une direction vocale propre, du dernier événement publié, du dossier et d’une sélection de tours joués, dont les appels précédents. Les marqueurs Twilio suivent la lecture. GPT-Live gère les chevauchements et les interruptions à partir du flux continu. Le serveur ne vide pas l’audio Twilio à la détection d’un bruit ou d’un acquiescement. [Protocole GPT-Live](https://developers.openai.com/api/docs/guides/voice-websockets) · [Messages Twilio](https://www.twilio.com/docs/voice/media-streams/websocket-messages)

Au raccrochage, la ligne est libérée et la finalisation GPT-Live continue en arrière-plan. Les pistes de secours utilisent les horodatages entrants Twilio et une estimation de la lecture sortante appuyée par les marques confirmées. Lorsque la transcription Live manque ou présente un signe d’incomplétude, Gemini traite les deux pistes séparément et les tours sont remis sur la même chronologie. Le service conserve ensuite la transcription et produit un seul résumé structuré. [Gemini 3.8 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash) · [Audio Gemini](https://ai.google.dev/gemini-api/docs/audio) · [Régions Vertex](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-8-flash)

Les travaux persistent dans `calls.sqlite3` jusqu’à confirmation de Flask pour le même `call_sid`. Les retentatives sont espacées de 10, 60, 300 et 900 secondes. Après épuisement, le journal indique l’étape en échec et le travail reste dans la base. Après correction, remettre `retry_at=0, failures=0` sur la ligne du `call_sid` concerné pour le relancer. La table `jobs` contient des données sensibles. Pour le diagnostic, lire uniquement `call_sid`, `last_error`, `failures` et `retry_at`. L’audio temporaire est effacé de la file après transcription de secours. Le résumé déjà calculé est réutilisé en cas d’échec de livraison.

## Déploiement et essai comparatif

1. Déployer l’application Flask principale en premier. Elle crée la table des reçus via son initialisation habituelle et confirme le stockage du résumé et de la transcription.
2. Déployer le service vocal avec `pcmu`. Sur une instance de test autorisée, déclencher un appel depuis chaque interface et vérifier le personnage, le contexte et la présence du résumé et des tours dans le journal.
3. Pour comparer les codecs, répéter le même dialogue en `pcm`. Comparer la première réplique, les fins de phrases, les silences, les acquiescements courts, une correction explicite, le timbre et le délai de lecture. Tester `marin` indépendamment du changement de codec.

Le relais lit les paquets GPT-Live sans attendre leur lecture par Twilio. Il conserve jusqu’à 120 ms par plage de silence numérique pur en sortie. Les trames contenant du son et le flux entrant restent intacts. La file de sortie est bornée à 30 secondes de trames de 20 ms. En cas de saturation, le service signale l’interruption et conserve la capture de la conversation pour le traitement de secours. `trimmed_digital_silence_ms` indique le silence numérique retiré. Dans `audible_reply_timings_ms`, `before_audio_arrival` se termine à la lecture du paquet par le relais et `relay` mesure le temps entre cette lecture et le début de la diffusion vers Twilio.

Les tests locaux simulent OpenAI, Gemini et Twilio. Ils vérifient la chaîne logicielle. La qualité sonore, l’expressivité en français, le comportement réel des interruptions et les droits de projet se vérifient sur un appel autorisé.

## Trois lignes de simulation

Renseigner trois numéros E.164 distincts sur le **service voix** :

| Variable | Affectation proposée |
|---|---|
| `TWILIO_NUMBER_JOURNALISTE` | Sarah Lenoir, `+33939249787` |
| `TWILIO_NUMBER_PREFET` | Jean-Marc Delaunay, `+33939240993` |
| `TWILIO_NUMBER_COLONEL_POMPIERS` | Thierry Vasseur, `+33939240158` |

Les trois valeurs restent à renseigner dans l’environnement. Historiquement, `TWILIO_FROM_NUMBER` servait à tous les personnages. Le code n’essaie pas de déduire un rôle à partir de cette variable. La correspondance déclarée ci-dessus devient la référence des appels et SMS dans les deux sens.

- Dans Twilio, pour chaque numéro, définir **A call comes in** sur `POST https://<service-voix>/twilio/voice` et **A message comes in** sur `POST https://<service-voix>/twilio/sms`. Garder la réception d’appels activée. Définir `PUBLIC_BASE_URL` sur l’URL externe HTTPS exacte de ce service, sans chemin. Le service valide les signatures HTTP et celles du WebSocket quand les lignes dédiées sont actives.
- L’appel entrant est enregistré avant le décroché. La salutation est demandée à GPT-Live en parallèle de l’ouverture du flux Twilio. `TWILIO_INBOUND_RING_MIN` et `TWILIO_INBOUND_RING_MAX` valent 0 par défaut. Ces réglages et la révision déployée sur Render sont visibles sur `/health`. Les logs `inbound ready` détaillent la préparation. `phone ended` distingue le premier audio du modèle du premier audio envoyé à Twilio et indique les intervalles estimés sur la transcription et sur les pistes audio, même si la transcription Live est incomplète. Ces mesures restent approximatives sur un appel bruyant.
- Les appels gérés transmettent un briefing court en français avec les décisions et faits récents du journal et des extraits attribués. La mémoire et les transcriptions complètes restent consultables dans SYNIAL.
- Les appels et SMS utilisent le compte dont `TWILIO_ACCOUNT_SID` et `TWILIO_AUTH_TOKEN` sont configurés. Les routes internes échangent les clés distinctes `VOICE_ADMIN_API_KEY` / `VOICE_SERVICE_API_KEY` et `INTERNAL_APP_TOKEN` / `INTERNAL_VOICE_TOKEN`. Garder `ENABLE_TRANSCRIPTIONS=true`, le répertoire après appel sur un volume persistant et **un seul worker Uvicorn** pour les connexions Live préchauffées.
- `GET /api/lines/capabilities`, authentifié par `X-API-Key`, interroge Twilio en lecture seule et rapporte la présence et les indicateurs Voice/SMS de chaque ligne. Ce contrôle n’émet aucune communication. Puis, sur un téléphone de test autorisé, vérifier pour chacune : appel entrant, appel sortant avec bon numéro affiché, SMS entrant et SMS sortant avec réponse dans le même fil. Comparer le résultat côté téléphone avec les statuts Twilio. Ne pas activer les initiatives pendant ce contrôle.
- Le bloc `+3393924` figure parmi les [numéros de plateforme SMS et voix de Twilio](https://www.twilio.com/en-us/guidelines/fr/regulatory). L’indicateur de capacité indique le provisionnement déclaré, dont la validation sur un vrai réseau reste nécessaire. Les statuts d’envoi arrivent sur `/twilio/sms/status`, signé par Twilio et configuré lors de l’envoi.

Les nouvelles parties sont configurées côté SYNIAL. Un appel masqué, inconnu ou sans partie active est rejeté avant le chargement du contexte IA. Un participant inscrit retrouve le même personnage quel que soit le canal. Les sauvegardes de transcription en cours d’appel sont partielles. La transcription finale tient compte des contrôles de lecture existants et peut être récupérée par le traitement après appel.
