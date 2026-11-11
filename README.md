# Appels IA SYNIAL

Les appels sortants du bouton « Demander un appel » utilisent par défaut `gpt-live-1`. Twilio continue à composer le numéro et à transporter le son avec Media Streams. Le service ne déclenche Twilio qu'après confirmation `session.started` d'OpenAI. L'interface joueur, le contrat `/api/prepare_call` et le relais Quest restent compatibles.

Sur le **service vocal** Render, configurez `OPENAI_API_KEY` dans les variables d'environnement avec une clé de projet valide. `VOICE_CALL_PROVIDER=openai` est la valeur par défaut. Gardez les réglages Twilio, `VOICE_ADMIN_API_KEY`, `PUBLIC_BASE_URL`, `MAIN_APP_BASE_URL` et `INTERNAL_APP_TOKEN` existants. Ne mettez la clé ni dans le dépôt ni dans le client. Le démarrage du service seul ne vérifie pas que la clé donne accès à GPT-Live ; `/health` indique uniquement si la configuration est renseignée.

Pour une démo, déployez d'abord sur une instance de test. Appelez un numéro de test en français depuis le parcours joueur, vérifiez les interruptions, la voix et le retour de la transcription dans le journal. Répétez depuis l'administration. Le test local simule les fournisseurs et ne valide pas la qualité vocale, la facturation ni les droits du projet OpenAI.

Le mode OpenAI exige un appel préparé par SYNIAL : l'ancien point d'entrée `/twilio/voice` pour les appels directs n'y est pas activé. Pour revenir volontairement à l'ancien parcours, utilisez `VOICE_CALL_PROVIDER=gemini` et conservez sa configuration Vertex. Il n'y a pas de bascule automatique vers un autre fournisseur.

Le prompt GPT-Live reprend les trois personnages et les tours récents, sans envoyer la trame future. La transcription en direct est envoyée au journal comme avant, sans enregistrer une copie audio pour une retranscription Gemini. Références : [OpenAI Docs — rédaction GPT-Live](https://developers.openai.com/api/docs/guides/live-prompting), [WebSocket vocal](https://developers.openai.com/api/docs/guides/voice-websockets).

Chaque rôle a sa direction vocale : curiosité et relances ciblées pour la journaliste, tempo maîtrisé pour le préfet, débit de terrain pour le colonel. Le pont OpenAI lisse les rafales en gardant environ 80 ms d'avance côté Twilio, sans retarder les premières trames. Media Streams reste en µ-law à 8 kHz. Si `TWILIO_SAY_BEFORE_STREAM` contient une annonce robotique facultative, laissez-la vide pour entendre directement le personnage ; conservez toute annonce utile aux participants.
