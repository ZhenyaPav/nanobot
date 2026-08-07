# Backend Emotion Classification

nanobot can classify every completed assistant response with the same
`Cohee/distilbert-base-uncased-go-emotions-onnx` model used by SillyTavern. Classification runs
behind a small local HTTP service so ONNX Runtime and the model do not increase the gateway's
default dependency or memory footprint.

Install and start the service:

```bash
pip install 'nanobot-ai[emotion]'
python -m nanobot.emotions.service
```

The commands are identical in Bash and Fish.

Enable it in `~/.nanobot/config.json`:

```json
{
  "emotionClassification": {
    "enabled": true,
    "endpoint": "http://127.0.0.1:5101/classify",
    "topK": 5,
    "timeout": 10,
    "fallbackLabel": "neutral"
  }
}
```

The gateway strips quotes and asterisks and applies SillyTavern's 500-character sampling rule
before calling the service. The service returns the five highest-scoring labels, sorted by
score. The winning label, score, and candidates are persisted on the assistant message and
sent in outbound metadata under `_emotion`:

```json
{
  "_emotion": {
    "label": "joy",
    "score": 0.82,
    "classification": [
      {"label": "joy", "score": 0.82}
    ]
  }
}
```

This metadata is channel-agnostic. The current WebUI and Signal channel ignore unknown metadata;
a future dynamic-avatar UI can consume `_emotion.label` without changing classification logic.

For a network-accessible classifier, start the service with `--api-key`, use the same value in
`emotionClassification.apiKey`, and put TLS or a trusted reverse proxy in front of it.
