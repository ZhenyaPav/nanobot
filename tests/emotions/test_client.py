from __future__ import annotations

from nanobot.config.schema import EmotionClassificationConfig
from nanobot.emotions.client import EmotionClassifierClient, sample_classification_text


def test_sample_classification_text_matches_sillytavern_limit() -> None:
    text = '"Hello." ' + "x" * 600 + " Last sentence!"
    sampled = sample_classification_text(text)

    assert '"' not in sampled
    assert len(sampled) <= 501
    assert sampled.startswith("Hello.")
    assert sampled.endswith("Last sentence!")


def test_parse_classification_sorts_scores() -> None:
    client = EmotionClassifierClient(EmotionClassificationConfig(enabled=True, top_k=2))

    result = client._parse(  # pyright: ignore[reportPrivateUsage]
        {
            "classification": [
                {"label": "sadness", "score": 0.2},
                {"label": "joy", "score": 0.8},
                {"label": "neutral", "score": 0.1},
            ]
        }
    )

    assert result is not None
    assert result.label == "joy"
    assert [row["label"] for row in result.classification] == ["joy", "sadness"]
