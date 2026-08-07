from nanobot.config.schema import Config


def test_emotion_classification_accepts_and_serializes_camel_case() -> None:
    config = Config.model_validate(
        {
            "emotionClassification": {
                "enabled": True,
                "topK": 3,
                "fallbackLabel": "neutral",
            }
        }
    )

    assert config.emotion_classification.enabled is True
    assert config.emotion_classification.top_k == 3
    dumped = config.model_dump(by_alias=True)
    assert dumped["emotionClassification"]["topK"] == 3


def test_emotion_classification_accepts_snake_case() -> None:
    config = Config.model_validate(
        {"emotion_classification": {"enabled": True, "top_k": 2}}
    )

    assert config.emotion_classification.enabled is True
    assert config.emotion_classification.top_k == 2
