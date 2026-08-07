"""Client and SillyTavern-compatible preprocessing for emotion classification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import httpx
from loguru import logger

if TYPE_CHECKING:
    from nanobot.config.schema import EmotionClassificationConfig

_SAMPLE_THRESHOLD = 500
_HALF_SAMPLE_THRESHOLD = _SAMPLE_THRESHOLD // 2
_END_SENTENCE_RE = re.compile(r"^(.+?[.!?])(?:\s|$)", re.DOTALL)
_START_SENTENCE_RE = re.compile(r"(?:^|[.!?]\s+)([^.!?].*)$", re.DOTALL)


@dataclass(frozen=True)
class EmotionClassification:
    label: str
    score: float
    classification: list[dict[str, Any]]

    def metadata(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "score": self.score,
            "classification": self.classification,
        }


def sample_classification_text(text: str) -> str:
    """Match SillyTavern's BERT sampling: strip markup and cap input at 500 chars."""
    result = text.replace("*", "").replace('"', "")
    if len(text) < _SAMPLE_THRESHOLD:
        return _trim_to_end_sentence(result).strip()
    start = _trim_to_end_sentence(result[:_HALF_SAMPLE_THRESHOLD])
    end = _trim_to_start_sentence(result[-_HALF_SAMPLE_THRESHOLD:])
    return f"{start} {end}".strip()


def _trim_to_end_sentence(text: str) -> str:
    matches = list(re.finditer(r"[.!?](?=\s|$)", text))
    return text[: matches[-1].end()] if matches else text


def _trim_to_start_sentence(text: str) -> str:
    match = re.search(r"[.!?]\s+", text)
    return text[match.end():] if match else text


class EmotionClassifierClient:
    """Fail-open client for an external Cohee ONNX classification service."""

    def __init__(self, config: EmotionClassificationConfig) -> None:
        self.config = config

    async def classify(self, text: str) -> EmotionClassification | None:
        if not self.config.enabled or not text.strip():
            return None
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout, trust_env=False) as client:
                response = await client.post(
                    self.config.endpoint,
                    headers=headers,
                    json={
                        "text": sample_classification_text(text),
                        "top_k": self.config.top_k,
                    },
                )
                response.raise_for_status()
                payload = response.json()
            result = self._parse(payload)
            if result is not None:
                return result
            raise ValueError("response did not contain classification results")
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            logger.warning("Emotion classification failed: {}", exc)
            if not self.config.fallback_label:
                return None
            return EmotionClassification(
                label=self.config.fallback_label,
                score=0.0,
                classification=[],
            )

    def _parse(self, payload: object) -> EmotionClassification | None:
        if not isinstance(payload, dict):
            return None
        raw = cast(dict[str, object], payload).get("classification")
        if not isinstance(raw, list):
            return None
        rows: list[dict[str, Any]] = []
        for item in cast(list[object], raw):
            if not isinstance(item, dict):
                continue
            data = cast(dict[str, object], item)
            label = data.get("label")
            score = data.get("score")
            if isinstance(label, str) and isinstance(score, int | float):
                rows.append({"label": label.lower(), "score": float(score)})
        rows.sort(key=lambda item: cast(float, item["score"]), reverse=True)
        if not rows:
            return None
        return EmotionClassification(
            label=cast(str, rows[0]["label"]),
            score=cast(float, rows[0]["score"]),
            classification=rows[: self.config.top_k],
        )
