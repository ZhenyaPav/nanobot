"""Standalone ONNX emotion-classification service used by nanobot.

Run with ``python -m nanobot.emotions.service`` after installing the
``nanobot-ai[emotion]`` extra. The model is downloaded from Hugging Face on
first use and then cached locally.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, cast

MODEL_ID = "Cohee/distilbert-base-uncased-go-emotions-onnx"
MODEL_FILE = "onnx/model_quantized.onnx"
_CACHE_SIZE = 512


class CoheeEmotionModel:
    """Lazy, concurrency-safe wrapper around the quantized Cohee ONNX model."""

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self._lock = asyncio.Lock()
        self._session: Any = None
        self._tokenizer: Any = None
        self._labels: list[str] = []
        self._cache: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    async def classify(self, text: str, top_k: int = 5) -> list[dict[str, Any]]:
        cached = self._cache.get(text)
        if cached is not None:
            self._cache.move_to_end(text)
            return list(cached[:top_k])
        await self.ensure_loaded()
        result = await asyncio.to_thread(self._infer, text)
        self._cache[text] = result
        self._cache.move_to_end(text)
        while len(self._cache) > _CACHE_SIZE:
            self._cache.popitem(last=False)
        return list(result[:top_k])

    async def ensure_loaded(self) -> None:
        """Download and initialize the model if it has not been loaded yet."""
        if self._session is not None:
            return
        async with self._lock:
            if self._session is not None:
                return
            await asyncio.to_thread(self._load)

    def _load(self) -> None:
        try:
            import onnxruntime as ort  # pyright: ignore[reportMissingImports]
            from huggingface_hub import (
                snapshot_download,  # pyright: ignore[reportMissingImports, reportUnknownVariableType]
            )
            from tokenizers import Tokenizer  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise RuntimeError(
                "Emotion service dependencies are missing; install nanobot-ai[emotion]"
            ) from exc

        root = Path(
            snapshot_download(
                repo_id=self.model_id,
                allow_patterns=[
                    MODEL_FILE,
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "vocab.txt",
                ],
            )
        )
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        id_to_label = cast(dict[str, str], config["id2label"])
        self._labels = [id_to_label[str(index)] for index in range(len(id_to_label))]
        tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        tokenizer.enable_truncation(max_length=512)  # pyright: ignore[reportUnknownMemberType]
        tokenizer.enable_padding()  # pyright: ignore[reportUnknownMemberType]
        self._tokenizer = tokenizer
        self._session = ort.InferenceSession(
            str(root / MODEL_FILE),
            providers=["CPUExecutionProvider"],
        )

    def _infer(self, text: str) -> list[dict[str, Any]]:
        import numpy as np

        encoding = self._tokenizer.encode(text)
        inputs = {
            "input_ids": np.asarray([encoding.ids], dtype=np.int64),
            "attention_mask": np.asarray([encoding.attention_mask], dtype=np.int64),
        }
        input_names = {item.name for item in self._session.get_inputs()}
        logits = self._session.run(None, {k: v for k, v in inputs.items() if k in input_names})[0][0]
        shifted = logits - np.max(logits)
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        order = np.argsort(probabilities)[::-1]
        return [
            {"label": self._labels[int(index)], "score": float(probabilities[int(index)])}
            for index in order
        ]


def create_app(model: CoheeEmotionModel | None = None, api_key: str = "") -> Any:
    try:
        from aiohttp import web
    except ImportError as exc:
        raise RuntimeError("Install nanobot-ai[emotion] to run the emotion service") from exc

    classifier = model or CoheeEmotionModel()

    def authorized(request: Any) -> bool:
        return not api_key or request.headers.get("Authorization") == f"Bearer {api_key}"

    async def classify(request: Any) -> Any:
        if not authorized(request):
            raise web.HTTPUnauthorized()
        raw_body: object = await request.json()
        body = cast(dict[object, object], raw_body) if isinstance(raw_body, dict) else {}
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            raise web.HTTPBadRequest(text="text must be a non-empty string")
        raw_top_k = body.get("top_k", 5)
        if isinstance(raw_top_k, bool) or not isinstance(raw_top_k, int):
            raise web.HTTPBadRequest(text="top_k must be an integer")
        top_k = max(1, min(28, raw_top_k))
        result = await classifier.classify(text, top_k=top_k)
        return web.json_response({"classification": result})

    async def labels(request: Any) -> Any:
        if not authorized(request):
            raise web.HTTPUnauthorized()
        await classifier.ensure_loaded()
        return web.json_response({"labels": classifier.labels})

    app = web.Application(client_max_size=64 * 1024)
    app.router.add_post("/classify", classify)
    app.router.add_post("/labels", labels)
    return app


def main() -> None:
    from aiohttp import web

    parser = argparse.ArgumentParser(description="Serve Cohee ONNX emotion classification")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=5101, type=int)
    parser.add_argument("--api-key", default="")
    args = parser.parse_args()
    web.run_app(create_app(api_key=args.api_key), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
