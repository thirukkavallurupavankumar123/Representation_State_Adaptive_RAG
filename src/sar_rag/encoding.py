from __future__ import annotations

from hashlib import blake2b
import os
import re
from typing import Protocol

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class Encoder(Protocol):
    dimension: int

    def encode(self, texts: list[str]) -> np.ndarray: ...
    def token_count(self, text: str) -> int: ...
    def prepare_embedding_text(self, text: str, max_tokens: int = 256) -> tuple[str, int, bool]: ...


class HashingEncoder:
    """Deterministic local encoder for tests and pipeline verification.

    This is deliberately not presented as a research-quality semantic model.
    """

    def __init__(self, dimension: int = 512) -> None:
        self.dimension = dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in TOKEN_RE.findall(text.lower()):
                digest = blake2b(token.encode(), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                column = value % self.dimension
                sign = 1.0 if (value >> 8) & 1 else -1.0
                matrix[row, column] += sign
            norm = np.linalg.norm(matrix[row])
            if norm:
                matrix[row] /= norm
        return matrix

    def token_count(self, text: str) -> int:
        return len(TOKEN_RE.findall(text))

    def prepare_embedding_text(self, text: str, max_tokens: int = 256) -> tuple[str, int, bool]:
        count = self.token_count(text)
        return text, count, count > max_tokens


class SentenceTransformerEncoder:
    """PyTorch inference for the all-MiniLM-L6-v2 Sentence Transformer checkpoint."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", revision: str | None = None):
        # Sentence Transformers is used through PyTorch. Prevent unrelated or partially
        # installed TensorFlow packages from being imported by Transformers.
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("USE_TORCH", "1")
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch/Transformers could not be imported; install the project dependencies "
                "before running research inference"
            ) from exc
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModel.from_pretrained(model_name, revision=revision)
        self.model.eval()
        self.dimension = int(self.model.config.hidden_size)
        self.encode_calls = 0
        self.encoded_texts = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.encode_calls += 1
        self.encoded_texts += len(texts)
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        with self._torch.inference_mode():
            token_embeddings = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            summed = (token_embeddings * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
            normalized = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
        return normalized.cpu().numpy().astype(np.float32, copy=False)

    def token_count(self, text: str) -> int:
        return len(self.tokenizer(
            text, add_special_tokens=True, truncation=False, verbose=False
        )["input_ids"])

    def prepare_embedding_text(self, text: str, max_tokens: int = 256) -> tuple[str, int, bool]:
        full = self.tokenizer(text, add_special_tokens=True, truncation=False, verbose=False)["input_ids"]
        if len(full) <= max_tokens:
            return text, len(full), False
        shortened = self.tokenizer(
            text, add_special_tokens=False, truncation=True, max_length=max_tokens - 2,
            verbose=False,
        )["input_ids"]
        return self.tokenizer.decode(shortened, skip_special_tokens=True), len(full), True
