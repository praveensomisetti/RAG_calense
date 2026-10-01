"""Embedding backends.

* ``thenlper/gte-large`` (default) via sentence-transformers, CPU-only, loaded lazily and tuned for an
  8 GB laptop (short max_seq_length, inference_mode, capped torch threads).
* ``hash-ngram``: a dependency-free character-trigram hashing embedder. It is *not* semantic; it exists so
  the vector path (Qdrant collections, fusion, tests) still works fully offline when the model cannot be
  downloaded. The README is explicit about this.
"""

from __future__ import annotations

import hashlib
import logging
import os
from functools import lru_cache
from typing import Protocol

import numpy as np

from chemrag.etl.normalize import norm_key

log = logging.getLogger(__name__)


class EmbeddingUnavailable(RuntimeError):
    pass


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, name: str, threads: int = 4, batch_size: int = 32, max_seq_length: int = 64):
        self.name = name
        self.batch_size = batch_size
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(max(1, threads))
            self._torch = torch
            self.model = SentenceTransformer(name, device="cpu")
            self.model.max_seq_length = max_seq_length
        except Exception as e:  # network blocked, model missing, torch missing, ...
            raise EmbeddingUnavailable(f"could not load embedding model {name!r}: {type(e).__name__}: {e}") from e
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        with self._torch.inference_mode():
            vecs = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=False)
        return vecs.astype(np.float32)


class HashNgramEmbedder:
    """Signed feature hashing of character 2/3-grams of the normalised text, L2-normalised."""

    def __init__(self, dim: int = 512):
        self.name = "hash-ngram"
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        s = f"  {norm_key(text)}  "
        for n in (2, 3):
            for i in range(len(s) - n + 1):
                h = int.from_bytes(hashlib.blake2b(s[i:i + n].encode(), digest_size=8).digest(), "little")
                v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)


@lru_cache(maxsize=2)
def get_embedder(name: str, threads: int = 4, batch_size: int = 32) -> Embedder:
    if name == "hash-ngram":
        return HashNgramEmbedder()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return SentenceTransformerEmbedder(name, threads=threads, batch_size=batch_size)
