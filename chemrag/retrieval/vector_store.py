"""Embedded Qdrant (no server): `QdrantClient(path=...)` persists to local files.

Local mode allows one client per storage path per process, so clients are cached here.
"""

from __future__ import annotations

import atexit
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

ENTITIES = "entities"
PRODUCTS = "products"


@lru_cache(maxsize=4)
def _client(path: str) -> QdrantClient:
    Path(path).mkdir(parents=True, exist_ok=True)
    # ~37k short-string points: local brute-force search takes milliseconds, so the size advice does not apply.
    warnings.filterwarnings("ignore", message="Local mode is not recommended")
    client = QdrantClient(path=path)
    atexit.register(client.close)  # close cleanly before interpreter teardown (avoids noisy __del__)
    return client


class VectorStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.client = _client(str(self.path))

    def exists(self, name: str) -> bool:
        return self.client.collection_exists(name)

    def count(self, name: str) -> int:
        return self.client.count(name, exact=True).count if self.exists(name) else 0

    def recreate(self, name: str, dim: int) -> None:
        if self.exists(name):
            self.client.delete_collection(name)
        self.client.create_collection(
            name, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
        )

    def upsert(self, name: str, ids: list[int], vectors: np.ndarray, payloads: list[dict[str, Any]]) -> None:
        self.client.upsert(
            name,
            points=[models.PointStruct(id=i, vector=v.tolist(), payload=p) for i, v, p in zip(ids, vectors, payloads)],
            wait=True,
        )

    def search(self, name: str, vector: np.ndarray, limit: int = 10,
               must: dict[str, Any] | None = None) -> list[tuple[dict[str, Any], float]]:
        flt = None
        if must:
            conds = []
            for key, val in must.items():
                if isinstance(val, list):
                    conds.append(models.FieldCondition(key=key, match=models.MatchAny(any=val)))
                else:
                    conds.append(models.FieldCondition(key=key, match=models.MatchValue(value=val)))
            flt = models.Filter(must=conds)
        res = self.client.query_points(name, query=vector.tolist(), query_filter=flt, limit=limit, with_payload=True)
        return [(p.payload or {}, float(p.score)) for p in res.points]
