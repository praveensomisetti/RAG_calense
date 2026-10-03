"""Embedded Chroma (no server): `chromadb.PersistentClient(path=...)` persists to local files.

One persistent store holds two collections, `entities` (~3.8k alias names) and the optional `products`
(~33k product names). Chroma keeps its HNSW index on disk and loads a collection only when it is queried,
so opening the store is cheap (~25 MB) and even the products collection adds only ~0.1-0.2 GB when used.
We always pass our own embeddings (gte-large), so Chroma's default embedding function is disabled.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

ENTITIES = "entities"
PRODUCTS = "products"


@lru_cache(maxsize=4)
def _client(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=path, settings=ChromaSettings(anonymized_telemetry=False))


def _clean_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Chroma metadata values must be str/int/float/bool or non-empty lists of one scalar type; no None/dicts."""
    out: dict[str, Any] = {}
    for k, v in payload.items():
        if v is None:
            continue
        if isinstance(v, dict):
            out[k] = json.dumps(v)
        elif isinstance(v, (list, tuple)):
            if v:
                out[k] = list(v)
        else:
            out[k] = v
    return out


class VectorStore:
    """Minimal vector-store API used by the index builder and the resolver."""

    def __init__(self, root: Path):
        self.path = Path(root)
        self.client = _client(str(self.path))

    def exists(self, name: str) -> bool:
        return name in {c.name for c in self.client.list_collections()}

    def count(self, name: str) -> int:
        return self.client.get_collection(name).count() if self.exists(name) else 0

    def recreate(self, name: str, dim: int) -> None:  # dim is implied by the first upsert in Chroma
        if self.exists(name):
            self.client.delete_collection(name)
        self.client.create_collection(name, metadata={"hnsw:space": "cosine"}, embedding_function=None)

    def drop(self, name: str) -> None:
        if self.exists(name):
            self.client.delete_collection(name)

    def upsert(self, name: str, ids: list[int], vectors: np.ndarray, payloads: list[dict[str, Any]]) -> None:
        col = self.client.get_collection(name, embedding_function=None)
        col.upsert(ids=[str(i) for i in ids], embeddings=np.asarray(vectors, dtype=np.float32),
                   metadatas=[_clean_metadata(p) for p in payloads])

    def search(self, name: str, vector: np.ndarray, limit: int = 10,
               must: dict[str, Any] | None = None) -> list[tuple[dict[str, Any], float]]:
        """Return (metadata, cosine similarity) pairs. `must` = {key: value} or {key: [any of values]}."""
        where = None
        if must:
            conds = []
            for key, val in must.items():
                if isinstance(val, list):
                    # list-valued metadata (e.g. brand_keys): match if it contains any requested value
                    alts = [{key: {"$contains": v}} for v in val]
                    conds.append(alts[0] if len(alts) == 1 else {"$or": alts})
                else:
                    conds.append({key: {"$eq": val}})
            where = conds[0] if len(conds) == 1 else {"$and": conds}
        col = self.client.get_collection(name, embedding_function=None)
        res = col.query(query_embeddings=[np.asarray(vector, dtype=np.float32).tolist()], n_results=limit,
                        where=where, include=["metadatas", "distances"])
        metas = res["metadatas"][0] if res["metadatas"] else []
        dists = res["distances"][0] if res["distances"] else []
        return [(m or {}, 1.0 - float(d)) for m, d in zip(metas, dists)]
