"""Build the Qdrant collections from the DuckDB alias/product tables.

Memory-conscious: strings are embedded in small batches and upserted immediately; the `products`
collection is resumable (point ids are stable row ordinals, so a re-run continues where it stopped).
A manifest records model/dimension/CSV hash/counts so unchanged indexes are not rebuilt.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from chemrag.retrieval.embed import get_embedder
from chemrag.retrieval.vector_store import ENTITIES, PRODUCTS, VectorStore, store_path


def _read(db_path: Path, sql: str) -> list[dict[str, Any]]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        con.close()


def entity_points(db_path: Path) -> list[dict[str, Any]]:
    return _read(db_path, """
        SELECT entity_type, canonical_id, display_name, alias, alias_norm, kind, n_products, extra
        FROM entity_alias WHERE kind NOT IN ('cas') ORDER BY entity_type, canonical_id, alias_norm""")


def product_points(db_path: Path) -> list[dict[str, Any]]:
    return _read(db_path, """
        SELECT product_norm, mode(product_name) AS product_name, list(cdph_id ORDER BY cdph_id) AS cdph_ids,
               list_distinct(list(brand_key) FILTER (WHERE brand_key IS NOT NULL)) AS brand_keys,
               list_distinct(list(company_key)) AS company_keys
        FROM dim_product WHERE product_norm <> '' GROUP BY product_norm ORDER BY product_norm""")


def _embed_upsert(store: VectorStore, name: str, embedder, texts: list[str], payloads: list[dict],
                  batch: int, start: int, log) -> None:
    t0 = time.time()
    for i in range(start, len(texts), batch):
        vecs = embedder.encode(texts[i:i + batch])
        store.upsert(name, list(range(i, i + len(vecs))), vecs, payloads[i:i + batch])
        done = i + len(vecs)
        if done % (batch * 50) < batch or done == len(texts):
            rate = (done - start) / max(time.time() - t0, 1e-6)
            log(f"  {name}: {done:,}/{len(texts):,} ({rate:.0f}/s)")


def build_vector_index(db_path: Path, qdrant_path: Path, manifest_path: Path, model_name: str,
                       csv_sha256: str, products: bool = False, batch: int = 32, threads: int = 4,
                       force: bool = False, log=print) -> dict[str, Any]:
    embedder = get_embedder(model_name, threads, batch)
    store = VectorStore(qdrant_path, ENTITIES)
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    same_basis = old.get("model") == model_name and old.get("csv_sha256") == csv_sha256 and not force
    manifest: dict[str, Any] = {"model": model_name, "dim": embedder.dim, "csv_sha256": csv_sha256,
                                "collections": dict(old.get("collections", {})) if same_basis else {}}

    ents = entity_points(db_path)
    if same_basis and old.get("collections", {}).get(ENTITIES) == len(ents) and store.count(ENTITIES) == len(ents):
        log(f"entities collection up to date ({len(ents):,} points)")
    else:
        log(f"embedding {len(ents):,} entity aliases with {model_name} …")
        store.recreate(ENTITIES, embedder.dim)
        payloads = [{k: (json.loads(v) if k == "extra" else v) for k, v in e.items()} for e in ents]
        _embed_upsert(store, ENTITIES, embedder, [e["alias"] for e in ents], payloads, batch, 0, log)
        manifest["collections"][ENTITIES] = len(ents)

    # Calibration: gte-style models give unrelated short strings fairly high cosine (~0.7+). Record the
    # median similarity of random alias pairs so the resolver can rescale scores to [0, 1].
    rnd = random.Random(7)
    sample = [e["alias"] for e in rnd.sample(ents, min(300, len(ents)))]
    vecs = embedder.encode(sample)
    sims = [float(vecs[i] @ vecs[j]) for i, j in (rnd.sample(range(len(sample)), 2) for _ in range(600))]
    manifest["semantic_floor"] = float(np.median(sims))

    # Persist progress now so an interrupted products build can resume on the same basis.
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({**manifest, "products_complete": False}, indent=2))

    if not products:
        log("products collection not built (default on 8 GB machines; product names use lexical matching). "
            "Use `chemrag build --products` to add it.")
        manifest["collections"].pop(PRODUCTS, None)
        if store_path(qdrant_path, PRODUCTS).exists():
            import shutil

            shutil.rmtree(store_path(qdrant_path, PRODUCTS))
    else:
        store = VectorStore(qdrant_path, PRODUCTS)
        prods = product_points(db_path)
        payloads = [{"product_name": p["product_name"], "product_norm": p["product_norm"],
                     "cdph_ids": [int(x) for x in p["cdph_ids"]],
                     "brand_keys": [int(x) for x in p["brand_keys"] or []],
                     "company_keys": [int(x) for x in p["company_keys"] or []]} for p in prods]
        basis = old.get("model") == model_name and old.get("csv_sha256") == csv_sha256 and not force
        have = store.count(PRODUCTS) if basis else 0
        if have >= len(prods):
            log(f"products collection up to date ({len(prods):,} points)")
        else:
            if have == 0:
                store.recreate(PRODUCTS, embedder.dim)
            else:
                log(f"resuming products collection at {have:,}/{len(prods):,}")
            log(f"embedding {len(prods):,} product names with {model_name} (resumable) …")
            _embed_upsert(store, PRODUCTS, embedder, [p["product_name"] for p in prods], payloads, batch, have, log)
        manifest["collections"][PRODUCTS] = len(prods)

    manifest["products_complete"] = products
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest
