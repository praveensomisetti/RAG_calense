"""Chroma vector store round-trip + resolver fusion, using the offline hash-ngram embedder (no downloads)."""

from chemrag.retrieval.embed import HashNgramEmbedder
from chemrag.retrieval.vector_store import PRODUCTS, VectorStore


def test_chroma_roundtrip_and_filters(tmp_path):
    emb = HashNgramEmbedder()
    store = VectorStore(tmp_path / "chroma")
    names = ["Glover's Medicated Shampoo", "Medicated Shampoo", "Lipstick"]
    store.recreate(PRODUCTS, emb.dim)
    store.upsert(PRODUCTS, [0, 1, 2], emb.encode(names),
                 [{"product_name": n, "brand_keys": bk, "extra": {"a": 1}, "missing": None}
                  for n, bk in zip(names, [[912], [5, 6], []])])
    assert store.count(PRODUCTS) == 3
    hits = store.search(PRODUCTS, emb.encode(["glovers medicated shampo"])[0], 2)
    assert hits[0][0]["product_name"] == "Glover's Medicated Shampoo" and 0 < hits[0][1] <= 1
    only_brand = store.search(PRODUCTS, emb.encode(["medicated shampoo"])[0], 3, must={"brand_keys": [5, 999]})
    assert [m["product_name"] for m, _ in only_brand] == ["Medicated Shampoo"]
    store.drop(PRODUCTS)
    assert not store.exists(PRODUCTS)
