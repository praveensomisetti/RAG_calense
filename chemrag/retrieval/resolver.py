"""Entity resolution: raw mention -> canonical IDs, with ranked candidates and ambiguity flags.

Pipeline per mention (short-circuits on success):
  1. CAS path      - normalise + check-digit validate, exact lookup
  2. exact/alias   - normalised alias table lookup (names, curated synonyms, company match keys)
  3. fuzzy+vector  - rapidfuzz lexical score fused with calibrated gte-large cosine from Qdrant
No LLM is involved; every decision is reproducible from the candidate scores in the trace.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from rapidfuzz import fuzz, process

from chemrag.etl.cas import looks_like_cas, near_miss_suggestions, normalize_cas
from chemrag.etl.normalize import company_match_key, norm_key
from chemrag.query.tools import QueryEngine
from chemrag.retrieval.embed import EmbeddingUnavailable, get_embedder
from chemrag.retrieval.vector_store import ENTITIES, PRODUCTS, VectorStore
from chemrag.schemas import Candidate, EntityType, Mention, ProductResolution, Resolution
from chemrag.settings import Settings

log = logging.getLogger(__name__)

RESOLVABLE = [EntityType.CHEMICAL, EntityType.COMPANY, EntityType.BRAND, EntityType.SUBCATEGORY,
              EntityType.PRIMARY_CATEGORY]
# When an exact match exists in several types, prefer in this order.
TYPE_PRIORITY = {EntityType.CHEMICAL: 0, EntityType.CHEMICAL_FAMILY: 1, EntityType.SUBCATEGORY: 2,
                 EntityType.PRIMARY_CATEGORY: 3, EntityType.COMPANY: 4, EntityType.BRAND: 5}


def lexical_score(query: str, alias: str, **_: Any) -> float:
    """0..100. Exact-ish spelling dominates; token-set/partial matches are discounted.

    Token-set similarity is scaled by how much of the *query* the alias covers, so a short alias that
    is merely contained in a long query ("lipstick" inside "ultra color ... lipstick") scores low,
    while a short query contained in a longer alias ("pure" in "pure ice") stays high (and ambiguous).
    Partial matching only applies when the query could be a fragment of the alias.
    """
    r = fuzz.ratio(query, alias)
    q_tokens, a_tokens = set(query.split()), set(alias.split())
    coverage = len(q_tokens & a_tokens) / len(q_tokens) if q_tokens else 0.0
    ts = fuzz.token_set_ratio(query, alias) * (0.5 + 0.5 * coverage)
    pr = fuzz.partial_ratio(query, alias) if len(query) >= 5 and len(alias) >= len(query) else 0.0
    return max(r, 0.9 * ts, 0.85 * pr)


@dataclass
class AliasRow:
    entity_type: EntityType
    canonical_id: int | None
    display_name: str
    alias: str
    alias_norm: str
    kind: str
    n_products: int
    extra: dict[str, Any] = field(default_factory=dict)


class VectorIndex:
    """Lazy wrapper around the embedder + embedded Qdrant. Any failure -> unavailable (lexical-only)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._ready: bool | None = None
        self.reason = ""
        self.floor = 0.0
        self.has_products = False

    def available(self) -> bool:
        if self._ready is None:
            self._ready = self._init()
        return self._ready

    def _init(self) -> bool:
        mp = self.settings.manifest_path
        if not mp.exists():
            self.reason = "vector index not built (run `chemrag build`)"
            return False
        manifest = json.loads(mp.read_text())
        if manifest.get("model") != self.settings.embed_model:
            self.reason = (f"index built with {manifest.get('model')!r} but CHEMRAG_EMBED_MODEL="
                           f"{self.settings.embed_model!r}; rebuild")
            return False
        try:
            self.embedder = get_embedder(self.settings.embed_model, self.settings.torch_threads,
                                         self.settings.embed_batch)
            self.store = VectorStore(self.settings.qdrant_path)
        except EmbeddingUnavailable as e:
            self.reason = str(e)
            return False
        except Exception as e:  # pragma: no cover - e.g. qdrant storage locked
            self.reason = f"vector store unavailable: {e}"
            return False
        self.floor = float(manifest.get("semantic_floor", 0.0))
        self.has_products = bool(manifest.get("products_complete")) and self.store.exists(PRODUCTS)
        return True

    def calibrate(self, raw: float) -> float:
        return max(0.0, min(1.0, (raw - self.floor) / max(1e-6, 1.0 - self.floor)))

    def search_entities(self, text: str, etype: EntityType, limit: int = 10) -> list[tuple[dict, float]]:
        vec = self.embedder.encode([text])[0]
        return [(p, self.calibrate(s)) for p, s in
                self.store.search(ENTITIES, vec, limit, must={"entity_type": etype.value})]

    def search_products(self, text: str, limit: int = 10, brand_keys: list[int] | None = None,
                        company_keys: list[int] | None = None) -> list[tuple[dict, float]]:
        vec = self.embedder.encode([text])[0]
        must: dict[str, Any] = {}
        if brand_keys:
            must["brand_keys"] = brand_keys
        if company_keys:
            must["company_keys"] = company_keys
        return [(p, self.calibrate(s)) for p, s in self.store.search(PRODUCTS, vec, limit, must=must or None)]


class EntityResolver:
    def __init__(self, engine: QueryEngine, settings: Settings, vectors: VectorIndex | None = None,
                 use_vectors: bool = True):
        self.engine = engine
        self.settings = settings
        self.th = settings.thresholds
        self.vectors = vectors if vectors is not None else VectorIndex(settings)
        self.use_vectors = use_vectors
        self.by_norm: dict[str, list[AliasRow]] = defaultdict(list)
        self.by_type: dict[EntityType, list[AliasRow]] = defaultdict(list)
        self.cas_index: dict[str, list[AliasRow]] = defaultdict(list)
        self.family_groups: dict[str, list[int]] = defaultdict(list)
        for r in engine.table("SELECT * FROM entity_alias"):
            row = AliasRow(EntityType(r["entity_type"]), r["canonical_id"], r["display_name"], r["alias"],
                           r["alias_norm"], r["kind"], r["n_products"], json.loads(r["extra"] or "{}"))
            if row.kind == "cas":
                self.cas_index[row.alias].append(row)
                continue
            self.by_norm[row.alias_norm].append(row)
            self.by_type[row.entity_type].append(row)
        for r in engine.table("SELECT chem_group_id, family FROM dim_chemical_group WHERE family IS NOT NULL"):
            self.family_groups[r["family"]].append(r["chem_group_id"])
        self.group_info = {r["chem_group_id"]: r for r in engine.table(
            "SELECT chem_group_id, group_name, n_products, member_names FROM dim_chemical_group")}
        self.known_cas = sorted(self.cas_index)
        self._choices = {t: [r.alias_norm for r in rows] for t, rows in self.by_type.items()}
        prods = engine.table("""SELECT product_norm, mode(product_name) AS product_name,
                                       list(cdph_id ORDER BY cdph_id) AS cdph_ids,
                                       list(DISTINCT brand_name) FILTER (WHERE brand_name IS NOT NULL) AS brands,
                                       list(DISTINCT company_name) AS companies,
                                       list(DISTINCT brand_key) FILTER (WHERE brand_key IS NOT NULL) AS brand_keys,
                                       list(DISTINCT company_key) AS company_keys
                                FROM dim_product WHERE product_norm <> '' GROUP BY product_norm""")
        self.products = {p["product_norm"]: p for p in prods}
        self._product_choices = list(self.products)
        self.vector_warning: str | None = None

    # ------------------------------------------------------------------ helpers
    def _cand(self, row: AliasRow, score: float, method: str, lexical: float = 0.0,
              semantic: float | None = None) -> Candidate:
        return Candidate(entity_type=row.entity_type, canonical_id=row.canonical_id, display_name=row.display_name,
                         matched_alias=row.alias, score=round(score, 4), lexical=round(lexical, 4),
                         semantic=None if semantic is None else round(semantic, 4), method=method,
                         n_products=row.n_products, extra=row.extra)

    def _vectors_on(self) -> bool:
        if not self.use_vectors:
            return False
        ok = self.vectors.available()
        if not ok and self.vector_warning is None:
            self.vector_warning = f"semantic search unavailable, lexical only: {self.vectors.reason}"
        return ok

    def allowed_types(self, mention: Mention, hints: list[EntityType] | None = None) -> list[EntityType]:
        if mention.type not in (EntityType.UNKNOWN, EntityType.CAS):
            return [mention.type, EntityType.CHEMICAL_FAMILY] if mention.type == EntityType.CHEMICAL else [mention.type]
        return hints or RESOLVABLE + [EntityType.CHEMICAL_FAMILY]

    # ------------------------------------------------------------------ main entry
    def resolve(self, mention: Mention, subtask_id: str, hints: list[EntityType] | None = None) -> Resolution:
        text = mention.text.strip()
        if mention.type == EntityType.CAS or looks_like_cas(text):
            return self._resolve_cas(mention, subtask_id)
        types = self.allowed_types(mention, hints)
        key = norm_key(text)
        if len(key) < 2:
            return Resolution(subtask_id=subtask_id, mention=mention, status="not_found", note="mention too short")

        exact = [r for r in self.by_norm.get(key, []) if r.entity_type in types]
        if EntityType.COMPANY in types:
            ck = company_match_key(text)
            exact += [r for r in self.by_type[EntityType.COMPANY] if r.kind == "match_key" and r.alias_norm == ck
                      and r not in exact]
        if exact:
            return self._from_exact(mention, subtask_id, exact)
        return self._fuzzy(mention, subtask_id, key, types)

    def _resolve_cas(self, mention: Mention, subtask_id: str) -> Resolution:
        raw = mention.text.strip()
        cas, status = normalize_cas(raw)
        mention = mention.model_copy(update={"type": EntityType.CAS})
        if cas is None:
            sugg = near_miss_suggestions(raw, self.known_cas)
            note = f"'{raw}' is not a valid CAS number (check digit fails)"
            cands = [self._cand(self.cas_index[s][0], 0.0, "cas").model_copy(update={"matched_alias": s})
                     for s in sugg]
            return Resolution(subtask_id=subtask_id, mention=mention, status="not_found", candidates=cands,
                              note=note + (f"; did you mean {', '.join(sugg)}?" if sugg else ""))
        rows = self.cas_index.get(cas, [])
        if not rows:
            return Resolution(subtask_id=subtask_id, mention=mention, status="not_found",
                              note=f"{cas} is a valid CAS number but does not appear in this dataset")
        cands = [self._cand(r, 1.0, "cas", 1.0).model_copy(update={"matched_alias": cas}) for r in rows]
        note = None
        if status == "repaired":
            note = f"interpreted '{raw}' as CAS {cas}"
        if len(cands) > 1:
            extra = f"CAS {cas} is reported under {len(cands)} chemical entries: " + \
                    ", ".join(c.display_name for c in cands)
            note = f"{note}; {extra}" if note else extra
        return Resolution(subtask_id=subtask_id, mention=mention, status="resolved", chosen=cands,
                          candidates=cands, note=note)

    def _from_exact(self, mention: Mention, subtask_id: str, rows: list[AliasRow]) -> Resolution:
        rows = sorted(rows, key=lambda r: (TYPE_PRIORITY.get(r.entity_type, 9), -r.n_products))
        top_type = rows[0].entity_type
        if top_type == EntityType.CHEMICAL_FAMILY:
            fam = rows[0].extra["family"]
            chosen = [Candidate(entity_type=EntityType.CHEMICAL, canonical_id=g,
                                display_name=self.group_info[g]["group_name"], matched_alias=rows[0].alias,
                                score=1.0, lexical=1.0, method="family",
                                n_products=self.group_info[g]["n_products"]) for g in self.family_groups[fam]]
            return Resolution(subtask_id=subtask_id, mention=mention, status="resolved", chosen=chosen,
                              candidates=chosen, note=f"'{mention.text}' expanded to the {rows[0].display_name} "
                                                      f"family ({len(chosen)} chemical groups)")
        same = {}
        for r in rows:
            if r.entity_type == top_type:
                same.setdefault(r.canonical_id, r)
        cands = [self._cand(r, 1.0, "exact" if r.kind == "name" else "alias", 1.0) for r in same.values()]
        others = [self._cand(r, 1.0, "exact", 1.0) for r in rows if r.entity_type != top_type]
        if len(cands) > 1:
            return Resolution(subtask_id=subtask_id, mention=mention, status="ambiguous", candidates=cands + others,
                              note=f"'{mention.text}' exactly matches {len(cands)} different {top_type.value}s")
        note = None
        if cands[0].method == "alias" and norm_key(cands[0].display_name) != norm_key(mention.text):
            note = f"'{mention.text}' resolved to {top_type.value} {cands[0].display_name}"
        if others:
            alt = ", ".join(f"{o.entity_type.value} {o.display_name}" for o in others[:3])
            note = (note + "; " if note else "") + f"also matches {alt}; using {top_type.value}"
        return Resolution(subtask_id=subtask_id, mention=mention, status="resolved", chosen=cands,
                          candidates=cands + others, note=note)

    def _fuzzy(self, mention: Mention, subtask_id: str, key: str, types: list[EntityType]) -> Resolution:
        best: dict[tuple[EntityType, int | None], Candidate] = {}
        use_vec = self._vectors_on()
        w = self.th.lexical_weight
        for t in types:
            rows = self.by_type.get(t, [])
            if not rows:
                continue
            hits = process.extract(key, self._choices[t], scorer=lexical_score, limit=10, score_cutoff=50)
            lex: dict[int | None, tuple[float, AliasRow]] = {}
            for _, score, idx in hits:
                r = rows[idx]
                if r.canonical_id not in lex or score / 100.0 > lex[r.canonical_id][0]:
                    lex[r.canonical_id] = (score / 100.0, r)
            sem: dict[int | None, tuple[float, dict]] = {}
            if use_vec:
                for payload, s in self.vectors.search_entities(mention.text, t, 10):
                    cid = payload.get("canonical_id")
                    if cid not in sem or s > sem[cid][0]:
                        sem[cid] = (s, payload)
            for cid in set(lex) | set(sem):
                l_score, row = lex.get(cid, (0.0, None))
                s_score = sem[cid][0] if cid in sem else (0.0 if use_vec else None)
                if row is None:  # semantic-only hit: recover alias row
                    p = sem[cid][1]
                    row = next((r for r in rows if r.canonical_id == cid and r.alias_norm == p.get("alias_norm")),
                               None) or next(r for r in rows if r.canonical_id == cid)
                    l_score = lexical_score(key, row.alias_norm) / 100.0
                fused = w * l_score + (1 - w) * s_score if use_vec else l_score
                c = self._cand(row, fused, "vector" if use_vec and s_score and s_score > l_score else "fuzzy",
                               l_score, s_score)
                k = (t, cid)
                if k not in best or c.score > best[k].score:
                    best[k] = c
        ranked = sorted(best.values(), key=lambda c: (-c.score, -c.n_products))
        # expand family candidates to their chemical groups
        top = ranked[:5]
        if not top or top[0].score < self.th.resolve_low:
            return Resolution(subtask_id=subtask_id, mention=mention, status="not_found", candidates=top[:3],
                              note=f"no confident match for '{mention.text}'"
                                   + (f"; closest: {', '.join(c.display_name for c in top[:3])}" if top else ""))
        first = top[0]
        rivals = [c for c in top[1:] if c.score >= self.th.resolve_low
                  and first.score - c.score < self.th.ambiguity_margin]
        if rivals:
            return Resolution(subtask_id=subtask_id, mention=mention, status="ambiguous", candidates=top,
                              note=f"'{mention.text}' is ambiguous: " +
                                   ", ".join(f"{c.display_name} ({c.entity_type.value}, {c.score:.2f})"
                                             for c in [first] + rivals))
        if first.entity_type == EntityType.CHEMICAL_FAMILY:
            return self._from_exact(mention, subtask_id, [r for r in self.by_type[EntityType.CHEMICAL_FAMILY]
                                                          if r.display_name == first.display_name])
        note = None
        if first.score < self.th.resolve_high or norm_key(first.matched_alias) != key:
            note = (f"interpreted '{mention.text}' as {first.entity_type.value} {first.display_name} "
                    f"(match score {first.score:.2f})")
        return Resolution(subtask_id=subtask_id, mention=mention, status="resolved", chosen=[first],
                          candidates=top, note=note)

    # ------------------------------------------------------------------ products (record retrieval)
    def resolve_product(self, mention: Mention, subtask_id: str, brand_keys: list[int] | None = None,
                        company_keys: list[int] | None = None) -> ProductResolution:
        key = norm_key(mention.text)
        pool = self._product_choices
        if brand_keys or company_keys:
            bk, ck = set(brand_keys or []), set(company_keys or [])
            pool = [n for n in pool if (not bk or bk & set(self.products[n]["brand_keys"] or []))
                    and (not ck or ck & set(self.products[n]["company_keys"] or []))]
        scores: dict[str, float] = {}
        if key in self.products and key in pool:
            scores[key] = 1.0
        for name, _score, _ in process.extract(key, pool, scorer=fuzz.WRatio, limit=25, score_cutoff=60):
            scores[name] = max(scores.get(name, 0.0), lexical_score(key, name) / 100.0)
        method = "lexical"
        if self._vectors_on() and self.vectors.has_products:
            method = "lexical+vector"
            w = self.th.lexical_weight
            for payload, s in self.vectors.search_products(mention.text, 10, brand_keys, company_keys):
                n = payload.get("product_norm")
                if n in self.products:
                    lex = scores.get(n, lexical_score(key, n) / 100.0)
                    scores[n] = max(scores.get(n, 0.0), w * lex + (1 - w) * s)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:10]
        cands = [{"product_name": self.products[n]["product_name"], "cdph_ids": self.products[n]["cdph_ids"],
                  "brands": self.products[n]["brands"], "companies": self.products[n]["companies"],
                  "score": round(s, 4)} for n, s in ranked]
        if not ranked or ranked[0][1] < self.th.resolve_low:
            return ProductResolution(subtask_id=subtask_id, mention=mention, status="not_found", candidates=cands[:5],
                                     note=f"no product name matches '{mention.text}' ({method})")
        top_name, top_score = ranked[0]
        close = [c for c in cands[1:] if top_score - c["score"] < self.th.ambiguity_margin
                 and c["score"] >= self.th.resolve_low]
        top = self.products[top_name]
        many_companies = len(top["companies"]) > 1 and len(top["cdph_ids"]) > 5
        if close or (many_companies and not (brand_keys or company_keys)):
            note = (f"'{mention.text}' matches {len(close) + 1} product names" if close else
                    f"'{top['product_name']}' is a generic name used by {len(top['cdph_ids'])} products from "
                    f"{len(top['companies'])} companies")
            return ProductResolution(subtask_id=subtask_id, mention=mention, status="ambiguous",
                                     cdph_ids=list(top["cdph_ids"]), candidates=cands[:8], note=note)
        note = None
        if top_score < 1.0:
            note = f"interpreted '{mention.text}' as product '{top['product_name']}' (score {top_score:.2f}, {method})"
        if len(top["cdph_ids"]) > 1:
            note = (note + "; " if note else "") + f"{len(top['cdph_ids'])} products share this name"
        return ProductResolution(subtask_id=subtask_id, mention=mention, status="resolved",
                                 cdph_ids=list(top["cdph_ids"]), candidates=cands[:5], note=note)

    def gazetteer(self) -> dict[str, list[AliasRow]]:
        return self.by_norm


def load_resolver(engine: QueryEngine, settings: Settings, use_vectors: bool = True) -> EntityResolver:
    return EntityResolver(engine, settings, use_vectors=use_vectors)


__all__ = ["EntityResolver", "VectorIndex", "lexical_score", "load_resolver"]
