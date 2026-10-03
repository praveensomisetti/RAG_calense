# ChemRAG Cheat Sheet: the whole system on one page

> **One line:** clean the data once, use **SQL for every fact**, use **vectors only to understand names**, let the **LLM only read and phrase**, and **fact-check** before answering.

---

## 1. The end-to-end diagram

```
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║  PHASE A: BUILD TIME (run once: `chemrag build`; no LLM)                                 ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

  interviewtestdataset.csv  (114,635 rows × 22 columns)
            │
            ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ ① INGESTION + CLEANING      Python · pandas · DuckDB         │
  │   • SHA-256 fingerprint (skip rebuild if unchanged)          │
  │   • read all columns as text + strict 22-column schema check │
  │   • row_id = CSV line position (for citations)               │
  │   • trim text, "NA"/"None" → empty                           │
  │   • strict dates; impossible dates (2103) → empty + flagged  │
  │   • CAS repair ONLY if the check digit proves it             │
  │   • flag 254 exact duplicates, "Trade Secret", test record   │
  │   • synonym groups from chemical_groups.yaml (hand-curated)  │
  │   • every change logged in dq_issues; raw data kept          │
  │   • write to temp file → swap (never half-written)           │
  └───────────────┬──────────────────────────────┬──────────────┘
                  │ facts                         │ names only (~3.8k)
                  ▼                               ▼
  ┌───────────────────────────────┐   ┌──────────────────────────────────────────┐
  │ ② DuckDB  (cscp.duckdb)       │   │ ③ EMBEDDINGS + VECTOR DB                  │
  │   embedded SQL database        │   │   model: thenlper/gte-large (1024-dim,    │
  │   fact_report + dim_* tables   │   │          CPU, sentence-transformers)      │
  │   entity_alias, dq_issues,     │   │   store: Chroma (embedded, on disk,       │
  │   dataset_meta (date ranges)   │   │          cosine HNSW, no server)          │
  │   → source of EVERY number     │   │   → only answers "which name was meant?"  │
  └───────────────────────────────┘   └──────────────────────────────────────────┘


╔══════════════════════════════════════════════════════════════════════════════════════════╗
║  PHASE B: QUESTION TIME (`chemrag ask "..."`): a LangGraph StateGraph, 10 nodes            ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

  User question  e.g. "Which products contain acetaldehide, and how many were discontinued?"
        │
        ▼
  ④ GUARD ─────────── input guardrail: length cap, strip prompt-injection sentences
        │
        ▼
  ⑤ PLANNER ───────── LLM (OpenAI gpt-4o-mini / gpt-5-mini) + rules
        │               scope (in scope / medical / unrelated) · intent · split parts
        │               boundaries + wording come from deterministic code (keeps exact names)
        ▼
  ⑥ EXTRACTOR ─────── rules first (regex, name dictionary, cue phrases) + LLM may ADD mentions
        │               → chemical "acetaldehide", "discontinued", dates, CAS, brand…
        ▼
  ⑦ RESOLVER ──────── CAS check digit → exact alias → fuzzy (rapidfuzz) + vectors (Chroma)
        │               "acetaldehide" → Acetaldehyde (score 0.92)
        ├── ambiguous? ──► CLARIFY ("Which 'Pure' did you mean?"; pauses via LangGraph interrupt())
        ├── product name? ► RETRIEVAL (product names → product IDs)
        ▼
  ⑧ QUERY AGENT ───── 6 pre-written, parameterised SQL tools on read-only DuckDB
        │               out-of-range dates → "no data in range"; empty → explains why
        │               loops once per sub-question  → 30 products … then 13 discontinued
        ▼
  ⑨ SYNTHESIZER ───── facts F1..Fn → short answer from a TEMPLATE (no LLM numbers)
        │               + LLM detail bullets that must cite [F#] + evidence rows
        ▼
  ⑩ VERIFIER ──────── output guardrail: every number/citation must match the facts,
        │               no health advice, no prompt leak → else use template text
        ▼
  ⑪ FINALIZE ──────── answer · evidence (row_id, CDPHId, CSFId, ChemicalId) · query plan
                        with exact SQL · assumptions · warnings · confidence
                        + saved run file → `chemrag replay <id>` re-runs the SQL
```

---

## 2. "What did we use?" Quick answers

| Question | Answer |
|---|---|
| **Ingestion technique?** | Deterministic **ETL in Python/pandas into DuckDB**: schema check, CAS check-digit repair, dedup flags, strict dates, synonym YAML, data-quality log, atomic write. **No LLM in ingestion.** |
| **Database?** | **DuckDB**, embedded, read-only at query time, memory-capped at 1 GB |
| **Embedding model?** | **`thenlper/gte-large`**, 1024-dim, CPU, loaded lazily |
| **What gets embedded?** | Only **names**: ~3.8k chemicals/synonyms, companies, brands and categories (optionally 33k product names). **Never the 114k rows.** |
| **Vector database?** | **Chroma**, embedded `PersistentClient`, on-disk, cosine HNSW. It replaced Qdrant after measuring RAM: 0.41 GB vs 1.43 GB. |
| **Fuzzy matching?** | **rapidfuzz**: fused score = 65% spelling + 35% meaning |
| **LLM?** | **OpenAI** via structured outputs (`chat.completions.parse` + Pydantic). Default `gpt-5-mini`; evaluated with `gpt-4o-mini`. Gemini is an optional alternative. |
| **Where is the LLM used?** | Only 3 places: **planner** (scope/intent), **extractor** (extra mentions), **synthesizer** (wording). Never for numbers, never for SQL. |
| **Orchestration?** | **LangGraph** `StateGraph`: 10 nodes, deterministic routers, `interrupt()` for clarification, checkpointer |
| **Agents?** | 7 agents (planner, extractor, resolver, retrieval, query, synthesizer, verifier) + 3 support nodes (guard, clarify, finalize) |
| **Data contracts?** | **Pydantic** models for every hand-off and for the final JSON response |
| **Interface?** | **CLI** (`chemrag build / ask / replay / eval / doctor / graph`), with Rich output or JSON |
| **RAG type?** | **Structured RAG**: SQL rows and name vectors retrieve → numbered facts augment the prompt → the LLM generates → the verifier checks |

---

## 3. Guardrails: where safety happens

| # | Guardrail | Where | What it stops |
|---|---|---|---|
| 1 | Input length cap + injection filter | Guard | "Ignore previous instructions…", SQL like `DROP TABLE` |
| 2 | User text wrapped as **untrusted data** | LLM prompts | Instructions hidden inside the question |
| 3 | **Strict schema output** (Pydantic) | Every LLM call | Free-form or malformed LLM output |
| 4 | LLM has **no tool authority** | Graph routers | The LLM choosing what runs next |
| 5 | **No LLM-written SQL**; parameterised templates; **read-only** DB | Query agent | SQL injection and unverifiable queries |
| 6 | Deterministic sub-question splitting | Planner | The LLM rewriting or splitting the user's question |
| 7 | **Clarification** instead of guessing | Resolver → Clarify | Confident wrong entity ("Pure", "Lipstick") |
| 8 | **Not found** + suggestions | Resolver | Silently dropping an unknown chemical ("acetone", "glyphosate") |
| 9 | Date-coverage check | Query agent | Invented answers for 2024 ("no data in range") |
| 10 | Medical-advice refusal (rules + LLM) | Planner + Synthesizer | Health or safety advice; still gives the facts |
| 11 | **Templated short answer** | Synthesizer | LLM-invented numbers in the headline |
| 12 | **Verifier**: numbers, `[F#]` citations, advice, prompt-leak checks | Verifier | Hallucinated numbers and claims, so it falls back to template |
| 13 | **Circuit breaker** + rules fallback | LLM client | Outages, bad keys, unknown model |
| 14 | Warnings and assumptions shown to the user | Finalize | Hidden interpretation ("read 'acetaldehide' as Acetaldehyde") |

---

## 4. Numbers to remember

| | |
|---|---|
| Dataset | 114,635 rows · 36,972 products · 606 company names · 123 chemical names |
| Data issues | 254 exact duplicates · ~9.3k re-categorised rows · 115 impossible dates · ~25 dirty CAS strings · titanium dioxide = 81% of rows |
| Vectors | ~3.8k names embedded (optionally 33k product names), never the rows |
| Graph | 10 nodes · 7 agents · 3 call the LLM |
| **LLM-mode eval** (gpt-4o-mini + gte-large) | **10/10**, every metric 100% (first run 7/10, then fixed) |
| Rules-only eval | 37/37 golden · held-out 10/14 on first run → 14/14 |
| Tests | 98 automated tests |
| Cost | Well under 1 cent per question; quick eval ≈ 30 LLM calls |

---

## 5. The 30-second pitch

> "ChemRAG answers plain-English questions about California's cosmetics chemical-disclosure data.
> I ingest the CSV with deterministic cleaning into **DuckDB**, so every number comes from SQL.
> I embed only the **names** with **gte-large** into **Chroma**, so typos and synonyms resolve correctly.
> A **LangGraph** pipeline of seven agents plans, extracts, resolves and queries, and **OpenAI** only reads the question and phrases the answer.
> A **verifier** rejects any number that isn't in the query results.
> Every answer ships with source rows, the exact SQL, assumptions and confidence. It scored **10/10 with the LLM**, and it still works if the LLM is down."
