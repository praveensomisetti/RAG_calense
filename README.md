# ChemRAG: a multi-agent orchestrator for the California Safe Cosmetics chemical-disclosure data

Ask a question in plain English, for example *"Which products contain CAS 75-07-0?"*,
*"What chemicals are reported for Sally Hansen in Nail Polish and Enamel?"* or
*"Summarize reporting trends over time for Nail Products"*. Every answer includes:

- **a short answer and details**,
- **evidence**: the cited source rows (`row_id`, `CDPHId`, `CSFId`, `ChemicalId` and the fields used),
- **a query plan**: every agent step, including the exact SQL and the values passed to it,
- **assumptions, warnings and a confidence level**.

**New to the repo?** Start with [`docs/REPO_GUIDE.md`](docs/REPO_GUIDE.md), which explains every folder and file: what it does and why.
The full design is in [`PLAN.md`](PLAN.md). Profiling numbers are in [`docs/profiling_output.txt`](docs/profiling_output.txt).

Setup takes about 10 minutes; see §1.

---

## 1. Local setup (step by step)

Requirements: **Python 3.11 or newer**, git, about 3 GB free disk (the venv, plus the ~670 MB gte-large model), and 8 GB RAM. CPU only; no servers or Docker.

```bash
# 1. Clone and switch to the branch that has the code
git clone https://github.com/praveensomisetti/RAG_calense.git
cd RAG_calense
git checkout claude/chemical-disclosure-orchestrator-plan-bqeqyj

# 2. Create and activate a virtual environment
python3 --version                 # must be 3.11+
python3 -m venv .venv
source .venv/bin/activate         # Windows PowerShell: .venv\Scripts\Activate.ps1   (cmd: .venv\Scripts\activate.bat)
python -m pip install --upgrade pip

# 3. Install CPU-only PyTorch first (avoids a ~2 GB CUDA download; on macOS plain `pip install torch` is already CPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. Install the project + dependencies (defined in pyproject.toml)
pip install -r requirements.txt   # same as: pip install -e ".[dev]"

# 5. Configure
cp .env.example .env              # Windows: copy .env.example .env
#   edit .env: set OPENAI_API_KEY=sk-...   (leave empty to run in deterministic no-LLM mode)

# 6. Check the environment (RAM, model download, OpenAI key, model id + one test call)
chemrag doctor
#   if "LLM (openai): model not found", copy one of the listed mini/nano model ids into CHEMRAG_LLM_MODEL in .env

# 7. Build the data: CSV -> DuckDB (~25 s), then entity names -> gte-large -> Chroma
chemrag build                     # first run downloads gte-large (~670 MB)
#   optional: chemrag build --products   (also embeds 33k product names, ~10-25 min on CPU)

# 8. Verify
pytest -q                         # 94 tests
chemrag eval --quick              # 10 representative questions (~30 LLM calls; budget-friendly)
chemrag eval                      # full golden set, 37 questions (writes evals/results/)

# 9. Ask questions
chemrag ask "Which products contain CAS 75-07-0?"
chemrag ask "What chemicals are reported for Sally Hansen in Nail Polish and Enamel?"
chemrag ask "Summarize reporting trends over time for Nail Products" --json
```

On macOS/Linux, `make setup`, `make build` and `make demo` do steps 2–4, 7 and 9 in one go. On Windows, use the commands above, since `make` is usually not installed.

| Problem | Fix |
|---|---|
| `chemrag: command not found` | Activate the venv (step 2), or run `python -m chemrag.cli ...`. |
| gte-large download fails or is slow (proxy, firewall) | The build continues in lexical-only mode. Retry later, or set `CHEMRAG_EMBED_MODEL=thenlper/gte-base` (~220 MB). |
| Low memory | Use `CHEMRAG_EMBED_MODEL=thenlper/gte-base`, or pass `--no-vectors` to `ask`. |
| OpenAI errors (429 / quota / no credit) | Answers still come back (agents fall back to rules, with an `llm_unavailable` warning), or use `--no-llm`. |
| Changed the CSV or the synonym YAML | Run `chemrag build` again; it detects the change via the file hash. Use `--force` to rebuild everything. |

### Everyday commands

```bash
chemrag ask "..." [--json] [--no-llm] [--no-vectors] [--assume-best] [--limit 20] [--page 2]
chemrag replay <request_id>     # re-runs the logged SQL with no LLM and checks the results are identical
chemrag eval --holdout          # held-out paraphrase set
chemrag graph                   # regenerate docs/graph.md from the compiled LangGraph
```

| Situation | Behaviour |
|---|---|
| `OPENAI_API_KEY` unset or `--no-llm` | All agents use deterministic rules. Every example below works this way. |
| OpenAI call fails (429, timeout, bad output) | That agent falls back to rules and adds an `llm_unavailable` warning. A hard failure (bad key, unknown model, no network) switches the LLM off for the rest of the run, so later agents don't wait on retries. |
| gte-large or Chroma unavailable | Entity resolution uses spelling-based (lexical) matching only, with a `vector_unavailable` info warning. |
| Product names | `chemrag build --products` also embeds the ~33k product names. This is opt-in (see §6). |

---

### Choosing an OpenAI model (budget)

The LLM does three small jobs per question: classify and split the question, extract entity mentions, and write 2–6 bullet points from a fact sheet. Counting, SQL and citations never touch it. So a small model is enough.

| Model | Use when | Approx. cost per question* |
|---|---|---|
| `gpt-5-mini` (**default**) | Best balance; reliable structured outputs | ~$0.003–0.006 |
| `gpt-5-nano` | Cheapest; fine for a demo, slightly weaker on unusual phrasing | ~$0.001 |

\*About 3 calls per question, ~4k input and ~1–2k output tokens with `reasoning_effort=low`. Prices change, so check OpenAI's pricing page. `chemrag ask ... --json` reports the tokens each answer used in `meta.llm_tokens`. `chemrag eval --quick` (10 questions) costs roughly $0.03–0.06 with `gpt-5-mini`; the full set (37 questions) about $0.10–0.25.

---

## 2. Architecture

The orchestrator is a **LangGraph `StateGraph`**. Every agent is a node. All routing is done by small, deterministic, unit-tested router functions; **the LLM never decides which node runs next and never writes SQL**. [`docs/graph.md`](docs/graph.md) is generated from the compiled graph with `chemrag graph`.

```mermaid
flowchart LR
  Q([question]) --> G[guard] --> P[planner]
  P -->|in scope| X[extractor] --> R[resolver]
  P -->|unrelated / refusal| S
  R -->|all resolved| QA[query agent]
  R -->|product mention| RT[retrieval] --> QA
  R -->|ambiguous| C[clarify]
  RT -->|ambiguous| C
  C -->|"interrupt() + user choice"| QA
  C -->|non-interactive| F
  QA -->|next sub-question| X
  QA -->|done| S[synthesizer] --> V[verifier] --> F[finalize] --> A([answer JSON])
  R -. names .-> VDB[(Chroma embedded<br/>gte-large vectors)]
  RT -. names .-> VDB
  QA -. parameterised SQL .-> DB[(DuckDB<br/>read-only)]
```

| Node | What it does | LLM? |
|---|---|---|
| **guard** | Caps input at 1,000 characters and strips instruction-like sentences ("ignore previous instructions…"). | no |
| **planner** | Checks scope (in scope / medical advice / unrelated), classifies intent (lookup, list, compare, summarize, trend, data_quality, coverage, out_of_scope) and splits the question into ≤4 sub-questions. | LLM, with keyword rules as fallback |
| **extractor** | Finds CAS numbers, years and date ranges, discontinued/removed wording, group-by and top-N, plus entity mentions (exact alias matches, cue phrases, quoted product names). Follow-up sub-questions inherit the previous sub-question's constraints. | rules first; the LLM can only *add* mentions it quotes verbatim from the question |
| **resolver** | Maps each mention to canonical IDs, in order: CAS check-digit path → exact/alias match → rapidfuzz spelling score fused with gte-large/Chroma similarity. Returns ranked candidates and flags ambiguity. | no |
| **retrieval** | Turns product-name mentions into `CDPHId`s using lexical matching, plus vectors when the products index exists. | no |
| **clarify** | Asks which candidate was meant, using LangGraph `interrupt()` in the interactive CLI. With `--json` or `--assume-best` it either returns a `clarification` response or proceeds with the top candidate and a warning. | no |
| **query** | Builds a typed `Filters` object and runs fixed SQL templates. Also handles out-of-range dates, unmatched entities, empty results (re-runs the count with one constraint dropped at a time to show which one emptied the result), dominance, trade-secret and synonym warnings. | no |
| **synthesizer** | Builds a numbered fact sheet (F1..Fn) from query results. The **short answer is templated**. Details are optional LLM bullets citing `[F#]`, followed by a deterministic listing. | LLM (details only) |
| **verifier** | Checks that every number is in the facts or rows, every bullet cites existing facts, and the text has no medical advice or prompt echo. If a check fails, the LLM text is dropped and the templated details are used. | no |
| **finalize** | Builds the response JSON and computes confidence with a documented formula. Saves `runs/<id>.json` for replay. | no |

**State.** One Pydantic `TurnState` (`chemrag/state.py`) flows through the graph. List fields are append-only, so the `trace` *is* the query plan, and a clarification appends a newer resolution instead of overwriting the old one.

---

## 3. Data model & cleaning decisions

Source: 114,635 rows × 22 columns. The ETL (`chemrag/etl/build_db.py`) writes these DuckDB tables: `raw_rows`, `fact_report`, `dim_product`, `dim_chemical`, `dim_chemical_group`, `dim_company`, `dim_brand`, `dim_primary_category`, `dim_subcategory`, `entity_alias`, `dq_issues`, `dataset_meta`. Peak RAM is about 600 MB and the build takes about 25 s.

| Issue found while profiling | Decision |
|---|---|
| No column is a unique row key | `row_id` = 1-based CSV row number, stable and checkable with `sed -n`. The CSV's SHA-256 is stored, and `build` re-runs if the file changes. |
| 254 fully identical duplicate rows | Flagged with `is_exact_dup` and excluded from every query. |
| ~9.3k "duplicate" (CDPHId, CSFId, ChemicalId) rows | These are **re-categorisations**: the same product listed under several categories. They are kept, and a category filter matches a product if *any* of its rows is in that category. |
| `ChemicalId` is **not** a chemical | It is a product-chemical *report* id. A chemical is identified by `CasId` → canonical **chemical group**. |
| Dirty CAS values (`"13463-67-7 "`, `"CAS #79-81-2"`, `"93 15 2"`, `"79812"`, `"asdf"`, an EC number) | Normalised and **check-digit validated** (`etl/cas.py`). Unrecoverable values are filled from the `CasId` when it maps to exactly one valid CAS. Every repair is logged in `dq_issues`. Nothing is guessed: `"50-78-25"` stays invalid. |
| `"Trade Secret"` pseudo-chemical (668 rows, CAS `"0"`) | Counted as a product report, excluded from chemical lists, and reported separately in a warning. |
| Chemical synonyms (14 names map to >1 CAS; 7 CAS map to >1 name) | A curated [`config/chemical_groups.yaml`](config/chemical_groups.yaml) defines **groups** (same substance, e.g. Retinyl palmitate = Vitamin A palmitate = Retinol palmitate) and **families** (related but distinct substances, e.g. retinoids). A name query covers its group, the answer lists the reported names, and related families are mentioned in a warning but never silently added. |
| Removal dates in 2103/2104 (115 rows) | Cleaned date set to NULL (raw value kept). Those rows still count as "removed", with the date treated as unknown. |
| Discontinued before first report (2,866 rows), removed before created (192 rows) | Kept and flagged. |
| `"NA"`/`"None"`/`"N/A"` brands, trailing whitespace, the `"Test"` company record | Placeholder brands become NULL, whitespace is trimmed, and the test record is flagged. |
| Titanium dioxide is in 87% of products | Evidence is sampled evenly across chemicals, and a `dominant_chemical` warning appears in chemical listings. |

**Counting vocabulary.** A *product* is a distinct `CDPHId`. A *report record* is a distinct `ChemicalId`. A *row* is a `row_id`, used for citations.

**Default semantics** (echoed in `assumptions[]` whenever they are used):
- *discontinued* means the product has a DiscontinuedDate.
- *removed/reformulated* means the product-chemical record has a ChemicalDateRemoved.
- *reported in year X* means **InitialDateReported**; you can ask for "most recently reported" instead.
- *contains* includes chemicals that were later removed, and the answer gives the count.
- *last N years* is anchored to the dataset's last date (2020-06-23), not today's date.

---

## 4. Retrieval: gte-large + embedded Chroma

- **What is embedded:** only **names**. That is ~3.8k entity aliases (chemicals and synonyms, companies, brands, categories) and, optionally, ~33k product names. The 114k rows are never embedded: SQL answers row-level questions exactly, and Chroma only answers *"which entity did the user mean?"*.
- **Model:** `thenlper/gte-large` (1024-dim) via sentence-transformers on CPU. It loads lazily, so questions answered by exact, alias or CAS matches never load it. Settings are `max_seq_length=64`, `inference_mode`, and capped threads.
- **Score fusion:** `0.65·lexical + 0.35·semantic`. gte-style cosine scores are compressed, so the semantic score is rescaled against a "random pair" floor measured at build time.
  - **Resolved** if score ≥ 0.75 and the runner-up is not within 0.05.
  - **Ambiguous** (clarification) if two *different* entities are within 0.05 of each other.
  - **Not found** below 0.75, with suggestions.

  These thresholds are in `chemrag/settings.py`.
- **Chroma:** runs embedded (`chromadb.PersistentClient(path=...)`) with no server. One on-disk store holds an `entities` collection and an optional `products` collection, with cosine HNSW indexes. Each collection is loaded only when it is queried. Filters on list metadata (`brand_keys`, `company_keys`) use `$contains`. Qdrant was used first and replaced by Chroma; see §6 for why.

---

## 5. Evaluation

`make eval` runs [`evals/golden.yaml`](evals/golden.yaml), `make eval-quick` runs a fixed 10-question subset of it (one question per behaviour the brief asks for; listed in `evals/run_eval.py`), and `make eval-holdout` runs [`evals/holdout.yaml`](evals/holdout.yaml). Expected values come from **hand-written pandas over the raw CSV** ([`evals/ground_truth.py`](evals/ground_truth.py)), which shares no code with the system. Reports are in [`evals/results/`](evals/results).

Each run checks:
- response type,
- intents,
- resolved entities,
- exact counts, chemical sets, trends, top-N lists and comparisons,
- **citation precision**: every cited row is re-checked against the raw CSV,
- required warnings,
- refusal and clarification behaviour.

| Set (no-LLM mode) | Cases | Passed | Answer correctness | Citation precision | Entity resolution |
|---|---|---|---|---|---|
| Golden (also used during development, so in-sample) | 37 | 37 | 1.00 | 1.00 | 1.00 |
| Held-out paraphrases, **first run before any fixes** | 14 | **10** | 0.60 | 1.00 | 0.86 |
| Held-out after the generic fixes it exposed (now contaminated) | 14 | 14 | 1.00 | 1.00 | 1.00 |

**First LLM-mode run** (`eval --quick`, gpt-4o-mini, gte-large vectors on): **7/10**. All three failures came from the LLM planner rewriting or splitting questions: it shortened "Nail Polish and Enamel" to "Nail Polish", split "formaldehyde **or** CAS 75-07-0" into two questions, and split a comparison apart. The planner now lets the deterministic splitter fix sub-question boundaries and wording, and the LLM decides scope (plus intent only where the keyword rules have no explicit signal). Three regression tests replay that exact LLM behaviour. Re-run `chemrag eval --quick` to get the updated LLM-mode score.

The honest generalisation number is the **10/14 first run**
([`evals/results/holdout_first_run_untuned.md`](evals/results/holdout_first_run_untuned.md)). It failed on:
- misspelled chemicals with no domain keyword,
- "year by year" phrasing,
- "data problems",
- an unquoted product name.

All four were fixed with general rules, not per-question patches.

The golden set covers:
- every intent;
- misspellings, CAS-only and bare-digit CAS queries, and an invalid CAS check digit;
- synonyms and chemical families;
- out-of-range and invalid dates;
- ambiguous brands and generic product names;
- multi-part and follow-up questions;
- trends, comparisons and top-N;
- empty results with diagnostics;
- data quality and coverage;
- medical refusal, out-of-scope questions, prompt injection and missing entities.

**Tests:** `make test` runs 94 pytest tests covering:
- CAS normaliser cases;
- ETL invariants;
- SQL values are always bound as parameters, never pasted into the SQL text;
- query tools, with replay producing identical results;
- the resolver;
- the verifier;
- graph routers;
- end-to-end graph runs, including interactive clarification resume and a scripted fake LLM (both a grounded narrative and a hallucinated number that triggers the fallback).

---

## 6. Resource budget (8 GB RAM, CPU only)

All numbers below were measured in the build sandbox with the 512-dim offline embedder. gte-large (1024-dim) roughly doubles the vector part and adds the model itself.

| Step | Measured peak RSS |
|---|---|
| `build` ETL step (separate process) | ~0.6 GB |
| `build` index step, entities + 33k products into Chroma | ~0.34 GB (Qdrant: 0.60 GB) |
| `ask`, lexical resolution | ~0.35 GB |
| `ask` with vectors, entity question | ~0.34 GB |
| `ask` on a product-name question with the products collection | **~0.41 GB** (embedded Qdrant: 1.43 GB) |
| gte-large fp32 on CPU (estimate) | ~1.4–1.8 GB, loaded only for fuzzy matches |

**Why Chroma replaced Qdrant.** Embedded Qdrant loads a whole collection into RAM as Python objects. At 33k product vectors that was about 1 GB at 512-dim, so the products index could not fit next to gte-large on an 8 GB laptop. Chroma keeps its HNSW index on disk and loads a collection only when it is queried, so the same collection costs about 0.1 GB.

The products collection is still **opt-in** (`make build PRODUCTS=1`), but only because embedding 33k names takes about 10–25 minutes on a laptop CPU, not because of memory. Without it, product names are matched lexically, which already passes the golden and held-out product cases.

DuckDB is limited to `memory_limit=1GB` and 2 threads. If memory is still tight:
1. set `CHEMRAG_EMBED_MODEL=thenlper/gte-base`, then
2. use `--no-vectors`.

## 7. Design decisions & trade-offs

- **DuckDB for facts, Chroma for names.** Numbers always come from SQL. Vectors only help pick *which* entity was meant.
- **No text-to-SQL.** There is a fixed tool set (`find_products`, `count_products`, `chemicals_for`, `trend_by_year`, `dataset_coverage`, `dq_summary`, plus `compare`/`summarize` compositions). SQL is built from enum-whitelisted fragments with bound parameters, and the connection is read-only with external access disabled.
- **LangGraph used narrowly.** It provides a typed graph, conditional edges, `interrupt()` for clarification, and a checkpointer. There are no LangChain agents, retrievers or LLM wrappers. The LLM is called through a one-method `LLMClient` interface. The default is OpenAI (`openai` SDK, Structured Outputs via `chat.completions.parse` with Pydantic schemas). Gemini remains an optional alternative (`CHEMRAG_LLM_PROVIDER=gemini`).
- **Rules first, LLM second.** The system answers every example correctly with no LLM at all. The LLM improves paraphrase coverage and the readability of the details, and the verifier keeps its text grounded.
- **Curated synonym groups instead of embedding-based chemical synonymy.** There are only 123 names, so a reviewed YAML file is more correct than nearest neighbours.

## 8. Known limitations

- **Neither gte-large nor a live LLM was run in the build environment.**
  - Hugging Face and the OpenAI API were blocked by that sandbox's network policy. The vector path was exercised end to end with the offline `hash-ngram` embedder (same Chroma code, fusion and calibration).
  - The OpenAI client is tested with a fake SDK client (request shape, parsing, parameter fallback, circuit breaker), and the graph with a scripted fake LLM.
  - Run `chemrag doctor` on your machine. It confirms the gte-large download, lists the model ids your key can use, and makes one test structured-output call. Then run `chemrag eval` (without `--no-llm`) to get LLM-mode numbers.
- The rule-based extractor is pattern-driven. Unusual phrasing may be missed in no-LLM mode, which the held-out set shows. LLM mode is meant to cover that.
- Dates are handled at year granularity ("June 2019" becomes 2019).
- Comparing more than one dimension at once (e.g. companies × categories) is not supported; `compare` uses the first entity type that has two or more values.
- Embedded Chroma is designed for a single process writing at a time, so don't run `chemrag build` while another `chemrag` process is using the same index.
- Category membership follows the source data. A product re-categorised across subcategories counts in each of them.

## 9. Layout

```
chemrag/
  graph.py            LangGraph wiring + routers          orchestrator.py  entry point, replay
  state.py            TurnState (graph state)             schemas.py       all Pydantic contracts
  agents/             guard, planner, extractor, resolution (resolver/retrieval/clarify),
                      query_agent, synthesizer, verifier, finalize
  etl/                build_db.py, cas.py, normalize.py
  query/              filters.py (Filters → parameterised WHERE), tools.py (SQL templates)
  retrieval/          resolver.py, embed.py (gte-large / hash-ngram), vector_store.py (Chroma), build_index.py
  llm/                base.py (interface + LLM schemas), openai_client.py, gemini_client.py (optional), null_client.py, prompts/*.md
  render/cli_render.py, cli.py
config/chemical_groups.yaml   evals/   tests/   docs/   scripts/profile_data.py   data/raw/
```
