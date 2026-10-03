# Interview Preparation: ChemRAG

**Part 1** has 25 questions about this demo, each with an answer you can give.
**Part 2** has 20 practice questions per topic (GenAI, LLMs, RAG, embeddings, vector databases, agents and more), with questions only, so you can test yourself.

---

## Part 1: Questions about the demo, with answers

### Problem and approach

**1. In one minute, what did you build?**
A multi-agent assistant that answers plain-English questions about the California cosmetics chemical-disclosure dataset (114,635 rows). Every answer comes with:
- the exact source rows as evidence,
- a query plan that includes the exact SQL,
- assumptions, warnings and a confidence score.

The core rule is that the LLM understands the question and writes the sentences, but never produces a number. SQL does all the counting, and a verifier checks the final text.

**2. Why didn't you use classic RAG (chunk, embed, retrieve top-k)?**
Classic RAG fits this data poorly, for three reasons:
- The data is a table, and every question in the brief is a counting, filtering or trend question.
- Vector search returns the top-k most similar items, so it can't count. "How many products contain acetaldehyde?" would come back as "10" (k), not 30.
- 81% of rows are titanium dioxide, so the top-k results would be near-duplicates.

So answers are grounded in SQL rows, and vectors are used only to match messy names ("acetaldehide" → Acetaldehyde). It is still RAG: SQL and vectors retrieve, the facts augment the prompt, and the LLM generates.

**3. What are the agents, and which ones use the LLM?**
There are seven agents:
- **Planner:** decides scope and intent.
- **Extractor:** pulls out entities, CAS numbers and dates.
- **Resolver:** maps names to database IDs.
- **Retrieval:** finds product IDs from product names.
- **Query:** runs the SQL.
- **Synthesizer:** writes the answer.
- **Verifier:** checks every number.

There are also three support nodes (guard, clarify, finalize), making 10 graph nodes. Only the planner, extractor and synthesizer call the LLM, and each falls back to rules if the LLM is unavailable.

**4. Why LangGraph instead of a plain Python loop?**
The flow has branches (clarify, refuse, product retrieval), a loop (one pass per sub-question) and a pause for user clarification. LangGraph gives me:
- a typed shared state,
- conditional edges,
- `interrupt()` for human-in-the-loop clarification,
- a checkpointer to resume after the pause,
- a diagram generated from the code.

I deliberately did not use LangChain agents: the routing functions are plain, deterministic and unit-tested.

**5. How does a question flow through the system?**
Using "Which products contain acetaldehide, and how many were discontinued?":
1. The guard checks for prompt injection.
2. The planner splits it into two sub-questions: a list and a count.
3. The extractor finds "acetaldehide".
4. The resolver maps it to Acetaldehyde with a score of 0.92, using fuzzy and vector matching.
5. SQL finds 30 products.
6. The loop runs again; sub-question 2 inherits the chemical and adds "discontinued", giving 13.
7. The synthesizer turns these into numbered facts and an answer.
8. The verifier checks the numbers.
9. The response includes evidence rows and the SQL.

### Data and ingestion

**6. What data problems did you find, and how did you handle them?**

| Problem | Handling |
|---|---|
| No unique row key | `row_id` = the row's position in the CSV |
| 254 exact duplicates | Flagged and excluded from counts |
| ~9.3k rows that *look* like duplicates | Kept; they are the same product listed under several categories |
| Dirty CAS numbers | Repaired only when the CAS check digit proves the repair is correct |
| "Trade Secret" placeholder chemical | Counted as a product, kept out of chemical lists |
| Removal dates in 2103/2104 | Set to empty; the raw value is kept and the row is flagged |
| "NA"/"None" brand values | Set to empty |
| Synonym chemical names | Grouped using a curated YAML file |

The rules are to never guess, never delete, and log every change.

**7. Did you use an LLM to clean or ingest the data?**
No. Ingestion is deterministic Python and DuckDB, so the same file always produces the same database. An LLM "cleaning" data can silently change values, and that is not auditable. The only human-judgment step is the synonym file, which I reviewed by hand.

**8. How do you handle CAS numbers?**
CAS numbers have a built-in check digit. Messy values such as `"CAS #79-81-2"`, `"93 15 2"` or `"79812"` are reformatted, then accepted only if the checksum passes. Unrecoverable values become empty and are flagged. A user who types "75-07-1" gets *"invalid check digit, did you mean 75-07-0?"*.

**9. How would ingestion scale to 100× the data?**
- Move the cleaning into DuckDB SQL, which streams data and can work beyond memory.
- Clean *distinct values* instead of rows. There are only 125 distinct CAS strings, so that is about 1,000× less work.
- Load incrementally using a file-hash manifest.
- Quarantine bad rows instead of stopping the whole build.

Embedding cost barely grows, because I embed *names*, not rows.

### Retrieval and resolution

**10. What exactly is embedded, and with what?**
About 3.8k entity names (chemicals and their synonyms, companies, brands, categories), plus optionally 33k product names. They are embedded with `thenlper/gte-large` (1024 dimensions) running on CPU, and stored in an embedded Chroma database. The 114k rows are never embedded.

**11. How does the resolver decide what the user meant?**
It tries three paths in order:
1. **CAS path:** check-digit validation.
2. **Exact or alias match:** e.g. "BPA" → Bisphenol A.
3. **Fuzzy plus vector:** 65% rapidfuzz spelling score and 35% calibrated semantic score.

The thresholds are:
- ≥0.90: resolved.
- 0.75–0.90: resolved, with an assumption shown to the user.
- Two different entities within 0.05 of each other: ambiguous, so it asks the user.
- Below 0.75: not found, with suggestions.

**12. Why weight spelling over meaning?**
Most user errors are typos. Also, gte-style models give even unrelated short strings fairly high similarity, so I rescale semantic scores using a "random pair" baseline measured at build time.

**13. Why Chroma instead of Qdrant?**
I measured both. In embedded mode, Qdrant held the vectors in RAM: a product-name question peaked at 1.43 GB. Chroma keeps its index on disk and peaked at 0.41 GB. With an 8 GB laptop and no servers allowed, that decided it. The swap only touched one small wrapper class.

**14. How do you handle ambiguity, like the brand "Pure"?**
Several brands score about the same (Pure Ice, Pure Cosmetics, Perfectly Pure), so the clarify node asks "Which brand did you mean?" and lists product counts. In the CLI the graph pauses with LangGraph `interrupt()` and resumes from the same point after the user picks. In JSON mode it returns a clarification response, or, with `--assume-best`, proceeds with the top match and a warning.

### Safety, grounding and correctness

**15. How do you prevent hallucinated numbers?**
There are three layers:
1. The short answer is filled into a template directly from query results.
2. The LLM writes the detail bullets from numbered facts and must cite `[F#]` on each one.
3. The verifier rejects any number that isn't in the facts or results. If the LLM text fails the check, it is replaced with the templated text.

A test injects "31 products" where the truth is 30, and the system still answers 30.

**16. How do you protect against prompt injection?**
There are six layers:
- The guard removes instruction-like sentences.
- User text is wrapped as untrusted data in prompts.
- LLM output must match a strict schema.
- The LLM has no tool authority; deterministic routers decide what runs.
- The database is read-only, and SQL values are bound as parameters.
- The verifier checks for echoes of the system prompt.

**17. Can the LLM write or alter SQL?**
No. There are six pre-written, parameterised query tools. The LLM's output only influences which typed filters are set, and every value is bound as a parameter. A test confirms that a malicious string never appears in the SQL text.

**18. What happens with "discontinued in 2024" or "is this safe for my kids?"**
- For 2024, the query agent checks the data's date coverage and answers "no data in range", giving the real range (discontinued dates end 2020-06-12). It does not invent results.
- Medical questions are refused politely, with the factual count still given ("it is reported in 32,054 products") and a pointer to CDPH resources.

**19. What if the LLM is down or slow?**
Every LLM step has a rules fallback. Hard failures (bad key, unknown model, no network) trip a circuit breaker, so the remaining agents skip the LLM immediately. With no LLM at all, the system still passes 37/37 golden questions.

### Evaluation

**20. How did you evaluate it?**
- **Golden set:** 37 questions covering every intent and edge case.
- **Quick set:** a 10-question subset, used for LLM mode to save cost.
- **Held-out set:** 14 paraphrased questions written after the build.

Expected answers come from hand-written pandas over the **raw CSV**, sharing no code with the system. Metrics are answer correctness, citation precision (every cited row re-checked), entity resolution, intent, warning recall and refusal/clarification accuracy. There are also 98 unit and integration tests.

**21. What were the results?**
- **LLM mode** (gpt-4o-mini + gte-large): 10/10, with every metric at 100%.
- **Rules only:** 37/37 golden.
- **Held-out set:** 10/14 on the first untuned run, now 14/14.

**22. Tell me about a failure you found and fixed.**
My first LLM-mode run scored 7/10. All three failures came from the LLM planner rewriting and splitting questions:
- it shortened "Nail Polish and Enamel" to "Nail Polish",
- it split "formaldehyde or CAS 75-07-0" into two questions,
- it split a comparison apart.

Citation precision stayed at 100%, so no wrong facts were produced. I moved sub-question splitting into deterministic code, limited the LLM to scope and fuzzy intent, and added regression tests that replay that exact LLM behaviour. The re-run scored 10/10.

**23. Isn't 37/37 suspicious?**
The golden set was also my development set, so it is in-sample, and I say so. That is why I wrote a held-out set after the build and reported its first untuned score (10/14) rather than only the final one. I also found and fixed two bugs in the brief's own example questions while reviewing: "X **or** CAS Y" was being combined with AND, and "acetone" was falsely matching Spironolactone.

### Engineering and trade-offs

**24. Why gpt-5-mini / gpt-4o-mini rather than a bigger model?**
The LLM only classifies, extracts names and phrases bullet points. All the hard work is done by SQL and code. A small model is enough at well under a cent per question, and each answer reports the tokens it used.

**25. What are the limitations, and what would you do next?**

Limitations:
- Dates work at year level only.
- Comparisons cover one dimension at a time.
- Rules-only mode relies on patterns, so unusual phrasing needs the LLM.
- Ingestion uses pandas, which is fine up to about 1M rows.
- CLI only, no API.

Next steps:
- Run the full held-out set with the LLM.
- Add month-level dates.
- Move ingestion into DuckDB with incremental loads.
- Collect real user questions as a new test set.
- Add an optional API or UI.

**Bonus quick answers**

| Question | Answer |
|---|---|
| Why DuckDB? | Embedded, fast group-bys, can be opened read-only and memory-capped |
| Why Pydantic? | Every agent hand-off is typed and validated, including LLM output |
| How is it reproducible? | `chemrag replay <id>` re-runs the logged SQL and checks the results are identical |
| How is confidence computed? | The lowest of intent and match scores, multiplied by documented penalties per warning |
| What is `row_id`? | The CSV line position, so any citation can be checked by hand |

---

## Part 2: Practice questions by topic (questions only)

### A. Generative AI and LLM fundamentals
1. What is a large language model, and how does it generate text?
2. Explain the transformer architecture and self-attention in simple terms.
3. What are tokens, and why do token counts matter for cost and context limits?
4. What is a context window? What happens when the input exceeds it?
5. Explain temperature, top-p and top-k sampling. When would you set temperature to 0?
6. What is the difference between a base model, an instruction-tuned model and a reasoning model?
7. What is hallucination, and why do LLMs hallucinate?
8. Compare pre-training, fine-tuning, instruction tuning and RLHF.
9. When would you fine-tune a model instead of using RAG or prompting?
10. What are LoRA and QLoRA, and why are they popular?
11. What is the difference between encoder-only, decoder-only and encoder-decoder models?
12. How do you choose between a small and a large model for a task?
13. What is quantisation, and how does it affect quality and memory?
14. Explain zero-shot, one-shot and few-shot learning.
15. What are reasoning models, and what is "reasoning effort"?
16. Why are LLMs non-deterministic even at temperature 0, sometimes?
17. What is knowledge cutoff, and how do you work around it?
18. What are the main risks of using LLMs in regulated domains?
19. How would you explain an LLM's limitations to a non-technical stakeholder?
20. Open-source versus API models: what trade-offs would you consider?

### B. Prompt engineering and structured outputs
1. What makes a good system prompt? What did you put in yours?
2. How do structured outputs (JSON schema) improve reliability?
3. What is the difference between JSON mode and strict schema-constrained output?
4. How do you validate LLM output, and what do you do when validation fails?
5. Explain chain-of-thought prompting. When does it help, and when does it hurt?
6. How do you stop an LLM from inventing facts in its answer?
7. How do you make prompts robust to prompt injection?
8. What is prompt chaining, and when would you split one prompt into several?
9. How do you version, test and review prompts like code?
10. How do you write prompts that work across different model providers?
11. What are few-shot examples, and how do you choose them?
12. How do you get a model to say "I don't know"?
13. How do you control output length and format?
14. What are function calling and tool use? How do they differ from structured output?
15. How do you evaluate whether a prompt change made things better or worse?
16. What is prompt caching, and how does it reduce cost and latency?
17. Why would you ask the LLM to cite fact IDs like [F1]?
18. How do you handle a model refusing a legitimate request?
19. How do you keep prompts short without losing important constraints?
20. Give an example of a prompt you rewrote because it failed. What changed?

### C. Retrieval-augmented generation (RAG)
1. What is RAG, and what problem does it solve?
2. Walk through a typical RAG pipeline end to end.
3. When is RAG a bad fit? Give examples.
4. How do you choose a chunk size and chunk overlap?
5. Compare fixed-size, recursive, semantic and document-structure chunking.
6. What is hybrid search (BM25 plus vectors), and why use it?
7. What is re-ranking, and how do cross-encoders differ from bi-encoders?
8. How do you decide top-k?
9. What is "lost in the middle", and how do you mitigate it?
10. How do you add citations to RAG answers and make sure they are correct?
11. What is structured or text-to-SQL RAG? How does it differ from document RAG?
12. What are GraphRAG and knowledge-graph RAG, and when are they useful?
13. How do you handle tables, PDFs and images in a RAG system?
14. What is query rewriting or expansion (HyDE, multi-query)?
15. How do you keep a RAG index fresh when the source data changes?
16. How do you handle metadata filtering (date, brand, category) in retrieval?
17. What are the main failure modes of RAG, and how do you detect each?
18. How do you evaluate retrieval separately from generation?
19. How would you combine SQL retrieval and vector retrieval in one system?
20. What is agentic RAG, and how is it different from a fixed pipeline?

### D. Embeddings and semantic search
1. What is an embedding, and what does "semantic similarity" mean?
2. Compare cosine similarity, dot product and Euclidean distance.
3. Why normalise embeddings?
4. How do you choose an embedding model? What benchmarks (e.g. MTEB) help?
5. What are the trade-offs of embedding dimension size (384 vs 768 vs 1024)?
6. Why might two unrelated short strings get a high similarity score?
7. How do you calibrate or threshold similarity scores?
8. What is the difference between symmetric and asymmetric search?
9. When would you use query/passage prefixes in embedding models?
10. How do embeddings handle typos, abbreviations and synonyms?
11. What are sparse embeddings (e.g. SPLADE), and how do they compare to dense ones?
12. How do you fine-tune an embedding model for your domain?
13. How do you run embedding models efficiently on CPU?
14. What is the cost of re-embedding when you change models?
15. How do you combine lexical (fuzzy) and semantic scores?
16. What are multilingual embeddings, and when do you need them?
17. Why not embed every row of a structured table?
18. How do you evaluate embedding quality for entity matching?
19. What are late-interaction models like ColBERT?
20. How do you detect embedding drift over time?

### E. Vector databases
1. What does a vector database do that a normal database doesn't?
2. Explain approximate nearest neighbour (ANN) search.
3. How does the HNSW index work at a high level?
4. Compare HNSW, IVF and product quantisation.
5. Compare Chroma, Qdrant, FAISS, pgvector, Pinecone and Weaviate.
6. When would you choose embedded mode versus a server deployment?
7. How do metadata filters work with vector search? What are pre- and post-filtering?
8. How do you size memory for a vector index?
9. What is the recall/latency trade-off in ANN search?
10. How do you update or delete vectors safely?
11. How do you back up and version a vector index?
12. How do you handle multi-tenancy in a vector database?
13. What is vector quantisation, and how does it save memory?
14. When is brute-force search good enough?
15. How do you monitor vector database performance?
16. How would you migrate from one vector database to another?
17. Why did embedded Qdrant use more RAM than Chroma in this project?
18. How do you keep the vector index consistent with the source database?
19. What security concerns apply to vector databases?
20. Do you always need a vector database for RAG? Why or why not?

### F. Agents and multi-agent orchestration (LangGraph)
1. What is an AI agent? How is it different from a single LLM call?
2. What is the ReAct pattern?
3. Compare single-agent and multi-agent designs.
4. What is LangGraph? Explain nodes, edges, state and conditional edges.
5. How does LangGraph differ from LangChain chains and agents?
6. What is a checkpointer, and why is it needed for human-in-the-loop?
7. How does `interrupt()` work, and how do you resume a graph?
8. Should routing decisions be made by the LLM or by code? Why?
9. How do agents share state? What are reducers?
10. How do you stop infinite loops in an agent graph?
11. How do you add tools to an agent safely?
12. How do you debug and trace a multi-agent system?
13. What are planner–executor and supervisor patterns?
14. How do you handle a failing agent: retries, fallbacks, circuit breakers?
15. How do you test an agent graph deterministically?
16. Compare LangGraph, CrewAI, AutoGen and plain Python orchestration.
17. What is the Model Context Protocol (MCP), and why does it matter for agents?
18. How do you control agent cost and latency?
19. How do you keep an agent from taking unsafe actions?
20. When is a multi-agent system over-engineering?

### G. Hallucination, grounding and guardrails
1. What is grounding, and how do you measure it?
2. Name techniques to reduce hallucination.
3. What is a verifier or critic step, and how would you design one?
4. LLM-as-judge versus deterministic checks: when do you use each?
5. What is prompt injection? Compare direct and indirect injection.
6. How do you defend against data exfiltration through an LLM?
7. What are guardrail frameworks (e.g. NeMo Guardrails, Llama Guard), and what do they do?
8. How do you handle medical, legal or financial advice requests?
9. How do you detect and handle PII in inputs and outputs?
10. What is jailbreaking, and how do you test for it?
11. How do you make an LLM refuse safely without being unhelpful?
12. How do you handle the case where the answer is not in the data?
13. How do you show uncertainty or confidence to users?
14. What is citation precision, and why does it matter?
15. How do you red-team an LLM application?
16. How do you stop an LLM from inventing numbers in reports?
17. What logging do you need for audits?
18. How do you handle toxic or biased outputs?
19. What does "least privilege" mean for LLM tools?
20. How do you keep a safe default when the LLM fails?

### H. Evaluating LLM and RAG systems
1. How do you build a golden evaluation set?
2. Why do you need a held-out set, and how do you keep it honest?
3. What metrics would you use for a RAG system?
4. Explain faithfulness, answer relevance, context precision and context recall.
5. What are RAGAS, TruLens and DeepEval?
6. How do you compute ground truth independently of the system?
7. How do you evaluate citations automatically?
8. How do you evaluate clarification and refusal behaviour?
9. Offline versus online evaluation: what is the difference?
10. How do you run A/B tests for LLM features?
11. How do you evaluate on a limited budget?
12. How do you detect regressions when you change a prompt or model?
13. How reliable is LLM-as-judge, and how do you calibrate it?
14. How many evaluation examples are enough?
15. How do you handle non-determinism in evaluation runs?
16. What is the difference between in-sample and out-of-sample accuracy here?
17. How do you collect real user feedback for evaluation?
18. How do you evaluate latency and cost alongside quality?
19. Turn a failed evaluation case into a regression test: how?
20. What would make you distrust a 100% evaluation score?

### I. Structured data, text-to-SQL and DuckDB
1. What is text-to-SQL, and what are its main risks?
2. Why use parameterised query templates instead of LLM-generated SQL?
3. How do you prevent SQL injection in an LLM application?
4. When would free-form text-to-SQL be acceptable?
5. What is DuckDB, and how does it compare to SQLite and Postgres?
6. Why are columnar databases fast for analytics?
7. How do you design a star schema (facts and dimensions)?
8. How do you count correctly when rows are duplicated at different grains?
9. What does COUNT(DISTINCT) do, and why did it matter here?
10. How do you handle date ranges and out-of-range queries?
11. How do you make queries reproducible and auditable?
12. How do you limit query cost and timeouts?
13. How do you explain a SQL result in natural language safely?
14. How do you handle "X or Y" versus "X and Y" filters correctly?
15. What is a semantic layer, and how does it help LLM querying?
16. How do you test SQL query templates?
17. How do you paginate large result sets?
18. How do you design schemas that are easy for LLMs to use?
19. How would you join structured data with unstructured documents?
20. How do you handle schema changes in the source data?

### J. Data engineering, entity resolution and data quality
1. What is entity resolution, and why is it hard?
2. Compare exact, fuzzy and semantic matching.
3. What are Levenshtein distance and token-set ratio?
4. How do you choose thresholds for "match", "ambiguous" and "no match"?
5. How do you build and maintain a synonym or alias table?
6. How do you validate identifiers that have checksums (CAS, ISBN, IBAN)?
7. How do you handle duplicate records?
8. What is data lineage, and why does it matter?
9. How do you design an idempotent ETL pipeline?
10. How do you handle invalid or impossible values without losing information?
11. What is a data quality log, and what should it contain?
12. How do you detect schema drift?
13. How do you scale ETL from 100k to 100M rows?
14. Batch versus incremental loading: when do you use each?
15. How do you choose a stable primary key when none exists?
16. How do you test an ETL pipeline?
17. How do you handle missing values differently for different columns?
18. What is data skew (e.g. one chemical in 81% of rows), and how does it affect analysis?
19. How do you document data cleaning decisions for auditors?
20. When would you use an LLM in a data pipeline, and when not?

### K. LLMOps: cost, latency, deployment and monitoring
1. How do you estimate and control LLM cost per request?
2. How do you reduce latency in a multi-step LLM pipeline?
3. What is caching, at the prompt, response and embedding levels?
4. How do you handle rate limits (429s) and retries?
5. What is a circuit breaker, and why did you add one?
6. How do you manage API keys and secrets?
7. How do you switch LLM providers without rewriting the app?
8. How do you monitor an LLM application in production?
9. Which traces and logs would you keep for every request?
10. How do you run models on a low-memory machine?
11. How do you deploy this system as an API?
12. How do you scale it to many concurrent users?
13. How do you roll out a new model version safely?
14. What is model drift, and how do you detect it?
15. How do you handle data privacy when sending data to an LLM API?
16. Self-hosted versus API models: cost, privacy and quality trade-offs?
17. How do you budget for evaluation runs?
18. How do you make an LLM system reproducible?
19. What would you put on a dashboard for this system?
20. What would you change before putting this system in production?
