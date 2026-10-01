"""Entity & constraint extraction for the current sub-task.

Three passes, merged (deterministic first, the LLM only adds):
  1. regex     - CAS numbers, years/date ranges, status words, group-by phrases, top-N
  2. gazetteer - longest exact matches against the canonical alias table (chemicals, categories, ...)
  3. cue rules - phrases after "contains", "brand", "by/for/from <Capitalised>", quoted strings; these catch
                 misspellings that the gazetteer cannot, and are resolved fuzzily downstream
  (+ LLM)      - Gemini structured extraction, merged without duplicating deterministic mentions
"""

from __future__ import annotations

import re
from datetime import date

from chemrag.agents.context import AgentContext, Timer, event
from chemrag.etl.normalize import norm_key
from chemrag.llm.base import LLMExtraction, LLMUnavailable, prompt, wrap_user
from chemrag.schemas import DateConstraint, EntityType, Extraction, Intent, Mention, WarningItem
from chemrag.state import TurnState

CAS_HYPHEN = re.compile(r"(?<![\d-])(\d{2,7}[-\s]\d{2}[-\s]\d)(?![\d-])")
CAS_BARE = re.compile(r"(?i)\bcas\s*(?:no\.?|number|#)?\s*:?\s*(\d{5,10})\b")
YEAR = r"(19\d{2}|20\d{2}|21\d{2})"
RANGE = re.compile(rf"(?:between|from)\s+{YEAR}\s+(?:and|to|through|until|-)\s+{YEAR}|{YEAR}\s*(?:-|–|to|through)\s*{YEAR}", re.IGNORECASE)
SINCE = re.compile(rf"\b(since|after|from|starting(?: in)?)\s+{YEAR}", re.IGNORECASE)
BEFORE = re.compile(rf"\b(before|prior to|until|through|up to)\s+{YEAR}", re.IGNORECASE)
IN_YEAR = re.compile(rf"\b{YEAR}\b")
LAST_N = re.compile(r"\b(?:last|past|previous)\s+(\d{1,2})\s+years?\b", re.IGNORECASE)
TOP_N = re.compile(r"\btop\s+(\d{1,3})\b", re.IGNORECASE)

DISCONTINUED = re.compile(r"\bdiscontinu\w*", re.IGNORECASE)
NOT_DISCONTINUED = re.compile(r"\b(not|never|non)[ -]discontinu\w*|\bstill (sold|on the market|active)\b", re.IGNORECASE)
REMOVED = re.compile(r"\b(removed|remove|removal|reformulat\w*|no longer contain\w*|taken out)\b", re.IGNORECASE)
MOST_RECENT = re.compile(r"\b(most recent(ly)?|last reported|latest report\w*|recently reported)\b", re.IGNORECASE)
INITIAL = re.compile(r"\b(first reported|initially reported|initial report\w*|added|introduced|new)\b", re.IGNORECASE)

GROUP_RULES = [
    ("company", re.compile(r"\b(by|per|each|which|what|top \d+|list( all)?|all)\s+(the\s+)?compan(y|ies)\b"
                           r"|\bcompanies\b.*\b(most|top)\b", re.IGNORECASE)),
    ("brand", re.compile(r"\b(by|per|each|which|what|top \d+|list( all)?|all)\s+(the\s+)?brands?\b", re.IGNORECASE)),
    ("subcategory", re.compile(r"\b(by|per|each|which|what|top \d+)\s+(the\s+)?sub-?categor(y|ies)\b", re.IGNORECASE)),
    ("primary_category", re.compile(r"\b(by|per|each|which|what|top \d+)\s+(the\s+)?(primary\s+)?categor(y|ies)\b",
                                    re.IGNORECASE)),
]
CHEMICAL_ASK = re.compile(r"\b(what|which|list( the)?|show( the)?|top \d+)\s+(\w+\s+)?(chemicals?|ingredients?|substances?)\b"
                          r"|\bchemicals?\s+(are\s+|were\s+)?reported\b|\bby chemical\b"
                          r"|^\s*(chemicals?|ingredients?|substances?)\s+(in|of|for|reported|used|listed)\b", re.IGNORECASE)

# Words that never form an entity mention on their own and are trimmed from mention edges.
STOP = set(["a", "an", "the", "any", "all", "some", "of", "in", "on", "for", "by", "from", "with", "to", "and", "or", "that", "which", "what", "who", "whose", "how", "many", "much", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "has", "have", "had", "contain", "contains", "containing", "contained", "include", "includes", "including", "included", "list", "lists", "listed", "show", "shows", "give", "me", "tell", "about", "products", "product", "items", "item", "records", "record", "rows", "row", "chemicals", "chemical", "ingredient", "ingredients", "substance", "substances", "reported", "report", "reports", "reporting", "brand", "brands", "company", "companies", "category", "categories", "subcategory", "subcategories", "cosmetic", "cosmetics", "there", "their", "its", "it", "they", "them", "those", "these", "this", "please", "number", "count", "total", "also", "still", "ever", "were", "discontinued", "removed", "reformulated", "year", "years", "between", "since", "before", "after", "during", "until", "through", "most", "least", "top", "per", "each", "trend", "trends", "over", "time", "compare", "versus", "vs", "summarize", "summary", "overview", "cas", "data", "dataset", "sold", "made", "makes", "make", "using", "use", "used", "currently", "first", "initially", "recently", "new", "added", "introduced", "not", "no", "only", "chemicalname", "chemical_name", "casnumber", "cas_number", "brandname", "companyname", "productname", "=", "had", "having"])
# Single words that are also brand/company names but are far more often ordinary English.
COMMON_SINGLE = STOP | set(["pure", "bare", "mineral", "natural", "organic", "professional", "beauty", "classic", "basic", "fresh", "color", "colors", "nail", "nails", "hair", "skin", "lip", "lips", "eye", "eyes", "face", "body", "baby", "sun", "care", "essentials", "lipstick", "shampoo", "polish", "cream", "lotion", "spray", "oil", "gel", "powder", "foundation", "mascara", "sunscreen", "makeup", "make", "up", "kids", "men", "women", "spa", "salon", "studio"])

CUE_CHEMICAL = re.compile(
    r"\b(?P<v>contain(?:s|ing|ed)?|with|includ(?:e|es|ing|ed)|has|had|have|chemical(?:\s*name)?\s*(?:=|is|of)?|"
    r"ingredient)\s+(?P<x>[^,;?]+?)(?=\s+(?:in|for|by|from|and|or|that|which|were|was|is|are|during|between|"
    r"since|before|after|discontinued|removed|over|across|under)\b|[,;?]|\.(?:\s|$)|$)", re.IGNORECASE)
CUE_BRAND = re.compile(r"\bbrand\s+(?:name\s+)?[\"“']?(?P<x>[^,;?\"”']+?)[\"”']?(?=\s+(?:in|for|by|from|and|that|which|"
                       r"with|were|was|is|are|have|has|had)\b|[,;?]|\.(?:\s|$)|$)", re.IGNORECASE)
CUE_COMPANY = re.compile(r"\bcompany\s+(?:name\s+)?[\"“']?(?P<x>[^,;?\"”']+?)[\"”']?(?=\s+(?:in|for|and|that|which|with|"
                         r"were|was|is|are|have|has|had)\b|[,;?]|\.(?:\s|$)|$)", re.IGNORECASE)
CUE_PROPER = re.compile(r"\b(?:by|from|for|of|did|does)\s+(?P<x>[A-Z][\w'&.\-]*(?:\s+(?:[A-Z][\w'&.\-]*|&|and|of|de|du|la))*)")
QUOTED = re.compile(r"[\"“”]([^\"“”]{2,120})[\"“”]|(?<![\w])'([^']{2,120})'(?![\w])")
NEGATION = re.compile(r"\b(excluding|except|other than|apart from|without)\s+(?P<x>[^,;?]+?)(?=[,;?]|\.(?:\s|$)|$)", re.IGNORECASE)


def _trim(text: str) -> str:
    toks = re.findall(r"[\w'&./+-]+", text)
    while toks and toks[0].casefold() in STOP:
        toks.pop(0)
    while toks and toks[-1].casefold() in STOP:
        toks.pop()
    return " ".join(toks).strip(" .'-")


def _blank(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for a, b in spans:
        for i in range(a, b):
            chars[i] = " "
    return "".join(chars)


def extract_dates(text: str, intent: Intent, discontinued: bool, removed: bool,
                  ctx: AgentContext) -> tuple[list[DateConstraint], list[str], list[tuple[int, int]]]:
    """Year-level date constraints, with the field chosen from nearby words or the documented default."""
    spans: list[tuple[int, int]] = []
    found: list[tuple[int | None, int | None, str, int]] = []  # start_year, end_year, source, position
    for m in RANGE.finditer(text):
        ys = [int(y) for y in m.groups() if y]
        found.append((min(ys), max(ys), m.group(0), m.start()))
        spans.append(m.span())
    for m in SINCE.finditer(text):
        if any(a <= m.start() < b for a, b in spans):
            continue
        y = int(m.group(2))
        found.append((y + 1 if m.group(1).lower() == "after" else y, None, m.group(0), m.start()))
        spans.append(m.span())
    for m in BEFORE.finditer(text):
        if any(a <= m.start() < b for a, b in spans):
            continue
        y = int(m.group(2))
        found.append((None, y - 1 if m.group(1).lower() in ("before", "prior to") else y, m.group(0), m.start()))
        spans.append(m.span())
    for m in IN_YEAR.finditer(text):
        if any(a <= m.start() < b for a, b in spans):
            continue
        y = int(m.group(1))
        found.append((y, y, m.group(0), m.start()))
        spans.append(m.span())
    assumptions: list[str] = []
    for m in LAST_N.finditer(text):
        n = int(m.group(1))
        _, hi = ctx.engine.coverage("initial_reported")
        found.append((hi.year - n + 1, hi.year, m.group(0), m.start()))
        spans.append(m.span())
        assumptions.append(f"'{m.group(0)}' is anchored to the dataset's latest report date ({hi.isoformat()}), "
                           f"not today's date: {hi.year - n + 1}-{hi.year}.")

    out: list[DateConstraint] = []
    for start_y, end_y, src, pos in found:
        window = text[max(0, pos - 60): pos].lower()
        defaulted = False
        if re.search(r"discontinu", window) or (discontinued and not re.search(r"report|remov|reformul", window)):
            field = "discontinued_date"
        elif re.search(r"remov|reformul|no longer", window) or (removed and not re.search(r"report", window)):
            field = "chem_removed_date"
        elif MOST_RECENT.search(window):
            field = "most_recent_reported"
        elif INITIAL.search(window) or re.search(r"report", window):
            field = "initial_reported"
        else:
            field, defaulted = "initial_reported", True
        out.append(DateConstraint(
            field=field, start=date(start_y, 1, 1) if start_y else None,
            end=date(end_y, 12, 31) if end_y else None, source_text=src, defaulted_field=defaulted,
        ))
    return out, assumptions, spans


def gazetteer_mentions(text: str, ctx: AgentContext, blocked: list[tuple[int, int]],
                       negated_spans: list[tuple[int, int]] | None = None) -> tuple[list[Mention], list[tuple[int, int]]]:
    """Longest-first exact alias matches over token n-grams of the question."""
    gaz = ctx.resolver.gazetteer()
    tokens = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[\w'&./+-]+", text)]
    used = [False] * len(tokens)
    for i, (_, s, _e) in enumerate(tokens):
        if any(a <= s < b for a, b in blocked):
            used[i] = True
    mentions: list[Mention] = []
    spans: list[tuple[int, int]] = []
    for n in range(8, 0, -1):
        for i in range(len(tokens) - n + 1):
            if any(used[i:i + n]):
                continue
            surface = text[tokens[i][1]: tokens[i + n - 1][2]]
            key = norm_key(surface)
            rows = gaz.get(key)
            if not rows:
                continue
            if n == 1 and (key in COMMON_SINGLE or len(key) < 3) and not any(
                    r.entity_type in (EntityType.CHEMICAL, EntityType.CHEMICAL_FAMILY) and len(key) >= 3 for r in rows):
                continue
            if n >= 2 and all(t.casefold() in STOP for t, _, _ in tokens[i:i + n]):
                continue
            types = {r.entity_type for r in rows}
            if types <= {EntityType.CHEMICAL, EntityType.CHEMICAL_FAMILY}:
                etype = EntityType.CHEMICAL
            else:
                etype = next(iter(types)) if len(types) == 1 else EntityType.UNKNOWN
            start = tokens[i][1]
            negated = any(a <= start < b for a, b in (negated_spans or []))
            mentions.append(Mention(type=etype, text=surface.strip(" .'"), source="gazetteer", confidence=1.0,
                                    negated=negated))
            spans.append((tokens[i][1], tokens[i + n - 1][2]))
            for k in range(i, i + n):
                used[k] = True
    return mentions, spans


CATEGORY_WORDS = re.compile(
    r"\b(?:(?:nail|lip|eye|hair|face|body|baby|sun|bath|shaving|tattoo)\s+)?(?:polish(?:es)?|lipsticks?|lip ?gloss|"
    r"lip ?balm|shampoos?|conditioners?|foundations?|mascaras?|sunscreens?|lotions?|eye ?shadows?|eyeliners?|"
    r"blush(?:es)?|hair (?:dyes?|colou?rs?)|soaps?|deodorants?|toothpastes?|fragrances?|perfumes?|"
    r"bronzers?|concealers?|primers?|nail (?:products?|care))\b", re.I)

BRAND_TAIL = re.compile(r"^\s+((?:[\w'&.+-]+\s*){1,6}?)(?=\s+(?:in|for|by|from|and|or|that|which|with|were|was|is|are|"
                        r"have|has|had|contain\w*)\b|[,;?!]|\.(?:\s|$)|$)", re.I)


def brand_product_guesses(text: str, gz: list[Mention], spans: list[tuple[int, int]]
                          ) -> tuple[list[Mention], list[tuple[int, int]]]:
    """'glovers medicated shampo' -> also try the product 'glovers medicated shampo' (brand-constrained).

    Low confidence, so if no product matches it is ignored rather than blocking the brand answer.
    """
    out, out_spans = [], []
    for m, (start, end) in zip(gz, spans):
        if m.type != EntityType.BRAND:
            continue
        tail = BRAND_TAIL.match(text[end:])
        if not tail:
            continue
        words = tail.group(1).split()
        if any(w.casefold() not in COMMON_SINGLE for w in words):
            out.append(Mention(type=EntityType.PRODUCT, text=f"{m.text} {' '.join(words)}", source="cue",
                               confidence=0.6))
            out_spans.append((start, end + tail.end(1)))
    return out, out_spans


def cue_mentions(text: str, original: str, existing: list[Mention]) -> list[Mention]:
    have = {norm_key(m.text) for m in existing}
    out: list[Mention] = []

    def add(etype: EntityType, raw: str, conf: float, negated: bool = False) -> None:
        x = _trim(raw)
        k = norm_key(x)
        if not k or len(k) < 3 or k in have or all(t in STOP for t in k.split()):
            return
        if any(k in h or h in k for h in have if h):
            return
        have.add(k)
        out.append(Mention(type=etype, text=x, source="cue", confidence=conf, negated=negated))

    for m in NEGATION.finditer(text):
        add(EntityType.CHEMICAL, m.group("x"), 0.8, negated=True)
    for m in CUE_BRAND.finditer(text):
        add(EntityType.BRAND, m.group("x"), 0.85)
    for m in CUE_COMPANY.finditer(text):
        add(EntityType.COMPANY, m.group("x"), 0.85)
    for m in CUE_CHEMICAL.finditer(text):
        # "contains X" / "chemical X" clearly names a chemical; "has/with X" is weaker (non-blocking if unmatched)
        strong = m.group("v").lower().startswith(("contain", "chemical", "ingredient"))
        add(EntityType.CHEMICAL, m.group("x"), 0.75 if strong else 0.6)
    for m in CUE_PROPER.finditer(text):
        add(EntityType.UNKNOWN, m.group("x"), 0.6)
    return out


def rules_extract(text: str, intent: Intent, subtask_id: str, ctx: AgentContext) -> tuple[Extraction, list[str]]:
    mentions: list[Mention] = []
    spans: list[tuple[int, int]] = []
    for m in CAS_HYPHEN.finditer(text):
        mentions.append(Mention(type=EntityType.CAS, text=m.group(1), source="regex"))
        spans.append(m.span())
    for m in CAS_BARE.finditer(text):
        if not any(a <= m.start(1) < b for a, b in spans):
            mentions.append(Mention(type=EntityType.CAS, text=m.group(1), source="regex"))
            spans.append(m.span())
    discontinued = None
    if NOT_DISCONTINUED.search(text):
        discontinued = False
    elif DISCONTINUED.search(text):
        discontinued = True
    removed = True if REMOVED.search(text) else None
    # Quoted strings first: a quoted name is one mention (a product unless it is exactly a known entity or
    # is introduced by "brand"/"company"), and its words must not be re-matched by the gazetteer.
    gaz = ctx.resolver.gazetteer()
    for m in QUOTED.finditer(text):
        inner = (m.group(1) or m.group(2)).strip()
        before = text[max(0, m.start() - 12): m.start()].lower()
        if "brand" in before:
            etype = EntityType.BRAND
        elif "company" in before:
            etype = EntityType.COMPANY
        elif norm_key(inner) in gaz:
            etype = EntityType.UNKNOWN
        else:
            etype = EntityType.PRODUCT
        mentions.append(Mention(type=etype, text=inner, source="regex", confidence=0.95))
        spans.append(m.span())
    neg_spans = [(m.start(), m.end()) for m in NEGATION.finditer(text)]
    dates, assumptions, dspans = extract_dates(_blank(text, spans), intent, bool(discontinued), bool(removed), ctx)
    spans += dspans
    gz, gspans = gazetteer_mentions(text, ctx, spans, neg_spans)
    mentions += gz
    spans += gspans
    guesses, guess_spans = brand_product_guesses(text, gz, gspans)
    mentions += guesses
    spans += guess_spans
    if not any(m.type in (EntityType.SUBCATEGORY, EntityType.PRIMARY_CATEGORY) for m in mentions):
        blanked = _blank(text, spans)
        for m in CATEGORY_WORDS.finditer(blanked):
            mentions.append(Mention(type=EntityType.SUBCATEGORY, text=m.group(0).strip(), source="cue",
                                    confidence=0.65))
            spans.append(m.span())
            break
    # Quoted product names and cue phrases run on the text with matched spans blanked out.
    mentions += cue_mentions(_blank(text, spans), text, mentions)
    group_by = [g for g, pat in GROUP_RULES if pat.search(text)]
    asks_chem = bool(CHEMICAL_ASK.search(text))
    trend_field = None
    if intent == Intent.TREND:
        if DISCONTINUED.search(text):
            trend_field = "discontinued_date"
        elif REMOVED.search(text):
            trend_field = "chem_removed_date"
        elif MOST_RECENT.search(text):
            trend_field = "most_recent_reported"
        else:
            trend_field = "initial_reported"
        # years in a trend question bound the series on the trend field
        dates = [d.model_copy(update={"field": trend_field, "defaulted_field": False}) for d in dates]
    top = TOP_N.search(text)
    ex = Extraction(subtask_id=subtask_id, mentions=mentions, dates=dates, discontinued=discontinued,
                    chem_removed=removed, group_by=group_by, asks_for_chemicals=asks_chem, trend_field=trend_field,
                    limit=int(top.group(1)) if top else None, mode="rules")
    return ex, assumptions


def merge_llm(ex: Extraction, text: str, ctx: AgentContext) -> Extraction:
    out: LLMExtraction = ctx.llm.structured(prompt("extractor"), wrap_user(text), LLMExtraction)
    have = [norm_key(m.text) for m in ex.mentions]
    added = []
    for m in out.mentions:
        k = norm_key(m.text)
        if not k or k in have or any(k in h or h in k for h in have):
            continue
        if norm_key(m.text) not in norm_key(text):  # LLM must quote the user, not invent
            continue
        added.append(Mention(type=EntityType(m.type), text=m.text, source="llm", confidence=0.75))
        have.append(k)
    upd: dict = {"mentions": ex.mentions + added, "mode": "llm+rules"}
    if ex.discontinued is None and out.discontinued != "unspecified":
        upd["discontinued"] = out.discontinued == "yes"
    if ex.chem_removed is None and out.chemical_removed == "yes":
        upd["chem_removed"] = True
    if ex.dates and out.date_field != "none":
        upd["dates"] = [d.model_copy(update={"field": out.date_field, "defaulted_field": False})
                        if d.defaulted_field else d for d in ex.dates]
    return ex.model_copy(update=upd)


def extractor_node(state: TurnState, ctx: AgentContext) -> dict:
    t = Timer()
    sub = state.current_subtask
    assert sub is not None
    warnings: list[WarningItem] = []
    assumptions: list[str] = []
    if sub.intent in (Intent.COVERAGE, Intent.DATA_QUALITY, Intent.OUT_OF_SCOPE):
        ex = Extraction(subtask_id=sub.id, mode="rules")
    else:
        ex, assumptions = rules_extract(sub.text, sub.intent, sub.id, ctx)
        if ctx.llm.available:
            try:
                ex = merge_llm(ex, sub.text, ctx)
            except LLMUnavailable as e:
                warnings.append(WarningItem(code="llm_unavailable", severity="info",
                                            message=f"Extractor used rules only: {e}"))
        # Follow-up sub-questions ("..., and how many were discontinued?") inherit the previous entities.
        if state.current > 0 and not [m for m in ex.mentions if not m.negated] and not ex.dates:
            prev = state.extraction_for(state.plan.subtasks[state.current - 1].id)
            if prev and (prev.mentions or prev.dates):
                ex = ex.model_copy(update={
                    "mentions": [m.model_copy(update={"source": "inherited"}) for m in prev.mentions],
                    "dates": ex.dates or prev.dates,
                    "discontinued": ex.discontinued if ex.discontinued is not None else prev.discontinued,
                    "chem_removed": ex.chem_removed if ex.chem_removed is not None else prev.chem_removed,
                })
                carried = [m.text for m in prev.mentions] + [d.source_text for d in prev.dates]
                assumptions.append(f"Sub-question '{sub.text}' reuses the constraints of the previous sub-question: "
                                   + ", ".join(carried) + ".")
        for d in ex.dates:
            if d.defaulted_field:
                assumptions.append(f"'{d.source_text}' interpreted as InitialDateReported (year first reported); "
                                   "say 'most recently reported' to use MostRecentDateReported instead.")
    mode = "llm+rules" if ex.mode == "llm+rules" else "rules"
    summary = (f"{sub.id}: " + (", ".join(f"{m.type.value}='{m.text}'" for m in ex.mentions) or "no entities")
               + (f"; dates={[(d.field, str(d.start), str(d.end)) for d in ex.dates]}" if ex.dates else "")
               + (f"; discontinued={ex.discontinued}" if ex.discontinued is not None else "")
               + (f"; removed={ex.chem_removed}" if ex.chem_removed else "")
               + (f"; group_by={ex.group_by}" if ex.group_by else ""))
    return {"extractions": [ex], "assumptions": assumptions, "warnings": warnings,
            "trace": [event("extractor", summary, t, mode=mode, extraction=ex.model_dump(mode="json"))]}
