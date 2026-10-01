"""`chemrag` command line: build, ask, replay, eval, doctor, graph."""

from __future__ import annotations

import json
import logging
import os
import platform
import resource
import sys
import time

import typer
from rich.console import Console
from rich.prompt import IntPrompt
from rich.table import Table

from chemrag.settings import ROOT, get_settings
from chemrag.state import RunOptions

app = typer.Typer(add_completion=False, help="Multi-agent orchestrator for the CSCP chemical disclosure dataset.")
console = Console()


def _peak_mb() -> float:
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1024 if platform.system() != "Darwin" else kb / 1024 / 1024


@app.command()
def build(fast: bool = typer.Option(False, help="Skip embedding product names (faster, less RAM)."),
          skip_index: bool = typer.Option(False, help="Only build DuckDB; no embeddings / Qdrant."),
          force: bool = typer.Option(False, help="Rebuild everything even if unchanged.")) -> None:
    """ETL the CSV into DuckDB, then embed entity/product names into embedded Qdrant."""
    import subprocess

    s = get_settings()
    t0 = time.time()
    from chemrag.etl.build_db import file_sha256

    sha = file_sha256(s.csv_path)
    rebuild_db = force or not s.db_path.exists()
    if not rebuild_db:
        import duckdb

        con = duckdb.connect(str(s.db_path), read_only=True)
        try:
            old = json.loads(con.execute("SELECT value FROM dataset_meta WHERE key='csv_sha256'").fetchone()[0])
        except Exception:
            old = None
        con.close()
        rebuild_db = old != sha
    if rebuild_db:
        console.print("[bold]1/2 ETL[/] CSV → DuckDB (separate process so its memory is released)")
        code = subprocess.call([sys.executable, "-m", "chemrag.cli", "etl"], cwd=ROOT)
        if code:
            raise typer.Exit(code)
    else:
        console.print("[bold]1/2 ETL[/] DuckDB is up to date with the CSV (sha256 matches)")
    if skip_index:
        console.print("[yellow]Skipping vector index (--skip-index); resolution will be lexical-only.[/]")
        return
    console.print(f"[bold]2/2 Index[/] {s.embed_model} → embedded Qdrant at {s.qdrant_path}")
    from chemrag.retrieval.build_index import build_vector_index
    from chemrag.retrieval.embed import EmbeddingUnavailable

    try:
        m = build_vector_index(s.db_path, s.qdrant_path, s.manifest_path, s.embed_model, sha, fast=fast,
                               batch=s.embed_batch, threads=s.torch_threads, force=force, log=console.print)
        console.print(f"index ready: {m['collections']} (dim {m['dim']}, semantic floor {m['semantic_floor']:.3f})")
    except EmbeddingUnavailable as e:
        console.print(f"[yellow]Embedding model unavailable — continuing lexical-only.[/] {e}\n"
                      "Fix network access to huggingface.co, or set CHEMRAG_EMBED_MODEL=thenlper/gte-base "
                      "(smaller) or hash-ngram (offline, non-semantic).")
    console.print(f"done in {time.time() - t0:.0f}s · peak RSS {_peak_mb():.0f} MB")


@app.command(hidden=True)
def etl() -> None:
    from chemrag.etl.build_db import build_database

    s = get_settings()
    build_database(s.csv_path, s.db_path, s.groups_path, log=console.print)
    console.print(f"ETL peak RSS {_peak_mb():.0f} MB")


@app.command()
def ask(question: str = typer.Argument(..., help="Natural-language question."),
        as_json: bool = typer.Option(False, "--json", help="Print the raw JSON response contract."),
        no_llm: bool = typer.Option(False, "--no-llm", help="Deterministic rules only (no Gemini calls)."),
        no_vectors: bool = typer.Option(False, "--no-vectors", help="Lexical-only entity resolution."),
        assume_best: bool = typer.Option(False, "--assume-best", help="On ambiguity, use the best match + warning."),
        interactive: bool = typer.Option(True, "--interactive/--no-interactive", help="Ask on ambiguity."),
        limit: int = typer.Option(20, help="Max listed items."),
        page: int = typer.Option(1, help="Page of listed items (1-based)."),
        evidence: int = typer.Option(10, help="Evidence rows to display."),
        verbose: bool = typer.Option(False, "-v", "--verbose")) -> None:
    """Ask a question; prints answer, evidence, query plan, assumptions and warnings."""
    logging.basicConfig(level=logging.WARNING)
    from chemrag.orchestrator import Orchestrator
    from chemrag.render.cli_render import render

    orch = Orchestrator(no_llm=no_llm, use_vectors=not no_vectors)
    opts = RunOptions(assume_best=assume_best or as_json, limit=limit, offset=(page - 1) * limit)

    def on_clarify(payload: dict) -> int | None:
        console.print(f"[bold yellow]?[/] {payload['question']}")
        for i, o in enumerate(payload["options"], 1):
            console.print(f"  {i}. {o['name']} [dim]({o['type']}; {o['detail']})[/]")
        choice = IntPrompt.ask("Choose a number (0 to skip)", default=0)
        return choice - 1 if 1 <= choice <= len(payload["options"]) else None

    use_interactive = interactive and not as_json and sys.stdin.isatty()
    resp = orch.ask(question, opts, on_clarify=on_clarify if use_interactive else None)
    if as_json:
        print(resp.model_dump_json(indent=2))
    else:
        render(resp, console, max_evidence=evidence, verbose=verbose)


@app.command()
def replay(request_id: str) -> None:
    """Re-execute the logged SQL of a previous answer (no LLM) and check results are identical."""
    from chemrag.orchestrator import Orchestrator

    rows = Orchestrator(no_llm=True, use_vectors=False).replay(request_id)
    t = Table("call", "tool", "rows", "identical")
    for r in rows:
        t.add_row(r["id"], r["tool"], str(r["rows"]), "[green]yes[/]" if r["identical"] else "[red]NO[/]")
    console.print(t)
    if not all(r["identical"] for r in rows):
        raise typer.Exit(1)


@app.command(name="eval")
def run_eval(no_llm: bool = typer.Option(False, "--no-llm"), only: str = typer.Option("", help="Comma-separated ids"),
             holdout: bool = typer.Option(False, "--holdout", help="Run the held-out paraphrase set."),
             out: str = typer.Option("evals/results", help="Output directory.")) -> None:
    """Run the golden evaluation set and write a markdown + JSON report."""
    sys.path.insert(0, str(ROOT))
    from evals.run_eval import main

    raise typer.Exit(main(no_llm=no_llm, only=[x for x in only.split(",") if x], out_dir=ROOT / out,
                          set_name="holdout" if holdout else "golden"))


@app.command()
def doctor() -> None:
    """Check environment: RAM, DB, index, embedding model, Gemini key and model id."""
    s = get_settings()
    t = Table("check", "status", "detail")
    try:
        import psutil  # type: ignore

        ram = f"{psutil.virtual_memory().total / 2**30:.1f} GB total"
    except Exception:
        ram = os.popen("free -h 2>/dev/null | awk '/Mem:/{print $2}'").read().strip() or "unknown"
    t.add_row("RAM", "info", ram)
    t.add_row("DuckDB", "ok" if s.db_path.exists() else "missing", str(s.db_path))
    if s.manifest_path.exists():
        m = json.loads(s.manifest_path.read_text())
        t.add_row("Qdrant index", "ok" if m.get("model") == s.embed_model else "stale",
                  f"{m.get('model')} {m.get('collections')} products_complete={m.get('products_complete')}")
    else:
        t.add_row("Qdrant index", "missing", "run `chemrag build`")
    from chemrag.retrieval.embed import EmbeddingUnavailable, get_embedder

    try:
        e = get_embedder(s.embed_model, s.torch_threads, s.embed_batch)
        t.add_row("Embedding model", "ok", f"{s.embed_model} dim={e.dim}")
    except EmbeddingUnavailable as ex:
        t.add_row("Embedding model", "unavailable", str(ex)[:160])
    if not s.gemini_api_key:
        t.add_row("Gemini", "off", "GEMINI_API_KEY not set → deterministic rules mode")
    else:
        from chemrag.llm.gemini_client import GeminiClient

        try:
            models = GeminiClient(s.gemini_api_key, s.llm_model, 15).list_models()
            ok = any(m.endswith("/" + s.llm_model) or m == s.llm_model for m in models)
            flash = [m for m in models if "flash" in m][:8]
            t.add_row("Gemini", "ok" if ok else "model not found",
                      f"configured {s.llm_model}; available flash models: {', '.join(flash)}")
        except Exception as ex:
            t.add_row("Gemini", "error", str(ex)[:160])
    t.add_row("Peak RSS", "info", f"{_peak_mb():.0f} MB")
    console.print(t)


@app.command()
def graph(out: str = typer.Option("docs/graph.md")) -> None:
    """Export the compiled LangGraph as Mermaid."""
    from chemrag.orchestrator import Orchestrator

    mermaid = Orchestrator(no_llm=True, use_vectors=False).app.get_graph().draw_mermaid()
    path = ROOT / out
    path.write_text("# Orchestrator graph (generated by `chemrag graph`)\n\n```mermaid\n" + mermaid + "```\n")
    console.print(f"wrote {path}")


if __name__ == "__main__":
    app()
