"""Rich terminal rendering of the Response contract."""

from __future__ import annotations

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from chemrag.schemas import Response

LEVEL_STYLE = {"high": "green", "medium": "yellow", "low": "red"}
TYPE_STYLE = {"answer": "cyan", "clarification": "yellow", "refusal": "magenta", "no_data": "yellow"}


def render(resp: Response, console: Console | None = None, show_sql: bool = True, max_evidence: int = 10,
           verbose: bool = False) -> None:
    c = console or Console()
    conf = resp.confidence
    title = (f"[bold {TYPE_STYLE[resp.response_type]}]{resp.response_type.upper()}[/] · confidence "
             f"[{LEVEL_STYLE[conf.level]}]{conf.level.upper()} ({conf.score:.2f})[/]")
    c.print(Panel(Text(resp.answer_short, style="bold"), title=title, title_align="left", border_style="cyan"))
    if resp.answer_details.strip():
        c.print(Panel(Markdown(resp.answer_details), title="Details", title_align="left", border_style="dim"))

    if resp.evidence:
        t = Table(title=f"Evidence (showing {min(max_evidence, len(resp.evidence))} of {len(resp.evidence)} cited rows)",
                  title_justify="left", show_lines=False, header_style="bold")
        for col in ["ref", "row_id", "CDPHId", "CSFId", "ChemicalId", "Product", "Company / Brand", "Chemical (CAS)",
                    "Dates", "supports"]:
            t.add_column(col, overflow="fold")
        for e in resp.evidence[:max_evidence]:
            f = e.fields
            dates = [f"init {f.get('InitialDateReported')}"]
            if f.get("DiscontinuedDate"):
                dates.append(f"disc {f['DiscontinuedDate']}")
            if f.get("ChemicalDateRemoved"):
                dates.append(f"removed {f['ChemicalDateRemoved']}")
            t.add_row(e.ref, str(e.row_id), str(e.cdph_id), str(e.csf_id or "–"), str(e.chemical_id),
                      (f.get("ProductName") or "")[:48], f"{f.get('CompanyName')} / {f.get('BrandName') or '–'}"[:40],
                      f"{f.get('ChemicalName')} ({f.get('CasNumber') or 'no CAS'})"[:44], ", ".join(dates),
                      ",".join(e.supports[:3]))
        c.print(t)

    plan_lines = []
    for ev in resp.query_plan:
        if ev.agent == "query" and ev.mode == "sql":
            if not show_sql:
                continue
            o = ev.output
            plan_lines.append(Text.assemble((f"  {ev.step:>2} ", "dim"), (f"{o['id']} {o['tool']}", "bold blue"),
                                            f" → {o['row_count']} rows  ({ev.elapsed_ms} ms)"))
            plan_lines.append(Text(f"       SQL: {o['sql'][:400]}{'…' if len(o['sql']) > 400 else ''}", style="dim"))
            plan_lines.append(Text(f"       params: {o['bound_params']}", style="dim"))
        else:
            plan_lines.append(Text.assemble((f"  {ev.step:>2} ", "dim"), (f"{ev.agent:<11}", "bold"),
                                            (f"[{ev.mode}] ", "magenta"), ev.summary[:300]))
    c.print(Panel(Group(*plan_lines), title="Query plan", title_align="left", border_style="dim"))

    if resp.assumptions:
        c.print(Panel("\n".join(f"• {a}" for a in resp.assumptions), title="Assumptions", title_align="left",
                      border_style="blue"))
    if resp.warnings:
        sev = {"info": "dim", "warn": "yellow", "error": "red"}
        lines = [Text.assemble((f"[{w.code}] ", sev[w.severity]), w.message) for w in resp.warnings]
        c.print(Panel(Group(*lines), title="Warnings", title_align="left", border_style="yellow"))
    if verbose:
        c.print(Text(f"confidence rationale: {conf.rationale}", style="dim"))
    c.print(Text(f"request_id={resp.request_id} · llm={resp.meta.get('llm')} · vectors={resp.meta.get('vectors')} · "
                 f"{resp.meta.get('elapsed_ms')} ms · replay: chemrag replay {resp.request_id}", style="dim"))
