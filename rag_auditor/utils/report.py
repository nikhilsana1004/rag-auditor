"""
Report utilities: render a rich summary table to terminal,
and export full results to JSON.
"""

import json
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box


LABELS = {
    "chunking":       "#1 Bad Chunking",
    "retrieval":      "#2 Poor Retrieval",
    "lost_in_middle": "#3 Lost in the Middle",
    "rg_alignment":   "#4 Retriever-Generator Mismatch",
}

STATUS_COLOR = {"pass": "green", "warn": "yellow", "fail": "red"}
STATUS_ICON  = {"pass": "✓", "warn": "⚠", "fail": "✗"}


def render_report(results: dict, query: str, console: Console):
    table = Table(
        title="[bold]RAG Audit Summary[/bold]",
        box=box.ROUNDED,
        border_style="purple",
        show_lines=True,
    )
    table.add_column("Check",   style="bold",  width=32)
    table.add_column("Status",  width=8,  justify="center")
    table.add_column("Score",   width=8,  justify="right")
    table.add_column("Top issue")

    overall_scores = []

    for key, result in results.items():
        label   = LABELS.get(key, key)
        status  = result.get("status", "?")
        score   = result.get("score", 0.0)
        issues  = result.get("issues", [])
        color   = STATUS_COLOR.get(status, "white")
        icon    = STATUS_ICON.get(status, "?")
        top_issue = issues[0][:72] + "…" if issues and len(issues[0]) > 72 else (issues[0] if issues else "—")
        table.add_row(
            label,
            f"[{color}]{icon} {status.upper()}[/{color}]",
            f"[{color}]{score:.0%}[/{color}]",
            f"[dim]{top_issue}[/dim]",
        )
        overall_scores.append(score)

    console.print(table)

    # Overall health score
    if overall_scores:
        avg = sum(overall_scores) / len(overall_scores)
        color  = "green" if avg > 0.75 else ("yellow" if avg > 0.5 else "red")
        emoji  = "✓ Healthy" if avg > 0.75 else ("⚠ Needs attention" if avg > 0.5 else "✗ At risk")
        console.print(Panel.fit(
            f"[bold]Overall pipeline health:[/bold] [{color}]{avg:.0%} — {emoji}[/{color}]\n"
            f"[dim]Query:[/dim] {query}",
            border_style=color,
        ))


def save_report(results: dict, query: str, docs_path: str, output: Path):
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "query": query,
        "docs_path": docs_path,
        "checks": {},
    }
    for key, result in results.items():
        # Strip non-serialisable objects (chunks, embeddings)
        safe = {
            k: v for k, v in result.items()
            if k not in ("chunks", "sent_chunks", "ranked_indices", "scores")
        }
        report["checks"][key] = safe

    Path(output).write_text(json.dumps(report, indent=2))
