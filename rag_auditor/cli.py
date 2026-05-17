"""
rag-auditor: CLI tool to audit RAG pipelines for failure modes.

Basic:
  rag-audit run --docs ./docs --query "What is the refund policy?"

With LLM (real citation check):
  rag-audit run --docs ./docs --query "..." --llm anthropic --llm-key $ANTHROPIC_API_KEY
  rag-audit run --docs ./docs --query "..." --llm openai   --llm-key $OPENAI_API_KEY

With reranker (apply + show before/after):
  rag-audit run --docs ./docs --query "..." --reranker cross-encoder
  rag-audit run --docs ./docs --query "..." --reranker cohere --reranker-key $COHERE_KEY

Combined:
  rag-audit run --docs ./docs --query "..." --llm anthropic --llm-key $KEY --reranker cross-encoder
"""

import os
import typer

# Auto-load .env from the project root (or current working directory)
def _load_env():
    for candidate in [
        os.path.join(os.path.dirname(__file__), '..', '.env'),  # project root
        os.path.join(os.getcwd(), '.env'),                       # wherever you run from
    ]:
        path = os.path.normpath(candidate)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, _, val = line.partition('=')
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and val and key not in os.environ:
                    os.environ[key] = val
        break

_load_env()
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from rag_auditor.ingest.loader import load_documents
from rag_auditor.auditors.chunking import ChunkingAuditor
from rag_auditor.auditors.retrieval import RetrievalAuditor
from rag_auditor.auditors.lost_in_middle import LostInMiddleAuditor
from rag_auditor.auditors.rg_alignment import RGAlignmentAuditor
from rag_auditor.utils.report import render_report, save_report

app = typer.Typer(
    name="rag-audit",
    help="Audit any RAG pipeline for failure modes.",
    add_completion=False,
)
console = Console()

FAILURE_MODES = {
    "chunking":       ("Bad Chunking",                "#1"),
    "retrieval":      ("Poor Retrieval / Low Recall", "#2"),
    "lost_in_middle": ("Lost in the Middle",          "#3"),
    "rg_alignment":   ("Retriever-Generator Mismatch","#4"),
}


@app.command("run")
def run_audit(
    docs:          Path           = typer.Option(...,    "--docs",          "-d",  help="Folder or file to audit."),
    query:         str            = typer.Option(...,    "--query",         "-q",  help="Test query."),
    chunk_size:    int            = typer.Option(512,    "--chunk-size",           help="Tokens per chunk."),
    chunk_overlap: int            = typer.Option(50,     "--chunk-overlap",        help="Token overlap between chunks."),
    top_k:         int            = typer.Option(5,      "--top-k",                help="Chunks to retrieve."),
    checks:        str            = typer.Option("all",  "--checks",               help="Comma-separated checks or 'all'."),
    # LLM options
    llm:           Optional[str]  = typer.Option(None,   "--llm",                  help="LLM provider: anthropic | openai"),
    llm_key:       Optional[str]  = typer.Option(None,   "--llm-key",              help="API key for the LLM provider."),
    llm_model:     Optional[str]  = typer.Option(None,   "--llm-model",            help="Override default model (e.g. gpt-4o, claude-3-5-sonnet-20241022)."),
    # Reranker options
    reranker:      Optional[str]  = typer.Option(None,   "--reranker",             help="Reranker: cross-encoder | cohere"),
    reranker_key:  Optional[str]  = typer.Option(None,   "--reranker-key",         help="API key for Cohere reranker."),
    reranker_model:Optional[str]  = typer.Option(None,   "--reranker-model",       help="Override reranker model."),
    # Output
    output:        Optional[Path] = typer.Option(None,   "--output",        "-o",  help="Save JSON report here."),
    verbose:       bool           = typer.Option(False,  "--verbose",       "-v",  help="Show chunk-level detail."),
):
    """Run a full RAG audit against your documents."""

    console.print(Panel.fit(
        "[bold purple]RAG Auditor[/bold purple] — failure mode detector\n"
        f"[dim]docs:[/dim] {docs}   [dim]query:[/dim] {query}",
        border_style="purple",
    ))

    # --- Resolve checks ---
    if checks.strip().lower() == "all":
        active = list(FAILURE_MODES.keys())
    else:
        active = [c.strip() for c in checks.split(",") if c.strip() in FAILURE_MODES]
        if not active:
            console.print("[red]No valid checks. Use 'all' or comma-separate from:[/red]")
            console.print(", ".join(FAILURE_MODES.keys()))
            raise typer.Exit(1)

    # --- Load documents ---
    with console.status("[bold]Loading documents...[/bold]"):
        documents = load_documents(docs)
    console.print(f"[green]✓[/green] Loaded {len(documents)} document(s)")

    # --- Build LLM (optional) ---
    llm_instance = None
    if llm:
        if llm_key:
            key = llm_key
        elif llm == "anthropic":
            key = os.environ.get("ANTHROPIC_API_KEY")
        elif llm == "openai":
            key = os.environ.get("OPENAI_API_KEY")
        else:
            key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            console.print(f"[red]--llm-key required for provider '{llm}' (or set env var).[/red]")
            raise typer.Exit(1)
        try:
            from rag_auditor.llm import get_llm
            llm_instance = get_llm(provider=llm, api_key=key, model=llm_model)
            console.print(f"[green]✓[/green] LLM: [bold]{llm}[/bold] ({llm_instance.model})")
        except Exception as e:
            console.print(f"[red]LLM setup failed: {e}[/red]")
            raise typer.Exit(1)

    # --- Build reranker (optional) ---
    reranker_instance = None
    if reranker:
        try:
            from rag_auditor.reranker import get_reranker
            resolved_reranker_key = reranker_key or os.environ.get("COHERE_API_KEY")
            reranker_instance = get_reranker(
                provider=reranker, api_key=resolved_reranker_key, model=reranker_model
            )
            label = reranker_model or ("ms-marco-MiniLM" if reranker == "cross-encoder" else "rerank-english-v3.0")
            console.print(f"[green]✓[/green] Reranker: [bold]{reranker}[/bold] ({label})")
        except Exception as e:
            console.print(f"[red]Reranker setup failed: {e}[/red]")
            raise typer.Exit(1)

    # --- Run auditors ---
    results = {}
    for key in active:
        label, num = FAILURE_MODES[key]
        console.print(f"\n[bold]{num} Auditing: {label}[/bold]")

        if key == "chunking":
            auditor = ChunkingAuditor(chunk_size=chunk_size, overlap=chunk_overlap, verbose=verbose)
        elif key == "retrieval":
            auditor = RetrievalAuditor(top_k=top_k, verbose=verbose)
        elif key == "lost_in_middle":
            auditor = LostInMiddleAuditor(top_k=top_k, verbose=verbose, reranker=reranker_instance)
        elif key == "rg_alignment":
            auditor = RGAlignmentAuditor(top_k=top_k, verbose=verbose, llm=llm_instance)

        result = auditor.run(documents=documents, query=query)
        results[key] = result
        _print_result(result, console)

    # --- Summary ---
    console.print("\n")
    render_report(results, query, console)

    if output:
        save_report(results, query, str(docs), output)
        console.print(f"\n[green]✓[/green] Report saved to [bold]{output}[/bold]")


@app.command("list-checks")
def list_checks():
    """List all available failure mode checks."""
    table = Table(title="Available Checks", box=box.ROUNDED, border_style="purple")
    table.add_column("Flag",         style="cyan",  no_wrap=True)
    table.add_column("Failure Mode", style="bold")
    table.add_column("Description")
    rows = {
        "chunking":       "Detects chunks that are too large, too small, or split mid-sentence.",
        "retrieval":      "Measures recall@k — whether the relevant chunk is actually retrieved.",
        "lost_in_middle": "Detects position bias; optionally applies a reranker to fix it.",
        "rg_alignment":   "Measures retriever-generator alignment; optionally calls your LLM.",
    }
    for key, (label, num) in FAILURE_MODES.items():
        table.add_row(key, f"{num} {label}", rows[key])
    Console().print(table)


@app.command("list-providers")
def list_providers():
    """List supported LLM providers and rerankers."""
    console = Console()
    console.print("\n[bold]LLM providers[/bold] (--llm)")
    console.print("  anthropic   Claude (default: claude-3-5-haiku-20241022)")
    console.print("  openai      GPT (default: gpt-4o-mini)")
    console.print("\n[bold]Rerankers[/bold] (--reranker)")
    console.print("  cross-encoder   Local MS-MARCO cross-encoder (no API key, ~80MB download)")
    console.print("  cohere          Cohere Rerank API (requires --reranker-key)")


def _print_result(result: dict, console: Console):
    status = result.get("status", "unknown")
    score  = result.get("score", None)
    issues = result.get("issues", [])
    fixes  = result.get("fixes", [])

    color = {"pass": "green", "warn": "yellow", "fail": "red"}.get(status, "white")
    score_str = f"  score: [bold]{score:.0%}[/bold]" if score is not None else ""
    console.print(f"  Status: [{color}]{status.upper()}[/{color}]{score_str}")
    for issue in issues:
        console.print(f"  [yellow]⚠[/yellow]  {issue}")
    for fix in fixes:
        console.print(f"  [cyan]→[/cyan]  {fix}")

    # Extra detail for reranker before/after
    details = result.get("details", {})
    if details.get("reranker_applied") and "before" in details and "after" in details:
        b, a = details["before"], details["after"]
        console.print(
            f"  [dim]Before rerank:[/dim] position #{b['best_position']+1}, "
            f"attention {b['best_attention']:.0%}  "
            f"[dim]After:[/dim] position #{a['best_position']+1}, "
            f"attention {a['best_attention']:.0%}"
        )

    # LLM answer preview
    if details.get("mode") == "llm" and details.get("llm_answer_preview"):
        preview = details["llm_answer_preview"].replace("\n", " ")[:120]
        console.print(f"  [dim]LLM answer:[/dim] {preview}…")


if __name__ == "__main__":
    app()
