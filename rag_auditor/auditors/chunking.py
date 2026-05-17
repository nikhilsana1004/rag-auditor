"""
Auditor #1 — Bad Chunking
Detects: oversized chunks, undersized chunks, mid-sentence splits.
Compares fixed vs sentence-aware chunking and scores the difference.
"""

from typing import List
from rich.console import Console
from rich.table import Table
from rich import box

from rag_auditor.ingest.loader import Document
from rag_auditor.utils.chunker import fixed_chunker, sentence_chunker, count_tokens

console = Console()

# Thresholds (in approximate tokens)
MIN_TOKENS = 50
MAX_TOKENS = 600
IDEAL_MIN  = 100
IDEAL_MAX  = 400


class ChunkingAuditor:
    def __init__(self, chunk_size: int = 512, overlap: int = 50, verbose: bool = False):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.verbose = verbose

    def run(self, documents: List[Document], query: str) -> dict:
        fixed  = fixed_chunker(documents, self.chunk_size, self.overlap)
        sent   = sentence_chunker(documents, self.chunk_size, self.overlap)

        issues, fixes = [], []
        details = {
            "fixed_chunks": len(fixed),
            "sentence_chunks": len(sent),
        }

        # --- Detect oversized chunks ---
        oversized = [c for c in fixed if c.token_count > MAX_TOKENS]
        if oversized:
            issues.append(
                f"{len(oversized)}/{len(fixed)} chunks exceed {MAX_TOKENS} tokens "
                f"(avg {sum(c.token_count for c in oversized)//len(oversized)} tokens). "
                "Large chunks dilute relevant content and trigger 'lost in the middle'."
            )
            fixes.append("Reduce chunk_size or switch to semantic/sentence chunking.")

        # --- Detect undersized chunks ---
        undersized = [c for c in fixed if c.token_count < MIN_TOKENS]
        if undersized:
            issues.append(
                f"{len(undersized)}/{len(fixed)} chunks are under {MIN_TOKENS} tokens. "
                "Tiny chunks lose context and hurt embedding quality."
            )
            fixes.append("Increase chunk_size or add more overlap to preserve context.")

        # --- Detect mid-sentence splits ---
        mid_splits = _count_mid_sentence_splits(fixed)
        if mid_splits > 0:
            issues.append(
                f"{mid_splits} chunks appear to start or end mid-sentence — "
                "this breaks semantic coherence."
            )
            fixes.append("Use sentence-aware chunking (already computed above as comparison).")

        # --- Score ---
        bad_ratio = (len(oversized) + len(undersized) + mid_splits) / max(len(fixed), 1)
        score = max(0.0, 1.0 - bad_ratio)
        status = "pass" if score > 0.8 else ("warn" if score > 0.5 else "fail")

        if not issues:
            issues = ["No critical chunking issues detected."]

        details.update({
            "oversized": len(oversized),
            "undersized": len(undersized),
            "mid_sentence_splits": mid_splits,
            "avg_fixed_tokens": sum(c.token_count for c in fixed) // max(len(fixed), 1),
            "avg_sent_tokens":  sum(c.token_count for c in sent)  // max(len(sent),  1),
        })

        if self.verbose:
            _print_chunk_table(fixed[:10], "Fixed chunker (first 10 chunks)")
            _print_chunk_table(sent[:10],  "Sentence chunker (first 10 chunks)")

        return {
            "status": status,
            "score": score,
            "issues": issues,
            "fixes": fixes,
            "details": details,
            "chunks": fixed,          # used downstream by other auditors
            "sent_chunks": sent,
        }


def _count_mid_sentence_splits(chunks) -> int:
    """Heuristic: chunk starts with lowercase or ends without punctuation."""
    count = 0
    for chunk in chunks:
        text = chunk.text.strip()
        if not text:
            continue
        starts_mid = text[0].islower()
        ends_mid   = text[-1] not in ".!?\"'"
        if starts_mid or ends_mid:
            count += 1
    return count


def _print_chunk_table(chunks, title: str):
    table = Table(title=title, box=box.SIMPLE, show_lines=False)
    table.add_column("#",       style="dim",  width=4)
    table.add_column("Tokens",  style="cyan", width=7)
    table.add_column("Preview", style="white")
    for c in chunks:
        preview = c.text[:80].replace("\n", " ") + ("…" if len(c.text) > 80 else "")
        color = "red" if c.token_count > MAX_TOKENS or c.token_count < MIN_TOKENS else "green"
        table.add_row(str(c.index), f"[{color}]{c.token_count}[/{color}]", preview)
    console.print(table)
