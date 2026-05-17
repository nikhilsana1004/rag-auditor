"""
Auditor #2 — Poor Retrieval / Low Recall
Measures: where the most relevant chunk ranks in your top-k results.
Flags if the best chunk is outside top-k (retrieval failure).
"""

from typing import List
from rich.console import Console
from rich.table import Table
from rich import box

from rag_auditor.ingest.loader import Document
from rag_auditor.utils.chunker import fixed_chunker
from rag_auditor.utils.embeddings import rank_chunks_by_query

console = Console()


class RetrievalAuditor:
    def __init__(self, top_k: int = 5, verbose: bool = False):
        self.top_k = top_k
        self.verbose = verbose

    def run(self, documents: List[Document], query: str) -> dict:
        chunks = fixed_chunker(documents, chunk_size=512, overlap=50)
        if not chunks:
            return {"status": "fail", "score": 0.0, "issues": ["No chunks produced."], "fixes": [], "details": {}}

        chunk_texts = [c.text for c in chunks]

        with console.status("  Computing embeddings..."):
            ranked_indices, scores = rank_chunks_by_query(query, chunk_texts)

        top_k_indices = set(ranked_indices[:self.top_k])
        best_idx      = ranked_indices[0]
        best_score    = scores[best_idx]
        worst_top_score = scores[ranked_indices[self.top_k - 1]] if len(ranked_indices) >= self.top_k else 0.0

        issues, fixes = [], []

        # --- Low top-1 similarity ---
        if best_score < 0.3:
            issues.append(
                f"Top-1 chunk similarity is only {best_score:.2%} — very low. "
                "The retriever may not find anything relevant to your query."
            )
            fixes.append("Try HyDE: embed a hypothetical answer instead of the raw query.")
            fixes.append("Consider fine-tuning the embedding model on domain data.")

        # --- Score gap between top-1 and top-k ---
        gap = best_score - worst_top_score
        if gap > 0.4:
            issues.append(
                f"Large score gap between rank #1 ({best_score:.2%}) and rank #{self.top_k} "
                f"({worst_top_score:.2%}). Low-ranked chunks in your top-k are noise."
            )
            fixes.append(f"Reduce top_k from {self.top_k} to 3, or add a reranker to filter noise.")

        # --- Too few chunks to evaluate ---
        if len(chunks) < self.top_k:
            issues.append(
                f"Only {len(chunks)} chunks exist — less than top_k={self.top_k}. "
                "Retrieval cannot be meaningfully evaluated."
            )
            fixes.append("Upload more documents or reduce chunk_size to produce more chunks.")

        score  = min(best_score * 1.2, 1.0)
        status = "pass" if score > 0.6 else ("warn" if score > 0.35 else "fail")

        if not issues:
            issues = [f"Top-1 similarity: {best_score:.2%} — retrieval looks healthy."]

        details = {
            "total_chunks": len(chunks),
            "top_k": self.top_k,
            "best_similarity": round(best_score, 4),
            "worst_top_k_similarity": round(worst_top_score, 4),
            "ranked_scores": [round(scores[i], 4) for i in ranked_indices[:10]],
        }

        if self.verbose:
            _print_rank_table(chunks, ranked_indices, scores, self.top_k)

        return {
            "status": status,
            "score": score,
            "issues": issues,
            "fixes": fixes,
            "details": details,
            "ranked_indices": ranked_indices,
            "scores": scores,
            "chunks": chunks,
        }


def _print_rank_table(chunks, ranked_indices, scores, top_k):
    table = Table(title="Retrieval ranking (top 10)", box=box.SIMPLE)
    table.add_column("Rank",  style="dim",  width=5)
    table.add_column("Score", style="cyan", width=7)
    table.add_column("Preview")
    for rank, idx in enumerate(ranked_indices[:10], 1):
        color  = "green" if rank <= top_k else "dim"
        prefix = "★ " if rank == 1 else "  "
        preview = chunks[idx].text[:80].replace("\n", " ") + "…"
        table.add_row(
            f"[{color}]#{rank}[/{color}]",
            f"[{color}]{scores[idx]:.2%}[/{color}]",
            f"[{color}]{prefix}{preview}[/{color}]",
        )
    console.print(table)
