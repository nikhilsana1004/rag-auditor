"""
Auditor #3 — Lost in the Middle
Detects position bias in the context window (Liu et al., 2023).

With --reranker flag: actually applies the reranker, shows before/after
chunk positions and attention weights so you can see the concrete improvement.

Without --reranker: uses hardcoded attention weight approximations.
"""

from typing import List
from rag_auditor.ingest.loader import Document
from rag_auditor.utils.chunker import fixed_chunker
from rag_auditor.utils.embeddings import rank_chunks_by_query


def _position_attention(k: int) -> List[float]:
    weights = []
    for i in range(k):
        if i == 0:
            weights.append(1.0)
        elif i == k - 1:
            weights.append(0.85)
        else:
            dist = min(i, k - 1 - i)
            weights.append(max(0.2, 0.85 - dist * 0.15))
    return weights


class LostInMiddleAuditor:
    def __init__(self, top_k: int = 5, verbose: bool = False, reranker=None):
        self.top_k = top_k
        self.verbose = verbose
        self.reranker = reranker

    def run(self, documents: List[Document], query: str) -> dict:
        chunks = fixed_chunker(documents, chunk_size=512, overlap=50)
        if not chunks:
            return {"status": "fail", "score": 0.0,
                    "issues": ["No chunks."], "fixes": [], "details": {}}

        chunk_texts = [c.text for c in chunks]
        retriever_ranked, scores = rank_chunks_by_query(query, chunk_texts)
        k = min(self.top_k, len(chunks))

        before = self._measure_position(retriever_ranked[:k], scores, k)
        details = {"top_k": k, "before": before, "reranker_applied": False}
        issues, fixes = [], []

        if self.reranker:
            from rich.console import Console
            with Console().status("  Running reranker..."):
                top_k_texts = [chunk_texts[i] for i in retriever_ranked[:k]]
                reranked = self.reranker.rerank(query, top_k_texts, top_k=k)

            reranked_global = [retriever_ranked[r.index] for r in reranked]
            after = self._measure_position(reranked_global, scores, k)

            details["reranker_applied"] = True
            details["after"] = after
            details["reranked_order"] = reranked_global
            details["reranker_scores"] = [r.score for r in reranked]

            improvement = after["best_attention"] - before["best_attention"]
            if improvement > 0.05:
                issues.append(
                    f"Reranker improved best-chunk attention from "
                    f"{before['best_attention']:.0%} to {after['best_attention']:.0%} "
                    f"(+{improvement:.0%}). Best chunk moved from position "
                    f"#{before['best_position']+1} to #{after['best_position']+1}."
                )
            else:
                issues.append(
                    f"Reranker applied. Attention: "
                    f"{before['best_attention']:.0%} to {after['best_attention']:.0%}. "
                    "Already well-positioned."
                )

            score = after["best_attention"]
            status = "pass" if score > 0.75 else ("warn" if score > 0.45 else "fail")

        else:
            pos = before["best_position"]
            attn = before["best_attention"]

            if pos == 0:
                issues.append(f"Best chunk at position #1 — attention: {attn:.0%}. Optimal.")
            elif pos == k - 1:
                issues.append(f"Best chunk at position #{pos+1} (end) — attention: {attn:.0%}. Acceptable.")
            else:
                issues.append(
                    f"Best chunk at position #{pos+1} of {k} (middle). "
                    f"Estimated attention: {attn:.0%} vs {_position_attention(k)[0]:.0%} at top. "
                    "LLMs systematically under-attend to middle positions."
                )
                fixes.append("Pass --reranker cross-encoder or --reranker cohere to fix this automatically.")
                fixes.append("Or use position-aware prompting: instruct the LLM to read all chunks first.")

            if k > 7:
                issues.append(f"Context has {k} chunks — large context increases middle-position risk.")
                fixes.append("Reduce top_k to 4-5 and use a reranker.")

            score = attn
            status = "pass" if score > 0.75 else ("warn" if score > 0.45 else "fail")

        return {"status": status, "score": score,
                "issues": issues, "fixes": fixes, "details": details}

    def _measure_position(self, ordered_indices, scores, k):
        attention_weights = _position_attention(k)
        best_global = max(ordered_indices, key=lambda i: scores[i])
        best_position = ordered_indices.index(best_global)
        best_attention = attention_weights[best_position]
        return {
            "best_chunk_global_idx": best_global,
            "best_position": best_position,
            "best_similarity": round(scores[best_global], 4),
            "best_attention": round(best_attention, 4),
            "attention_profile": [round(w, 2) for w in attention_weights],
        }
