"""
Auditor #4 — Retriever-Generator Misalignment
Based on: RAG-E paper (Jan 2026).

With --llm flag: makes a real LLM call with the top-k chunks and checks
which chunk indices the model actually cites in its answer vs what the
retriever ranked #1. Measures true retriever-generator alignment.

Without --llm flag: falls back to the HyDE proxy approximation.
"""

from typing import List, Optional
from rag_auditor.ingest.loader import Document
from rag_auditor.utils.chunker import fixed_chunker
from rag_auditor.utils.embeddings import rank_chunks_by_query


class RGAlignmentAuditor:
    def __init__(self, top_k: int = 5, verbose: bool = False, llm=None):
        self.top_k = top_k
        self.verbose = verbose
        self.llm = llm

    def run(self, documents: List[Document], query: str) -> dict:
        chunks = fixed_chunker(documents, chunk_size=512, overlap=50)
        if not chunks:
            return {"status": "fail", "score": 0.0,
                    "issues": ["No chunks."], "fixes": [], "details": {}}

        chunk_texts = [c.text for c in chunks]
        retriever_ranked, retriever_scores = rank_chunks_by_query(query, chunk_texts)
        k = min(self.top_k, len(chunks))
        top_k_texts = [chunk_texts[i] for i in retriever_ranked[:k]]

        if self.llm:
            return self._run_with_llm(
                query, chunk_texts, retriever_ranked, retriever_scores, k, top_k_texts
            )
        else:
            return self._run_hyde_proxy(
                query, chunk_texts, retriever_ranked, retriever_scores, k
            )

    def _run_with_llm(self, query, chunk_texts, retriever_ranked,
                      retriever_scores, k, top_k_texts):
        from rich.console import Console
        console = Console()

        with console.status("  Calling LLM for citation check..."):
            response = self.llm.answer_with_chunks(query, top_k_texts)

        cited = response.cited_indices
        retriever_top1_local = 0

        issues, fixes = [], []
        details = {
            "mode": "llm",
            "llm_model": response.model,
            "top_k": k,
            "retriever_top1_chunk": retriever_ranked[0],
            "llm_cited_local_indices": cited,
            "llm_answer_preview": response.answer[:200],
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "llm_used_retriever_top1": retriever_top1_local in cited,
        }

        if not cited:
            issues.append(
                "LLM did not cite any chunk — it may have hallucinated or the "
                "context was not relevant enough to use."
            )
            fixes.append("Check if the query is answerable from your documents at all.")
            fixes.append("Try lowering chunk_size so the relevant sentence is isolated.")
            score, status = 0.1, "fail"

        elif retriever_top1_local not in cited:
            issues.append(
                f"LLM cited chunks {[i+1 for i in cited]} but ignored the retriever's "
                f"top-ranked chunk — classic retriever-generator mismatch."
            )
            fixes.append("Add a reranker to re-order chunks before sending to the LLM.")
            fixes.append("Use HyDE: embed a hypothetical answer to better align retrieval.")
            score, status = 0.25, "fail"

        elif cited and cited[0] != retriever_top1_local:
            issues.append(
                f"LLM used the top chunk but preferred chunk #{cited[0]+1} — mild ordering mismatch."
            )
            fixes.append("A reranker would promote the LLM-preferred chunk to position #1.")
            score, status = 0.65, "warn"

        else:
            issues.append(
                f"LLM cited chunk(s) {[i+1 for i in cited]} and the retriever's top chunk was used — strong alignment."
            )
            score, status = 1.0, "pass"

        return {"status": status, "score": score,
                "issues": issues, "fixes": fixes, "details": details}

    def _run_hyde_proxy(self, query, chunk_texts, retriever_ranked,
                        retriever_scores, k):
        hyde_query = f"The answer to '{query}' is:"
        generator_ranked, _ = rank_chunks_by_query(hyde_query, chunk_texts)

        retriever_top_k = set(retriever_ranked[:k])
        generator_top_k = set(generator_ranked[:k])
        overlap = retriever_top_k & generator_top_k
        overlap_pct = len(overlap) / k
        mismatch_pct = 1.0 - overlap_pct

        common = list(overlap)
        ret_ranks = {idx: r for r, idx in enumerate(retriever_ranked[:k])}
        gen_ranks = {idx: r for r, idx in enumerate(generator_ranked[:k])}
        rank_corr = _rank_correlation(
            [ret_ranks[i] for i in common],
            [gen_ranks[i] for i in common],
        ) if len(common) > 1 else 0.0

        issues, fixes = [], []
        if mismatch_pct >= 0.6:
            issues.append(
                f"High retriever-generator mismatch: {mismatch_pct:.0%} of top-{k} chunks differ (HyDE proxy). "
                "Pass --llm anthropic --llm-key $KEY for a real citation check."
            )
            fixes.append("Use HyDE: generate a hypothetical answer, embed that, then retrieve.")
            fixes.append("Add a cross-encoder reranker to re-score chunks after retrieval.")
        elif mismatch_pct >= 0.3:
            issues.append(f"Moderate mismatch: {mismatch_pct:.0%} of top-{k} chunks differ (HyDE proxy).")
            fixes.append("Add a reranker to re-score retrieved chunks.")
        else:
            issues.append(f"Low mismatch ({mismatch_pct:.0%}): retriever and generator agree on {len(overlap)}/{k} top chunks.")

        score = overlap_pct * 0.7 + max(rank_corr, 0) * 0.3
        status = "pass" if score > 0.65 else ("warn" if score > 0.4 else "fail")

        return {
            "status": status, "score": score,
            "issues": issues, "fixes": fixes,
            "details": {
                "mode": "hyde_proxy",
                "top_k": k,
                "overlap_pct": round(overlap_pct, 4),
                "mismatch_pct": round(mismatch_pct, 4),
                "rank_correlation": round(rank_corr, 4),
            },
        }


def _rank_correlation(ranks_a, ranks_b):
    if len(ranks_a) < 2:
        return 1.0
    n = len(ranks_a)
    d2 = sum((a - b) ** 2 for a, b in zip(ranks_a, ranks_b))
    return 1 - (6 * d2) / (n * (n ** 2 - 1))
