"""Tests for rag-auditor — core auditors + LLM/reranker integration."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_auditor.ingest.loader import load_documents, Document
from rag_auditor.utils.chunker import fixed_chunker, sentence_chunker
from rag_auditor.auditors.chunking import ChunkingAuditor
from rag_auditor.auditors.retrieval import RetrievalAuditor
from rag_auditor.auditors.lost_in_middle import LostInMiddleAuditor
from rag_auditor.auditors.rg_alignment import RGAlignmentAuditor
from rag_auditor.llm.base import LLMResponse, BaseLLM, _parse_citations
from rag_auditor.reranker.base import BaseReranker, RankedChunk

SAMPLE = Path(__file__).parent.parent / "docs/sample/acme_refund_policy.txt"
QUERY  = "How long does a refund take?"


@pytest.fixture
def sample_docs():
    return load_documents(SAMPLE)


# ── Core auditors ──────────────────────────────────────────────────────────────

def test_load_documents(sample_docs):
    assert len(sample_docs) >= 1
    assert all(isinstance(d, Document) for d in sample_docs)
    assert all(len(d.content) > 0 for d in sample_docs)


def test_fixed_chunker(sample_docs):
    chunks = fixed_chunker(sample_docs, chunk_size=100, overlap=10)
    assert len(chunks) > 0


def test_sentence_chunker(sample_docs):
    chunks = sentence_chunker(sample_docs, chunk_size=100, overlap=10)
    assert len(chunks) > 0


def test_chunking_auditor(sample_docs):
    result = ChunkingAuditor(chunk_size=512, overlap=50).run(sample_docs, QUERY)
    assert result["status"] in ("pass", "warn", "fail")
    assert 0.0 <= result["score"] <= 1.0
    assert isinstance(result["issues"], list)


def test_retrieval_auditor(sample_docs):
    result = RetrievalAuditor(top_k=3).run(sample_docs, QUERY)
    assert result["status"] in ("pass", "warn", "fail")
    assert result["details"]["total_chunks"] > 0


def test_lost_in_middle_no_reranker(sample_docs):
    result = LostInMiddleAuditor(top_k=5).run(sample_docs, QUERY)
    assert result["status"] in ("pass", "warn", "fail")
    assert "best_position" in result["details"]["before"]
    assert result["details"]["reranker_applied"] is False


def test_rg_alignment_no_llm(sample_docs):
    result = RGAlignmentAuditor(top_k=5).run(sample_docs, QUERY)
    assert result["status"] in ("pass", "warn", "fail")
    assert result["details"]["mode"] == "hyde_proxy"


# ── Citation parser ────────────────────────────────────────────────────────────

def test_parse_citations_basic():
    answer = "The refund takes 5-7 days.\nSOURCES: 1, 3"
    indices = _parse_citations(answer, n_chunks=5)
    assert indices == [0, 2]


def test_parse_citations_none():
    answer = "I don't know.\nSOURCES: none"
    indices = _parse_citations(answer, n_chunks=5)
    assert indices == []


def test_parse_citations_missing():
    answer = "The refund takes 5-7 days."
    indices = _parse_citations(answer, n_chunks=5)
    assert indices == []


def test_parse_citations_out_of_range():
    answer = "SOURCES: 1, 99"
    indices = _parse_citations(answer, n_chunks=3)
    assert 0 in indices
    assert 98 not in indices


# ── Mock LLM integration ───────────────────────────────────────────────────────

class MockLLM(BaseLLM):
    """LLM that always cites chunk #1 (index 0)."""
    def __init__(self, cited=(0,)):
        self.model = "mock-llm"
        self._cited = list(cited)

    def complete(self, system: str, user: str) -> LLMResponse:
        sources = ", ".join(str(i + 1) for i in self._cited)
        return LLMResponse(
            answer=f"The refund takes 5-7 business days.\nSOURCES: {sources}",
            model=self.model,
        )


def test_rg_alignment_with_llm_pass(sample_docs):
    llm = MockLLM(cited=[0])  # cites retriever's top chunk
    result = RGAlignmentAuditor(top_k=3, llm=llm).run(sample_docs, QUERY)
    assert result["details"]["mode"] == "llm"
    assert result["details"]["llm_used_retriever_top1"] is True
    assert result["status"] == "pass"


def test_rg_alignment_with_llm_fail(sample_docs):
    llm = MockLLM(cited=[2])  # cites chunk #3, ignores retriever top
    result = RGAlignmentAuditor(top_k=3, llm=llm).run(sample_docs, QUERY)
    assert result["details"]["mode"] == "llm"
    assert result["details"]["llm_used_retriever_top1"] is False
    assert result["status"] in ("fail", "warn")


def test_rg_alignment_with_llm_no_citations(sample_docs):
    llm = MockLLM(cited=[])
    result = RGAlignmentAuditor(top_k=3, llm=llm).run(sample_docs, QUERY)
    assert result["score"] < 0.5


# ── Mock reranker integration ──────────────────────────────────────────────────

class MockReranker(BaseReranker):
    """Reranker that always promotes the last chunk to rank #1."""
    def rerank(self, query, chunks, top_k=None):
        k = top_k or len(chunks)
        # Reverse order — promotes last to first
        order = list(reversed(range(min(k, len(chunks)))))
        return [
            RankedChunk(
                index=orig_idx,
                text=chunks[orig_idx],
                score=1.0 - (new_rank * 0.1),
                original_rank=orig_idx,
                new_rank=new_rank,
            )
            for new_rank, orig_idx in enumerate(order)
        ]


def test_lost_in_middle_with_reranker(sample_docs):
    reranker = MockReranker()
    result = LostInMiddleAuditor(top_k=3, reranker=reranker).run(sample_docs, QUERY)
    assert result["details"]["reranker_applied"] is True
    assert "before" in result["details"]
    assert "after" in result["details"]
    assert "reranker_scores" in result["details"]


# ── LLM factory ───────────────────────────────────────────────────────────────

def test_get_llm_unknown_provider():
    from rag_auditor.llm import get_llm
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_llm("fakeai", api_key="key")


def test_get_reranker_unknown_provider():
    from rag_auditor.reranker import get_reranker
    with pytest.raises(ValueError, match="Unknown reranker"):
        get_reranker("fakeranker")


def test_get_reranker_cohere_missing_key():
    from rag_auditor.reranker import get_reranker
    with pytest.raises(ValueError, match="requires --reranker-key"):
        get_reranker("cohere", api_key=None)
