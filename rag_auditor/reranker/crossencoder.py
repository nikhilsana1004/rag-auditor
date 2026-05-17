from typing import List, Optional
from rag_auditor.reranker.base import BaseReranker, RankedChunk

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker(BaseReranker):
    """
    Local reranker using a cross-encoder model from sentence-transformers.
    No API key needed. Downloads ~80MB model on first use.
    Default: cross-encoder/ms-marco-MiniLM-L-6-v2
    """

    def __init__(self, model: str = None):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
        self.model_name = model or DEFAULT_MODEL
        self._model = None  # lazy load

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, chunks: List[str], top_k: int = None) -> List[RankedChunk]:
        model = self._get_model()
        pairs = [[query, chunk] for chunk in chunks]
        scores = model.predict(pairs).tolist()

        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )
        if top_k:
            ranked = ranked[:top_k]

        return [
            RankedChunk(
                index=orig_idx,
                text=chunks[orig_idx],
                score=round(float(score), 4),
                original_rank=orig_idx,
                new_rank=new_rank,
            )
            for new_rank, (orig_idx, score) in enumerate(ranked)
        ]
