from typing import List
from rag_auditor.reranker.base import BaseReranker, RankedChunk

DEFAULT_MODEL = "rerank-english-v3.0"


class CohereReranker(BaseReranker):
    """
    Reranker using the Cohere Rerank API.
    Requires a Cohere API key (free tier available at cohere.com).
    Default model: rerank-english-v3.0
    """

    def __init__(self, api_key: str, model: str = None):
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "cohere package not installed. Run: pip install cohere"
            )
        self.client = cohere.Client(api_key)
        self.model_name = model or DEFAULT_MODEL

    def rerank(self, query: str, chunks: List[str], top_k: int = None) -> List[RankedChunk]:
        k = top_k or len(chunks)
        response = self.client.rerank(
            model=self.model_name,
            query=query,
            documents=chunks,
            top_n=k,
        )

        return [
            RankedChunk(
                index=result.index,
                text=chunks[result.index],
                score=round(result.relevance_score, 4),
                original_rank=result.index,
                new_rank=new_rank,
            )
            for new_rank, result in enumerate(response.results)
        ]
