from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class RankedChunk:
    index: int        # original chunk index
    text: str
    score: float      # reranker relevance score (higher = more relevant)
    original_rank: int
    new_rank: int


class BaseReranker(ABC):
    """Common interface for all rerankers."""

    @abstractmethod
    def rerank(self, query: str, chunks: List[str], top_k: int = None) -> List[RankedChunk]:
        """
        Re-score and re-order chunks for a given query.
        Returns RankedChunk list sorted by descending score (best first).
        """
        ...

    def rerank_indices(self, query: str, chunks: List[str], top_k: int = None) -> List[int]:
        """Convenience: return just the re-ordered chunk indices."""
        ranked = self.rerank(query, chunks, top_k=top_k)
        return [r.index for r in ranked]
