from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LLMResponse:
    answer: str
    cited_indices: List[int] = field(default_factory=list)  # chunk indices the LLM used
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


class BaseLLM(ABC):
    """Common interface for all LLM providers."""

    @abstractmethod
    def complete(self, system: str, user: str) -> LLMResponse:
        """Send a system + user prompt, return a structured response."""
        ...

    def answer_with_chunks(self, query: str, chunks: List[str]) -> LLMResponse:
        """
        Ask the LLM to answer a query given numbered chunks.
        Returns the answer + which chunk numbers it cited.
        """
        numbered = "\n\n".join(
            f"[CHUNK {i+1}]\n{c}" for i, c in enumerate(chunks)
        )
        system = (
            "You are a precise question-answering assistant. "
            "Answer the user's question using ONLY the provided chunks. "
            "At the end of your answer, on a new line, write: "
            "SOURCES: followed by a comma-separated list of the chunk numbers "
            "you actually used (e.g. SOURCES: 1, 3). "
            "If no chunk is relevant, write SOURCES: none."
        )
        user = f"Chunks:\n{numbered}\n\nQuestion: {query}"
        response = self.complete(system=system, user=user)
        response.cited_indices = _parse_citations(response.answer, len(chunks))
        return response


def _parse_citations(answer: str, n_chunks: int) -> List[int]:
    """Extract zero-based chunk indices from 'SOURCES: 1, 3' lines."""
    import re
    match = re.search(r"SOURCES:\s*(.+)", answer, re.IGNORECASE)
    if not match:
        return []
    raw = match.group(1).strip()
    if raw.lower() == "none":
        return []
    indices = []
    for token in re.split(r"[,\s]+", raw):
        token = token.strip()
        if token.isdigit():
            idx = int(token) - 1  # convert 1-based to 0-based
            if 0 <= idx < n_chunks:
                indices.append(idx)
    return indices
