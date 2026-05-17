"""
Chunking utilities: fixed-size, sentence-aware, and semantic chunkers.
"""

from typing import List
from dataclasses import dataclass
from rag_auditor.ingest.loader import Document


@dataclass
class Chunk:
    text: str
    source: str
    index: int
    token_count: int
    char_count: int


def fixed_chunker(documents: List[Document], chunk_size: int, overlap: int) -> List[Chunk]:
    """Split documents into fixed-size token chunks (approximated by words)."""
    chunks = []
    for doc in documents:
        words = doc.content.split()
        step = max(1, chunk_size - overlap)
        for i, start in enumerate(range(0, len(words), step)):
            segment = words[start: start + chunk_size]
            if not segment:
                continue
            text = " ".join(segment)
            chunks.append(Chunk(
                text=text,
                source=doc.source,
                index=len(chunks),
                token_count=len(segment),
                char_count=len(text),
            ))
    return chunks


def sentence_chunker(documents: List[Document], chunk_size: int, overlap: int) -> List[Chunk]:
    """Split at sentence boundaries, then group up to chunk_size tokens."""
    import re
    chunks = []
    sentence_end = re.compile(r'(?<=[.!?])\s+')

    for doc in documents:
        sentences = sentence_end.split(doc.content.strip())
        current, current_tokens = [], 0
        for sent in sentences:
            tokens = len(sent.split())
            if current_tokens + tokens > chunk_size and current:
                text = " ".join(current)
                chunks.append(Chunk(
                    text=text,
                    source=doc.source,
                    index=len(chunks),
                    token_count=current_tokens,
                    char_count=len(text),
                ))
                # keep overlap sentences
                overlap_sents = []
                overlap_tokens = 0
                for s in reversed(current):
                    st = len(s.split())
                    if overlap_tokens + st <= overlap:
                        overlap_sents.insert(0, s)
                        overlap_tokens += st
                    else:
                        break
                current = overlap_sents + [sent]
                current_tokens = overlap_tokens + tokens
            else:
                current.append(sent)
                current_tokens += tokens

        if current:
            text = " ".join(current)
            chunks.append(Chunk(
                text=text,
                source=doc.source,
                index=len(chunks),
                token_count=current_tokens,
                char_count=len(text),
            ))
    return chunks


def count_tokens(text: str) -> int:
    """Approximate token count (word-based)."""
    return len(text.split())
