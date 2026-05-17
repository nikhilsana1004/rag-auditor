from rag_auditor.reranker.base import BaseReranker, RankedChunk


def get_reranker(provider: str, api_key: str = None, model: str = None) -> BaseReranker:
    provider = provider.lower().strip()
    if provider == "cross-encoder":
        from rag_auditor.reranker.crossencoder import CrossEncoderReranker
        return CrossEncoderReranker(model=model)
    elif provider == "cohere":
        if not api_key:
            raise ValueError("Cohere reranker requires --reranker-key")
        from rag_auditor.reranker.cohere_reranker import CohereReranker
        return CohereReranker(api_key=api_key, model=model)
    else:
        raise ValueError(
            f"Unknown reranker: '{provider}'. Choose from: cross-encoder, cohere"
        )
