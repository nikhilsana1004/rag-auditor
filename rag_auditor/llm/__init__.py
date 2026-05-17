from rag_auditor.llm.base import BaseLLM, LLMResponse

def get_llm(provider: str, api_key: str, model: str = None) -> BaseLLM:
    provider = provider.lower().strip()
    if provider == "anthropic":
        from rag_auditor.llm.anthropic_llm import AnthropicLLM
        return AnthropicLLM(api_key=api_key, model=model)
    elif provider == "openai":
        from rag_auditor.llm.openai_llm import OpenAILLM
        return OpenAILLM(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unknown LLM provider: '{provider}'. Choose from: anthropic, openai")
