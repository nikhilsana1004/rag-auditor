from rag_auditor.llm.base import BaseLLM, LLMResponse

DEFAULT_MODEL = "claude-3-5-haiku-20241022"


class AnthropicLLM(BaseLLM):
    def __init__(self, api_key: str, model: str = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model or DEFAULT_MODEL

    def complete(self, system: str, user: str) -> LLMResponse:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = response.content[0].text
        return LLMResponse(
            answer=text,
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
