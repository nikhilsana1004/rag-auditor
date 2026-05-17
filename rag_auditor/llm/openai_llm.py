from rag_auditor.llm.base import BaseLLM, LLMResponse

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAILLM(BaseLLM):
    def __init__(self, api_key: str, model: str = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model or DEFAULT_MODEL

    def complete(self, system: str, user: str) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=1024,
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
        return LLMResponse(
            answer=text,
            model=self.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )
