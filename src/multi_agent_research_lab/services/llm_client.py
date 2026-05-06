"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass
from time import sleep
from typing import Any

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.observability.tracing import configure_langsmith


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client skeleton."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        configure_langsmith(self.settings)

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion.

        Uses OpenAI when `OPENAI_API_KEY` is configured. Without a key, returns a
        deterministic local response so the lab can run offline and tests stay free.
        """

        if not self.settings.openai_api_key:
            return self._mock_complete(system_prompt, user_prompt)
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return self._openai_complete(system_prompt, user_prompt)
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    sleep(2**attempt)
        raise RuntimeError("OpenAI completion failed after 3 attempts") from last_error

    def _mock_complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        input_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt)
        topic = user_prompt.strip().splitlines()[0][:120] if user_prompt.strip() else "the request"
        citations = " ".join(f"[{index}]" for index in range(1, 6) if f"[{index}]" in user_prompt)
        citation_sentence = f" Sources referenced: {citations}." if citations else ""
        content = (
            "Offline LLM response\n\n"
            f"Task: {topic}\n"
            "Summary: This deterministic fallback is used because OPENAI_API_KEY is not set. "
            "It preserves the workflow shape for development, tracing, and benchmark tests."
            f"{citation_sentence}"
        )
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=_estimate_tokens(content),
            cost_usd=0.0,
        )

    def _openai_complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        from openai import OpenAI

        client: Any = OpenAI(
            api_key=self.settings.openai_api_key,
            timeout=self.settings.timeout_seconds,
        )
        if self.settings.langsmith_tracing and self.settings.langsmith_api_key:
            try:
                from langsmith.wrappers import wrap_openai
            except ImportError:
                pass
            else:
                client = wrap_openai(client)
        response = client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None
        cost = _estimate_openai_cost(self.settings.openai_model, input_tokens, output_tokens)
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()) * 4 // 3)


def _estimate_openai_cost(
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    if input_tokens is None or output_tokens is None:
        return None

    prices_per_million = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4.1-mini": (0.40, 1.60),
        "gpt-4.1-nano": (0.10, 0.40),
        "gpt-5.4-mini": (0.75, 4.50),
    }
    input_price, output_price = prices_per_million.get(model, prices_per_million["gpt-4o-mini"])
    return (input_tokens / 1_000_000 * input_price) + (output_tokens / 1_000_000 * output_price)
