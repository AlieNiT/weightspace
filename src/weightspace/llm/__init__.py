from __future__ import annotations

from .base import LLM, LLMResponse, ToolCall, normalize_speed, normalize_thinking
from .mock import ScriptedLLM, call, reply


def get_llm(spec: str | LLM, **kw) -> LLM:
    """`"openai:gpt-5"`, `"openresponses:gpt-6-astra"`, `"anthropic:claude-sonnet-4-5"`, or `"openrouter:<model>"`."""
    if isinstance(spec, LLM):
        return spec
    provider, _, model = spec.partition(":")
    if not model:
        provider, model = "openai", provider
    provider = provider.lower()
    if provider == "anthropic":
        from .anthropic import AnthropicChat

        return AnthropicChat(model, **kw)
    if provider == "openrouter":
        import os

        from .openai import OpenAIChat

        kw.setdefault("base_url", "https://openrouter.ai/api/v1")
        kw.setdefault("api_key", os.environ.get("OPENROUTER_API_KEY"))
        kw.setdefault("thinking_style", "openrouter")
        return OpenAIChat(model, **kw)
    if provider in ("openresponses", "openai-responses"):
        from .openai import OpenResponses

        return OpenResponses(model, **kw)
    if provider == "openai":
        from .openai import OpenAIChat

        return OpenAIChat(model, **kw)
    raise ValueError(f"unknown provider {provider!r}")


__all__ = ["normalize_thinking", "normalize_speed", "LLM", "LLMResponse", "ToolCall", "ScriptedLLM", "call", "reply", "get_llm"]
