"""Token usage normalized across providers, and what it costs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass
class Usage:
    """Token counts for one or more LLM requests.

    `input_tokens` counts every prompt token, cached or not; `cached_input_tokens` and `cache_write_tokens` are the
    parts read from and written to the provider's prompt cache. `output_tokens` includes `reasoning_tokens`.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    requests: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(**{f.name: getattr(self, f.name) + getattr(other, f.name) for f in fields(self)})

    def to_dict(self) -> dict:
        return {**asdict(self), "total_tokens": self.total_tokens}

    @classmethod
    def from_openai(cls, u: dict | None) -> Usage:
        """Chat Completions (`prompt_tokens`...) or Responses (`input_tokens`...) usage."""
        if not u:
            return cls(requests=1)
        in_details = u.get("prompt_tokens_details") or u.get("input_tokens_details") or {}
        out_details = u.get("completion_tokens_details") or u.get("output_tokens_details") or {}
        return cls(
            input_tokens=int(u.get("prompt_tokens", u.get("input_tokens")) or 0),
            output_tokens=int(u.get("completion_tokens", u.get("output_tokens")) or 0),
            cached_input_tokens=int(in_details.get("cached_tokens") or 0),
            reasoning_tokens=int(out_details.get("reasoning_tokens") or 0),
            requests=1,
        )

    @classmethod
    def from_anthropic(cls, u: dict | None) -> Usage:
        if not u:
            return cls(requests=1)
        read, write = int(u.get("cache_read_input_tokens") or 0), int(u.get("cache_creation_input_tokens") or 0)
        thinking = (u.get("output_tokens_details") or {}).get("thinking_tokens")
        return cls(
            input_tokens=int(u.get("input_tokens") or 0) + read + write,
            output_tokens=int(u.get("output_tokens") or 0),
            cached_input_tokens=read,
            cache_write_tokens=write,
            reasoning_tokens=int(thinking or 0),
            requests=1,
        )


@dataclass(frozen=True)
class Pricing:
    """USD per million tokens. Cache prices default to the input price when not given."""

    input: float
    output: float
    cached_input: float | None = None
    cache_write: float | None = None

    @classmethod
    def coerce(cls, value) -> Pricing | None:
        if value is None or isinstance(value, Pricing):
            return value
        if isinstance(value, dict):
            return cls(**value)
        return cls(*value)

    def cost(self, u: Usage) -> float:
        cached = self.input if self.cached_input is None else self.cached_input
        write = self.input if self.cache_write is None else self.cache_write
        fresh = max(u.input_tokens - u.cached_input_tokens - u.cache_write_tokens, 0)
        return (
            fresh * self.input + u.cached_input_tokens * cached + u.cache_write_tokens * write + u.output_tokens * self.output
        ) / 1e6
