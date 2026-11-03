# Providers and thinking

Pass `llm=` as a string or an `LLM` instance:

| String | Backend | Key |
|---|---|---|
| `openai:<model>` | OpenAI Chat Completions | `OPENAI_API_KEY` |
| `openresponses:<model>` | [Open Responses](https://www.openresponses.org) (OpenAI Responses API); needed for reasoning + tools on gpt-6 models | `OPENAI_API_KEY` |
| `openrouter:<model>` | OpenRouter | `OPENROUTER_API_KEY` |
| `anthropic:<model>` | Anthropic Messages | `ANTHROPIC_API_KEY` |

For vLLM, SGLang, Ollama or any OpenAI-compatible server, use `OpenAIChat(model, base_url=..., api_key=...)`.

## Open Responses

[Open Responses](https://www.openresponses.org) is the open, multi-provider spec built on the OpenAI Responses API. `openresponses:<model>` (`OpenResponses`) speaks it: OpenAI by default, or any compliant server with `OpenResponses(model, base_url=..., api_key=...)`. It runs with `store=False` and sends back each turn's output items, encrypted reasoning included, so nothing is kept server-side. `openai-responses:` still works as an alias.

## Thinking

`thinking=None` keeps the provider default. `False` or `"off"` disables reasoning and tells the LLM to go straight to tool calls. Levels are `"low"`, `"medium"`, `"high"`, `"xhigh"` and `"max"`.

- OpenAI: `reasoning_effort` (Chat Completions) or `reasoning.effort` (Open Responses). Some models reject `"none"`, so set `OpenAIChat(..., off_effort="minimal")`.
- OpenRouter: `reasoning.enabled` / `reasoning.effort`.
- vLLM / SGLang chat templates: `OpenAIChat(..., thinking_style="chat_template")` sends `enable_thinking`.
- Anthropic: adaptive thinking with `output_config.effort`, falling back to a token budget on models that don't support it. Models whose thinking is always on (Opus 5.5) reject `disabled`; there `thinking=False` falls back to `effort="low"`, where the model usually skips thinking, plus the act-directly prompt. `usage.thinking_tokens` in `result.json` shows how much it still thought. Prompt caching is on by default (`AnthropicChat(..., cache=False)` to turn it off).

## Fast mode

`speed="fast"` (CLI `--fast`) runs the same editing model with faster inference at a higher price. `None` keeps the provider's standard speed. The value is saved in `result.json`.

- Anthropic: `speed="fast"` on the request, up to 2.5x faster output on Opus 5.5 for about 2x the price.
- OpenAI / Open Responses: sent as `service_tier`. `"fast"` costs about 2x; `speed="ultrafast"` also passes through for models that support it.

```python
wsp.improve(model, "...", llm="anthropic:claude-opus-5-5", speed="fast")
```

## Custom LLMs

Subclass `weightspace.llm.LLM` and implement `complete(messages, tools) -> LLMResponse`. `wsp.ScriptedLLM([...])` replays a fixed script for offline tests.
