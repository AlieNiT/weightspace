# Providers and thinking

Pass `llm=` as a string or an `LLM` instance:

| String | Backend | Key |
|---|---|---|
| `openai:<model>` | OpenAI Chat Completions | `OPENAI_API_KEY` |
| `openrouter:<model>` | OpenRouter | `OPENROUTER_API_KEY` |
| `anthropic:<model>` | Anthropic Messages | `ANTHROPIC_API_KEY` |

For vLLM, SGLang, Ollama or any OpenAI-compatible server, use `OpenAIChat(model, base_url=..., api_key=...)`.

## Thinking

`thinking=None` keeps the provider default. `False` or `"off"` disables reasoning and tells the LLM to go straight to tool calls. Levels are `"low"`, `"medium"`, `"high"`, `"xhigh"` and `"max"`.

- OpenAI: `reasoning_effort`. Some models reject `"none"`, so set `OpenAIChat(..., off_effort="minimal")`.
- OpenRouter: `reasoning.enabled` / `reasoning.effort`.
- vLLM / SGLang chat templates: `OpenAIChat(..., thinking_style="chat_template")` sends `enable_thinking`.
- Anthropic: adaptive thinking with `output_config.effort`, falling back to a token budget on models that don't support it. Prompt caching is on by default (`AnthropicChat(..., cache=False)` to turn it off).

## Custom LLMs

Subclass `weightspace.llm.LLM` and implement `complete(messages, tools) -> LLMResponse`. `wsp.ScriptedLLM([...])` replays a fixed script for offline tests.
