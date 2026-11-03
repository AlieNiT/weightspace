from types import SimpleNamespace

from weightspace.llm import call, reply
from weightspace.llm.anthropic import to_anthropic
from weightspace.llm.openai import OpenAIChat, to_openai_messages


def _conversation():
    a, b = call("list_tensors"), call("read_tensor", name="x")
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        reply("looking", a, b).to_message(),
        {"role": "tool", "tool_call_id": a.id, "name": a.name, "content": "out1"},
        {"role": "tool", "tool_call_id": b.id, "name": b.name, "content": "out2"},
        {"role": "user", "content": "continue"},
    ]


def test_anthropic_conversion_merges_tool_results():
    system, msgs = to_anthropic(_conversation())
    assert system == "sys"
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert [b["type"] for b in msgs[1]["content"]] == ["text", "tool_use", "tool_use"]
    assert [b["type"] for b in msgs[2]["content"]] == ["tool_result", "tool_result", "text"]


def test_openai_conversion_and_parsing():
    msgs = to_openai_messages(_conversation())
    assert msgs[2]["tool_calls"][1]["function"]["arguments"] == '{"name": "x"}'
    assert msgs[3] == {"role": "tool", "tool_call_id": msgs[2]["tool_calls"][0]["id"], "content": "out1"}

    tc = SimpleNamespace(id="c1", function=SimpleNamespace(name="submit", arguments='{"answer": "a"}'))
    resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tc]))], usage=None)
    captured = {}

    def create(**kw):
        captured.update(kw)
        return resp

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    out = OpenAIChat("m", client=client, temperature=0).complete(
        [{"role": "user", "content": "hi"}], [{"name": "submit", "description": "d", "parameters": {}}]
    )
    assert out.tool_calls[0].arguments == {"answer": "a"}
    assert captured["tools"][0]["function"]["name"] == "submit"
    assert captured["temperature"] == 0


def test_anthropic_parses_real_sdk_types():
    from anthropic.types import Message

    from weightspace.llm.anthropic import AnthropicChat

    msg = Message.model_validate(
        {
            "id": "m1",
            "type": "message",
            "role": "assistant",
            "model": "x",
            "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": "reading"},
                {"type": "tool_use", "id": "tu1", "name": "read_tensor", "input": {"name": "w"}},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    )
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: msg))
    out = AnthropicChat("x", client=client).complete([{"role": "user", "content": "hi"}])
    assert out.text == "reading"
    assert out.tool_calls[0].arguments == {"name": "w"}
    replay = to_anthropic([{"role": "user", "content": "hi"}, out.to_message()])[1][1]["content"]
    assert replay[1]["type"] == "tool_use" and replay[1]["input"] == {"name": "w"}


def _capture_openai(**kw):
    captured = {}
    resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))], usage=None)

    def create(**k):
        captured.update(k)
        return resp

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    OpenAIChat("m", client=client, **kw).complete([{"role": "user", "content": "hi"}])
    return captured


def test_openai_thinking_styles():
    assert "reasoning_effort" not in _capture_openai()
    assert _capture_openai(thinking=False)["reasoning_effort"] == "none"
    assert _capture_openai(thinking=False, off_effort="minimal")["reasoning_effort"] == "minimal"
    assert _capture_openai(thinking="high")["reasoning_effort"] == "high"
    assert _capture_openai(thinking="off", thinking_style="openrouter")["extra_body"] == {"reasoning": {"enabled": False}}
    c = _capture_openai(thinking="off", thinking_style="chat_template", extra_body={"top_k": 20})
    assert c["extra_body"] == {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}
    assert "reasoning_effort" not in _capture_openai(thinking="off", thinking_style=None)


def test_anthropic_thinking():
    from anthropic.types import Message

    from weightspace.llm.anthropic import AnthropicChat

    msg = Message.model_validate(
        {
            "id": "m",
            "type": "message",
            "role": "assistant",
            "model": "x",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    seen = {}

    def create(**kw):
        seen.clear()
        seen.update(kw)
        return msg

    llm = AnthropicChat("x", client=SimpleNamespace(messages=SimpleNamespace(create=create)), max_tokens=4000)
    llm.complete([{"role": "user", "content": "hi"}])
    assert "thinking" not in seen
    llm.thinking = "off"
    llm.complete([{"role": "user", "content": "hi"}])
    assert seen["thinking"] == {"type": "disabled"}
    llm.thinking = "high"
    llm.complete([{"role": "user", "content": "hi"}])
    assert seen["thinking"] == {"type": "adaptive"} and seen["output_config"] == {"effort": "high"}
    assert llm.thinking_style == "adaptive"

    calls = []

    def create_old(**kw):
        calls.append(kw)
        if kw["thinking"]["type"] == "adaptive":
            raise ValueError('"thinking.type.adaptive" is not supported for this model')
        return msg

    old = AnthropicChat("x", client=SimpleNamespace(messages=SimpleNamespace(create=create_old)), thinking="high")
    old.complete([{"role": "user", "content": "hi"}])
    assert calls[-1]["thinking"]["budget_tokens"] == 16384 and calls[-1]["max_tokens"] > 16384
    assert "output_config" not in calls[-1] and old.thinking_style == "budget"
    old.complete([{"role": "user", "content": "hi"}])
    assert len(calls) == 3


def test_openresponses_replays_items():
    from weightspace.llm import get_llm

    sent = []
    out = [
        {"type": "reasoning", "id": "r1", "encrypted_content": "xyz", "summary": []},
        {"type": "function_call", "id": "fc1", "call_id": "c1", "name": "submit", "arguments": '{"answer": "a"}'},
    ]

    def create(**kw):
        sent.append(kw)
        return SimpleNamespace(output=[SimpleNamespace(model_dump=lambda exclude_none, o=o: o) for o in out], usage=None)

    llm = get_llm("openresponses:m", client=SimpleNamespace(responses=SimpleNamespace(create=create)), thinking="high")
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
    r = llm.complete(msgs, tools=[{"name": "submit", "description": "d", "parameters": {"type": "object"}}])
    assert r.tool_calls[0].arguments == {"answer": "a"}
    assert sent[0]["instructions"] == "sys" and sent[0]["reasoning"] == {"effort": "high"} and sent[0]["store"] is False
    msgs += [r.to_message(), {"role": "tool", "tool_call_id": "c1", "name": "submit", "content": "ok"}]
    llm.complete(msgs)
    assert sent[0]["input"] == [{"type": "message", "role": "user", "content": "go"}]
    assert llm.name == "openresponses:m" and get_llm("openai-responses:m", client=llm.client).name == llm.name
    assert sent[1]["input"][1:] == out + [{"type": "function_call_output", "call_id": "c1", "output": "ok"}]
