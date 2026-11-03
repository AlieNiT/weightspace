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
