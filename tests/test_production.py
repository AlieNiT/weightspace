import json
import threading
import time

import numpy as np
import pytest

import weightspace as wsp
from weightspace.cli import main
from weightspace.events import EventLog
from weightspace.llm import LLMResponse, ScriptedLLM, call, reply
from weightspace.sandbox import Sandbox
from weightspace.usage import Pricing, Usage


def _ws():
    rng = np.random.default_rng(0)
    return wsp.WeightSpace.from_numpy({"a": rng.normal(size=(8, 4)).astype(np.float32), "b": np.ones(5, np.float32)})


def test_usage_normalization_and_cost():
    a = Usage.from_anthropic(
        {"input_tokens": 10, "cache_read_input_tokens": 90, "cache_creation_input_tokens": 5, "output_tokens": 7}
    )
    assert (a.input_tokens, a.cached_input_tokens, a.cache_write_tokens, a.output_tokens) == (105, 90, 5, 7)
    o = Usage.from_openai(
        {
            "input_tokens": 100,
            "output_tokens": 20,
            "input_tokens_details": {"cached_tokens": 60},
            "output_tokens_details": {"reasoning_tokens": 12},
        }
    )
    assert (o.cached_input_tokens, o.reasoning_tokens) == (60, 12)
    c = Usage.from_openai({"prompt_tokens": 3, "completion_tokens": 4})
    assert (a + c).total_tokens == 112 + 7 and (a + c).requests == 2
    p = Pricing(input=1.0, output=10.0, cached_input=0.1, cache_write=1.25)
    assert p.cost(a) == pytest.approx((10 * 1 + 90 * 0.1 + 5 * 1.25 + 7 * 10) / 1e6)
    assert Pricing.coerce((2, 8)).cost(Usage(input_tokens=1_000_000)) == 2


def _with_usage(r, n=1000):
    r.tokens = Usage(input_tokens=n, output_tokens=n, requests=1)
    r.usage = {"input_tokens": n}
    return r


def test_events_usage_and_result_fields(tmp_path):
    events = []
    llm = ScriptedLLM(
        [
            _with_usage(reply(None, call("list_tensors"), call("describe_tensor", name="a"))),
            _with_usage(reply(None, call("submit", answer="ok"))),
        ]
    )
    res = wsp.read(_ws(), "what?", llm=llm, callbacks=[events.append, EventLog(str(tmp_path / "ev.jsonl"))], pricing=(1, 2))
    types = [e.type for e in events]
    assert types[0] == "run_start" and types[-1] == "run_end"
    assert types.count("llm_end") == 2 and types.count("tool_end") == 3
    assert res.ok and res.stop_reason == "submitted" and res.tokens.total_tokens == 4000
    assert res.cost == pytest.approx((2000 * 1 + 2000 * 2) / 1e6) and res.usage["input_tokens"] == 2000
    assert res.run_id and res.timings["llm"]["calls"] == 2
    d = res.to_dict()
    assert d["tokens"]["total_tokens"] == 4000 and d["stop_reason"] == "submitted"
    assert len((tmp_path / "ev.jsonl").read_text().splitlines()) == len(events)


def test_token_budget_forces_submit():
    loop = [_with_usage(reply(None, call("list_tensors"))) for _ in range(10)]
    llm = ScriptedLLM([*loop[:3], _with_usage(reply(None, call("submit", answer="partial")))])
    res = wsp.read(_ws(), "q", llm=llm, token_budget=5000)
    assert res.stop_reason == "token_budget" and res.answer == "partial" and res.steps == 3


def test_cost_budget_needs_pricing():
    with pytest.raises(ValueError):
        wsp.Session(_ws(), ScriptedLLM([]), cost_budget=1.0)


def test_error_keeps_partial_result():
    def boom(messages, tools):
        raise RuntimeError("api down")

    llm = ScriptedLLM([reply(None, call("write_tensor", name="b", values=2.0)), boom])
    with pytest.raises(wsp.SessionError) as ei:
        wsp.fix(_ws(), "x", llm=llm)
    r = ei.value.result
    assert r.stop_reason == "error" and "api down" in r.error and r.diff.changed_params == 5
    assert any(m["role"] == "tool" for m in r.transcript)


def test_interrupt_returns_partial_result():
    def stop(messages, tools):
        raise KeyboardInterrupt

    res = wsp.read(_ws(), "q", llm=ScriptedLLM([reply(None, call("list_tensors")), stop]))
    assert res.stop_reason == "interrupted" and not res.ok and res.steps == 2


def test_read_only_tools_run_in_parallel():
    active, peak, lock = [0], [0], threading.Lock()

    def slow(i):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        return f"r{i}"

    t = wsp.tool("slow", "slow read", parallel=True, i={"type": "integer"})(slow)
    calls = [call("slow", i=i) for i in range(4)]
    res = wsp.read(_ws(), "q", llm=ScriptedLLM([reply(None, *calls), reply(None, call("submit", answer="ok"))]), tools=[t])
    outs = [m["content"] for m in res.transcript if m["role"] == "tool" and m["name"] == "slow"]
    assert outs == ["r0", "r1", "r2", "r3"] and peak[0] > 1


def test_writes_stay_sequential():
    seen = []
    t = wsp.tool("w", "write", i={"type": "integer"})(lambda i: seen.append(threading.current_thread().name) or "ok")
    wsp.read(
        _ws(),
        "q",
        llm=ScriptedLLM([reply(None, call("w", i=1), call("w", i=2)), reply(None, call("submit", answer="ok"))]),
        tools=[t],
    )
    assert seen == [threading.main_thread().name] * 2


def test_replay_from_transcript(tmp_path):
    llm = ScriptedLLM(
        [reply(None, call("write_tensor", name="b", values=[0, 3, 0, 0, 0])), reply(None, call("submit", answer="done"))]
    )
    first = wsp.fix(_ws(), "x", llm=llm, log_path=str(tmp_path / "t.jsonl"))
    again = wsp.fix(_ws(), "x", llm=ScriptedLLM.from_transcript(str(tmp_path / "t.jsonl")))
    assert again.answer == first.answer == "done"
    assert again.ws.patch()["b::idx"].tolist() == first.ws.patch()["b::idx"].tolist() == [0, 1, 2, 3, 4]


def test_chunked_diff_matches_reference(monkeypatch):
    import weightspace._chunks as chunks

    monkeypatch.setattr(chunks, "CHUNK", 7)
    ws = wsp.WeightSpace.from_numpy({"m": np.arange(100, dtype=np.float32).reshape(10, 10)})
    ws.edit("m")[2, 3:9] += 0.5
    ws.edit("m")[9, 9] = -1
    t = ws.diff().tensors[0]
    orig, cur = np.arange(100, dtype=np.float64).reshape(10, 10), ws["m"].astype(np.float64)
    d = cur - orig
    assert t.changed == 7 and t.l2_delta == pytest.approx(np.linalg.norm(d))
    assert t.max_abs_delta == pytest.approx(100) and t.rel_delta == pytest.approx(np.linalg.norm(d) / np.linalg.norm(orig))
    assert sorted(ws.patch()["m::idx"].tolist()) == [23, 24, 25, 26, 27, 28, 99]


def test_describe_cache_invalidates_on_edit():
    ws = _ws()
    first = ws.describe("a")
    assert ws.describe("a") is first
    ws.edit("a")[:] = 0
    assert ws.describe("a") != first


def test_gram_singular_values():
    from weightspace.render import singular_values

    m = np.random.default_rng(1).normal(size=(200, 12))
    np.testing.assert_allclose(singular_values(m), np.linalg.svd(m, compute_uv=False), rtol=1e-8)
    np.testing.assert_allclose(singular_values(m.T), np.linalg.svd(m, compute_uv=False), rtol=1e-8)


def test_sandbox_timeout_in_worker_thread():
    sb = Sandbox(timeout=0.2)
    out = []
    th = threading.Thread(target=lambda: out.append(sb.run("while True:\n    pass")))
    th.start()
    th.join(5)
    assert not th.is_alive() and "TimeoutError" in out[0]
    assert sb.run("1 + 1") == "2\n"


def test_sandbox_no_grad():
    torch = pytest.importorskip("torch")
    sb = Sandbox({"torch": torch}, no_grad=True)
    assert "False" in sb.run("x = torch.ones(2, requires_grad=True)\n(x * 2).requires_grad")


def test_openai_prompt_cache_key_and_usage():
    from types import SimpleNamespace as NS

    from weightspace.llm.openai import OpenAIChat

    sent = {}

    def create(**kw):
        sent.update(kw)
        usage = NS(
            model_dump=lambda: {"prompt_tokens": 50, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 40}}
        )
        msg = NS(content="hi", tool_calls=None)
        return NS(choices=[NS(message=msg, finish_reason="stop")], usage=usage)

    llm = OpenAIChat("gpt-x", client=NS(chat=NS(completions=NS(create=create))), prompt_cache_key=True, parallel_tool_calls=True)
    llm.session_key = "abc"
    r = llm.complete([{"role": "user", "content": "x"}])
    assert sent["prompt_cache_key"] == "weightspace-abc" and "parallel_tool_calls" not in sent
    assert r.tokens.cached_input_tokens == 40 and r.stop_reason == "stop"


def test_cli_probe_json_and_replay(tmp_path, capsys):
    np.savez(tmp_path / "w.npz", a=np.ones((2, 2), np.float32))
    (tmp_path / "p.py").write_text("def ping(x: int):\n    '''echo'''\n    return {'x': x}\n")
    t = tmp_path / "t.jsonl"
    lines = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "name": "forward_ping", "arguments": {"x": 3}}]},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "2", "name": "submit", "arguments": {"answer": "ok"}}]},
    ]
    t.write_text("\n".join(json.dumps(m) for m in lines))
    rc = main(
        [
            "read",
            str(tmp_path / "w.npz"),
            "q",
            "--replay",
            str(t),
            "--probe",
            f"{tmp_path / 'p.py'}:ping",
            "--forward-budget",
            "2",
            "--json",
        ]
    )
    d = json.loads(capsys.readouterr().out)
    assert rc == 0 and d["answer"] == "ok" and d["forward_used"] == 1


def test_scripted_text_mode_reports_tokens():
    r = LLMResponse(text="ANSWER: 4", tokens=Usage(input_tokens=5, output_tokens=1, requests=1))
    res = wsp.read(_ws(), "q", llm=ScriptedLLM([r]), text_mode=True)
    assert res.answer == "4" and res.tokens.total_tokens == 6


def test_opentelemetry_spans():
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from weightspace.otel import OpenTelemetryCallback

    exp, tp = InMemorySpanExporter(), TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exp))
    llm = ScriptedLLM([_with_usage(reply(None, call("list_tensors"))), reply(None, call("submit", answer="x"))])
    wsp.read(_ws(), "q", llm=llm, callbacks=[OpenTelemetryCallback(tp.get_tracer("t"))])
    spans = {s.name: s for s in exp.get_finished_spans()}
    root = spans["invoke_agent weightspace"]
    assert spans["execute_tool list_tensors"].parent.span_id == root.context.span_id
    assert spans["chat scripted"].attributes["gen_ai.operation.name"] == "chat"
    assert root.attributes["weightspace.stop_reason"] == "submitted"
