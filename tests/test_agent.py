import json

import numpy as np
import torch

import weightspace as wsp
from weightspace.llm import ScriptedLLM, call, reply


def test_fix_flow_with_scripted_llm(quadrant_mlp, tmp_path):
    model, acc = quadrant_mlp
    with torch.no_grad():
        good_w, good_b = model[2].weight[3].clone(), model[2].bias[3].clone()
        model[2].weight[3] *= -1
        model[2].bias[3] = -10
    assert acc()[3] < 0.05

    probe = wsp.Probe(acc, name="accuracy", description="Per-class accuracy on 2048 points.")
    fix_code = f"row = W['2.weight'][3]\nrow *= -1\nW['2.bias'][3] = {float(good_b)}\ndiff()"
    llm = ScriptedLLM(
        [
            reply("Let me look.", call("list_tensors"), call("forward_accuracy")),
            reply(None, call("describe_tensor", name="2.weight"), call("checkpoint", label="before")),
            reply("Class 3 output row is negated and its bias is very negative.", call("python", code=fix_code)),
            reply(None, call("forward_accuracy")),
            reply(None, call("forward_accuracy")),  # over budget
            reply(None, call("submit", answer="Un-negated output row 3 and reset its bias.", confidence=0.9)),
        ]
    )
    res = wsp.fix(model, "The model never predicts class 3.", llm, probes=[probe], forward_budget=2)

    assert res.answer.startswith("Un-negated")
    assert res.confidence == 0.9
    assert res.forward_used == 2
    assert "budget exhausted" in [m for m in res.transcript if m["role"] == "tool"][-2]["content"]
    assert acc()[3] > 0.9
    torch.testing.assert_close(model[2].weight[3], good_w)
    assert res.diff.changed_params == 17
    assert {t.name for t in res.diff.tensors} == {"2.weight", "2.bias"}

    res.save(str(tmp_path))
    summary = json.loads((tmp_path / "result.json").read_text())
    assert summary["changed_params"] == 17
    assert (tmp_path / "patch.npz").exists()


def test_read_mode_cannot_edit(arrays):
    llm = ScriptedLLM(
        [
            reply(None, call("python", code="W['a.weight'][:] = 0"), call("python", code="R['a.weight'][0,0] = 1")),
            reply(None, call("write_tensor", name="a.weight", values=0)),
            reply(None, call("read_tensor", name="a.weight", index="0:2")),
            reply(None, call("submit", answer="it is a 3->4 linear layer")),
        ]
    )
    s = wsp.Session(wsp.WeightSpace.from_numpy(arrays), llm, mode="read")
    assert "write_tensor" not in s.tools and "checkpoint" not in s.tools
    res = s.run("What is this?")
    outs = [m["content"] for m in res.transcript if m["role"] == "tool"]
    assert "NameError" in outs[0]
    assert "read-only" in outs[1]
    assert "unknown tool" in outs[2]
    assert "shape=(2, 3)" in outs[3]
    assert res.diff.changed_params == 0


def test_write_mode_zeroes_and_tracks_from_zero(arrays):
    llm = ScriptedLLM(
        [
            reply(None, call("write_tensor", name="a.bias", index="1", values=2.0)),
            reply(None, call("submit", answer="done")),
        ]
    )
    res = wsp.write(dict(arrays), "make bias[1] two", llm)
    assert np.all(res.ws["a.weight"] == 0)
    assert res.diff.changed_params == 1


def test_nudges_then_step_limit(arrays):
    llm = ScriptedLLM([reply("thinking"), reply("still thinking"), reply("final words")])
    res = wsp.read(dict(arrays), "?", llm)
    assert res.answer == "final words"

    llm = ScriptedLLM([reply(None, call("list_tensors"))] * 3 + [reply(None, call("submit", answer="forced"))])
    res = wsp.read(dict(arrays), "?", llm, max_steps=3)
    assert res.answer == "forced" and res.steps == 3


def test_custom_tool_and_probe_schema(arrays):
    @wsp.probe(cost=2)
    def sample(n: int, prompt: str = "hi"):
        """Generate n samples."""
        return {"n": n, "prompt": prompt, "arr": np.arange(2)}

    assert sample.parameters["required"] == ["n"]
    extra = wsp.tool("double", "Double x.", x={"type": "number"})(lambda x: x * 2)
    llm = ScriptedLLM(
        [
            reply(None, call("double", x=21), call("forward_sample", n=3)),
            reply(None, call("submit", answer="ok")),
        ]
    )
    res = wsp.read(dict(arrays), "?", llm, probes=[sample], forward_budget=5, tools=[extra])
    outs = [m["content"] for m in res.transcript if m["role"] == "tool"]
    assert outs[0] == "42"
    assert '"arr"' in outs[1] and "3 of 5 left" in outs[1]


def test_text_mode_weights_block_and_python(arrays):
    reply_text = (
        "Plan.\n```weights\n### a.bias [4]\n1 2 3 4\n```\n"
        "```python\nW['a.weight'][0] = 0\n```\nANSWER: set biases\nCONFIDENCE: 0.7\n"
    )
    llm = ScriptedLLM([reply(reply_text)])
    res = wsp.fix(dict(arrays), "fix it", llm, text_mode=True)
    assert res.answer == "set biases" and res.confidence == 0.7
    np.testing.assert_array_equal(res.ws["a.bias"], [1, 2, 3, 4])
    assert np.all(res.ws["a.weight"][0] == 0)
    assert "### a.weight [4, 3]" in llm.seen[0][1]["content"]


def test_thinking_off_reaches_llm_and_prompt(arrays):
    llm = ScriptedLLM([reply(None, call("submit", answer="ok"))])
    res = wsp.fix(dict(arrays), "fix", llm, thinking=False)
    assert llm.thinking_seen == ["off"]
    assert "Thinking is off" in llm.seen[0][0]["content"]
    assert res.thinking == "off"

    llm = ScriptedLLM([reply("ANSWER: x")])
    wsp.read(dict(arrays), "?", llm, text_mode=True, thinking="low")
    assert llm.thinking_seen == ["low"]
    assert "Thinking is off" not in llm.seen[0][0]["content"]

    llm = ScriptedLLM([reply(None, call("submit", answer="ok"))])
    wsp.read(dict(arrays), "?", llm)
    assert llm.thinking_seen == [None]
