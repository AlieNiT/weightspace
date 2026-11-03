import numpy as np

from weightspace import WeightSpace, describe, parse_weights, render_space
from weightspace.render import parse_index
from weightspace.sandbox import Sandbox


def test_parse_index():
    a = np.arange(60).reshape(3, 4, 5)
    for text in ["0", "0:2", ":, 1", "1:3, ::2, -1", "[0, :, 2:4]", "..., 0"]:
        np.testing.assert_array_equal(a[parse_index(text)], eval(f"a[{text.strip('[]')}]"))


def test_render_parse_roundtrip(arrays):
    ws = WeightSpace.from_numpy(arrays)
    back = parse_weights(render_space(ws, precision=8))
    for k, v in arrays.items():
        np.testing.assert_allclose(back[k], v, rtol=1e-6)


def test_describe_mentions_spectrum(arrays):
    text = describe("a.weight", arrays["a.weight"])
    assert "singular values" in text and "row L2 norms" in text


def test_sandbox_persists_and_reports_errors():
    sb = Sandbox({"x": 1})
    assert sb.run("y = x + 1") == "(no output)"
    assert sb.run("y * 10").strip() == "20"
    assert "ZeroDivisionError" in sb.run("1/0")


def test_sandbox_timeout():
    assert "TimeoutError" in Sandbox(timeout=0.2).run("while True: pass")
