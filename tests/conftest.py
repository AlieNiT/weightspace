import numpy as np
import pytest
import torch


@pytest.fixture
def quadrant_mlp():
    """Tiny MLP trained to tell which quadrant a 2D point is in, plus a scorer."""
    torch.manual_seed(0)
    x = torch.randn(2048, 2)
    y = (x[:, 0] > 0).long() * 2 + (x[:, 1] > 0).long()
    model = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4))
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(model(x), y).backward()
        opt.step()

    def per_class_accuracy():
        with torch.no_grad():
            pred = model(x).argmax(-1)
        return {int(c): float((pred[y == c] == c).float().mean()) for c in range(4)}

    return model, per_class_accuracy


@pytest.fixture
def arrays():
    rng = np.random.default_rng(0)
    return {"a.weight": rng.normal(size=(4, 3)).astype(np.float32), "a.bias": np.zeros(4, np.float32)}
