# Probes and budgets

A probe is a Python function that the LLM can call as a tool. Before it runs, the current weights are synced into your model. Whatever it returns (a number, a dict, text) is what the LLM observes.

```python
@wsp.probe(cost=1)
def generate(prompts: list, max_new_tokens: int = 256):
    """Generate answers for up to 8 prompts and report repetition."""
    ...
```

- The docstring becomes the tool description. Type hints and defaults become the JSON schema, or pass `parameters={...}` explicitly.
- `cost` is how much of `forward_budget` one call uses. When the budget runs out, the probe refuses and the LLM is told to submit.
- Validate arguments inside the probe and raise `ValueError` to refuse a request. The error message goes back to the LLM.

## Choosing what the LLM can observe

The probe defines the experiment. A probe that only returns attention maps lets the LLM see attention and nothing else. A probe that returns paired original/edited accuracy on held-out problems lets it measure its own edit. Keep evaluation data out of the probes if you want an honest test.

Useful patterns:

- **Paired comparisons**: run the original and edited weights on the same inputs and seeds, and return the difference with its standard error.
- **Teacher forcing**: return per-token log-probs and selected activations for given texts, saved to disk for the sandbox to load.
- **Guards**: refuse edits that are out of scope (for example scaling a whole tensor, or touching the embeddings) before running.

[`experiments/exp8/run_edit.py`](../experiments/exp8/run_edit.py) has full examples of all three.
