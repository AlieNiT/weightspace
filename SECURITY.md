# Security

## Reporting a vulnerability

Please don't open a public issue. Use [GitHub private vulnerability reporting](https://github.com/leofmarciano/weightspace/security/advisories/new) or email leonardo.marciano@toolzz.me. You should get an answer within a week.

## Threat model

The Python sandbox that the LLM uses runs **in your process**. It is not a security boundary: code written by the LLM can do anything your Python process can do. Run untrusted models or LLMs in a container or VM, and don't keep secrets in the environment you don't want them to reach.

Loading `.pt` checkpoints uses `torch.load`, which can run arbitrary code for untrusted files. Prefer `.safetensors`.
