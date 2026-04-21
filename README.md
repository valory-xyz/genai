# genai

LLM and AI connections for [Open Autonomy](https://github.com/valory-xyz/open-autonomy) agents — thin wrappers that expose commercial AI APIs (Google Gemini, OpenAI) through the AEA messaging layer, with optional [x402](https://x402.org/) payment support.

## What's in this repo

| Package | Public ID | Description |
|---|---|---|
| Connection | `valory/genai` | Wrapper around Google's [google-generativeai](https://pypi.org/project/google-generativeai/) SDK (Gemini). Supports optional x402 payment routing. |
| Connection | `valory/openai` | Wrapper around the OpenAI SDK (chat completions). |
| Connection | `valory/x402` | Client for the x402 HTTP payment protocol — used by the `genai` connection when `use_x402=true`. |
| Protocol | `valory/llm` | LLM request/response protocol (prompt → completion). |
| Protocol | `valory/srr` | Simple json-based request/response protocol used by the genai connection. |

All packages live under `packages/` and are published to IPFS via the Open Autonomy registry.

## Requirements

- Python `>=3.10, <3.15`
- [uv](https://docs.astral.sh/uv/)

## Install

```bash
uv sync
source .venv/bin/activate
autonomy packages sync
```

## Development

```bash
make format          # black + isort
make code-checks     # lint + type check
make security        # bandit + safety
make generators      # regenerate protocols + hashes
make common-checks-1 # hash + copyright + docs + deps
```

See `CONTRIBUTING.md` for the full pre-PR checklist.

## License

Licensed under Apache License 2.0.
