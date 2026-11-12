# llm
LLM Example

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Layout: `tokenizer/`, `model/`, `train/`, `data/` are top-level packages; shared tests live in `tests/`.
