# Contributing

Thank you for your interest in improving `langgraph-section-flow`!

## Development setup

```bash
git clone <repo-url>
cd langgraph-section-flow

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

## Running tests

```bash
pytest
```

## Code style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and
formatting.

```bash
pip install ruff
ruff check .
ruff format .
```

## Submitting a pull request

1. Fork the repository and create a feature branch.
2. Write or update tests that cover your change.
3. Ensure `pytest` passes and `ruff check .` reports no errors.
4. Open a PR with a clear description of the problem and solution.

## Reporting issues

Open an issue on GitHub with a minimal reproducible example and the versions
of `langchain`, `langgraph`, and `langgraph-section-flow` you are using.
