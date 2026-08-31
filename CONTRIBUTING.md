# Contributing

Thanks for improving AutoIdea. The project is intentionally conservative about provenance, reproducibility, and generated artifacts.

## Development Setup

```bash
python3 --version  # must be 3.11-3.13
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install -e ".[dev,web]"
```

## Checks

Run these before opening a pull request:

```bash
python -m pytest -q
python -m ruff check autoidea tests
python -m build --sdist --wheel
```

For dashboard changes, also run:

```bash
autoidea web --workspace examples/sample_workspace --port 8765 --no-open
```

## Code Guidelines

- Keep runtime behavior covered by tests.
- Prefer small modules with clear interfaces.
- Keep web dashboard data loading independent from heavy agent runtime imports.
- Treat paper text and workspace artifacts as untrusted input.
- Do not commit generated workspaces, logs, API keys, local paths, or private prompts.

## Documentation

Update README and docs when changing:

- CLI options.
- Configuration fields.
- Workspace artifact formats.
- Web dashboard views or endpoints.
- Installation requirements.

## Pull Request Checklist

- Tests pass.
- Lint passes.
- New behavior is documented.
- No generated workspace artifacts are included.
- No secrets or local-only paths are included.
