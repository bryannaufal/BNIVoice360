# Contributing to BNIVoice360

## Working on a feature

Keep a change within the layer that owns its behavior:

| Change | Start here |
| --- | --- |
| Page layout or copy | `bnivoice360/pages/<page>.py` |
| Shared widgets or cards | `bnivoice360/ui/` |
| Theme | `bnivoice360/ui/assets/styles.css` |
| Chart styling | `bnivoice360/ui/charts.py` |
| Aggregations or filters | `bnivoice360/services/analytics.py` |
| Data validation | `bnivoice360/services/ingestion.py` |
| SQL or database access | `bnivoice360/data/repository.py` |
| NLP model or crawler integration | Adapter interface in `services/`, wiring in `container.py` |
| Common fields and settings | `domain.py` and `config.py` |

Pages have a single `render(services: AppServices) -> None` entry point. Keep SQL out
of pages and widgets. Keep Streamlit imports out of services and persistence so
these modules remain reusable in scripts and tests.

Pass dependencies explicitly through constructors or page parameters. Assemble
them in `container.py`; avoid constructing database connections inside widgets.
Use immutable dataclasses for requests and results, and plain functions for
stateless transformations. Add another interface only when there is a real
replacement point.

## Coding conventions

Use snake_case, descriptive names, public-function type annotations, and short
docstrings that explain intent. Database fields such as `is_IT` retain their
existing names for schema compatibility. Ruff enforces imports, common correctness
checks, and a consistent 100-character format.

Use meaningful widget keys scoped to the feature or source. Shared session-state
defaults and rerun notices belong in `ui/state.py`. Escape user-supplied values
before interpolating them into HTML. Present validation errors through the UI,
and let persistence errors retain their details for diagnosis.

## Verify a change

```powershell
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format .
python -m pytest
```

Add behavior-focused tests when changing validation, queries, analytics,
enrichment, or state transitions. Use the `services` fixture for an empty temporary
database and `streamlit_app` for the seeded app. Never test write paths against a
shared or production database.

Review page flows manually when changing visuals. Automated Streamlit tests verify
widget behavior and runtime errors; they do not replace browser layout inspection.

## Review and shared data

Keep feature branches and pull requests focused so multiple contributors can own
different pages or services. Include the behavior changed and checks run in the
pull request description. Do not commit database files, secrets, local development
tools, or virtual environments.

For schema changes, describe how existing records remain compatible and add an
upgrade test. The current startup uses additive schema/index initialization; do
not introduce destructive migrations or reseeding as a shortcut.

More extension examples are in [the architecture guide](docs/architecture.md).
