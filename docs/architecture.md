# Architecture

## Dependency direction

```text
app.py → application.py → page renderers → shared UI
               │               │
               └── container ──┴── services → repository → database → SQLite
                                      │
                                  domain/config
```

`application.py` bootstraps Streamlit, initializes session state, assembles
dependencies, and selects a registered page. `container.py` is the composition
root: it is the single place that chooses concrete crawler and classifier
implementations. Services and persistence import neither Streamlit nor page
modules.

The structure uses a small layered design:

- **Repository pattern:** `VoiceRepository` owns SQL and database record access.
- **Service layer:** ingestion, enrichment, and question answering own workflows;
  analytics uses pure functions.
- **Strategy interfaces:** `Crawler` and `VoiceClassifier` are Python protocols,
  allowing implementations to be replaced without changing page code.
- **Constructor injection:** services receive their repository or adapter
  explicitly. Tests can use isolated repositories and stub classifiers.
- **Typed domain values:** immutable dataclasses name request and result fields,
  replacing positional tuples across layers.

The app does not use an ORM, a dependency-injection framework, or abstract base
classes for every module. Add those only when the application's needs justify the
extra dependencies and indirection.

## Database lifecycle

`Database.initialize()` creates the existing `suara_nasabah` table if absent and
adds source/date and urgency/escalation indexes. There is no table rename or
destructive migration. `seed_if_empty()` serializes its empty check and insert in
a transaction to prevent concurrent sessions from seeding twice.

Every repository operation opens a short-lived connection through
`Database.transaction()`. Successful writes commit, exceptions roll back, and
the connection closes in a `finally` block. Values use SQL parameters; query
column names come exclusively from fixed application constants.

Ingestion validates each incoming row before persistence. Duplicate primary keys
are skipped with an explicit conflict clause; other database constraint failures
roll back the batch. Enrichment performs classification outside the write
transaction, then fills missing labels with `COALESCE`, preserving reviewed
values. Escalation is a single idempotent update.

Startup honors `BNIVOICE360_DB_PATH` through `Settings`. Tests supply a path in
`tmp_path`; no production database is needed.

## Streamlit reruns and session state

The sidebar owns the role and navigation widgets. If an admin switches back to
manager while viewing ingestion, navigation resets to Home. The ingestion page
also checks its role when rendered directly.

`ui/state.py` establishes role, page, and chat-history defaults. Source-specific
crawler previews use independent session keys. Successful writes place feedback
in session state and trigger a rerun, so health cards, charts, and feed actions
use a fresh database snapshot. Validation details survive that rerun. Saved
crawler previews are cleared before rerunning to prevent accidental resubmission.

Pages remain orchestration code: they retrieve data, call analytics or services,
and display results. Widget-to-filter conversion lives in `ui/filters.py`;
actual filtering and priority sorting live in `services/analytics.py`.

The voice feed slices the sorted results into `FEED_PAGE_SIZE` (12) cards before
creating any row widgets. Non-widget session keys store pagination and search;
filter callbacks copy selections into durable state so Streamlit's widget
cleanup does not erase them on page navigation. A filter/search context resets
the page only when criteria change, while data updates preserve or clamp it.
Escalation commits through `VoiceRepository.escalate()` before a full app rerun.
Each card uses a keyed `st.container()` and `st.columns([4, 1])` header, with CSS
scoped to those container keys.

`ui/charts.py` owns the figure theme, text colors, auto margins, label wrapping,
and common renderer. Every chart uses `width="stretch"`, `theme=None`, and
responsive Plotly configuration. `.streamlit/config.toml` supplies the light
theme for native widgets and dataframe canvases; custom cards and badges use
explicit contrast colors in `ui/assets/styles.css`.

## Add a page

Create `bnivoice360/pages/customer_journey.py`:

```python
import streamlit as st

from bnivoice360.container import AppServices
from bnivoice360.ui.components import header


def render(services: AppServices) -> None:
    header("Customer Journey", "Journey insights", "VoC Intelligence")
    st.dataframe(services.repository.load())
```

Import it and register its renderer in `application.PAGES`. Add any required
role filtering in `ui/sidebar.py` and a guard in the page, following the ingestion
page. Extend `tests/test_app.py` to navigate to it.

## Replace the demo classifier

Implement the existing protocol with a class exposing
`classify(text: str) -> Prediction`:

```python
from bnivoice360.domain import Prediction


class TrainedClassifier:
    def classify(self, text: str) -> Prediction:
        # Invoke the trained model and map its output to the domain fields.
        return Prediction("Neutral", 1, 0, 0, 0, "Model topic")
```

Pass an instance to `PredictionService(repository, TrainedClassifier())` in
`container.py`. The UI and repository do not need to change. The stub-adapter
test in `tests/test_services.py` demonstrates this boundary.

## Replace the crawler

Implement `crawl(request: CrawlRequest) -> pd.DataFrame` and wire the adapter in
`container.py`. Return `content`, `date_posted`, and `source`; optionally return
stable record IDs for deduplication. Existing source panels construct the typed
request, and the same ingestion service validates the returned frame.

The current simulator honors limits and date bounds. Search, app IDs, ratings,
and branch options affect its seed; it does not contact social platforms.

## Testing boundaries

- `test_repository.py`: initialization preserves records, escalation is
  idempotent, failed batches roll back, and connections close.
- `test_services.py`: validation, duplicate IDs, source mapping, timezone
  normalization, partial enrichment, adapter replacement, filters, sorting,
  aggregates, and companion query scopes.
- `test_app.py`: all pages render, roles update navigation, crawler data is saved
  and predicted, escalation persists, and chat history survives page changes.
- `test_voice_feed.py`: bounded card counts, page/filter/search persistence,
  immediate escalation, empty states, and last-page clamping.
- `test_charts.py`: readable color pairs, explicit chart fonts, automatic
  margins, and long label wrapping.

The local companion understands a limited set of question terms. It reads fresh
data through the repository and cannot run arbitrary SQL or update records.
