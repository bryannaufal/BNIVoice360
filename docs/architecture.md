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

`Database.initialize()` creates `suara_nasabah` and `topic_detail`, with a nullable
`topic_id` foreign key and indexes for source/date, urgency/escalation, and topics.
Each connection enables foreign-key enforcement. Legacy single-table databases
are backed up, then migrated transactionally, preserving voices and escalation.
The status schema uses `escalation_status TEXT NOT NULL DEFAULT 'Open'` with a
CHECK constraint permitting only Open, Escalated, and Resolved. Both single-table
and two-table binary schemas migrate with a backup: 0 becomes Open, 1 becomes
Escalated. Existing topic details, voice IDs, author information, and timestamps
are retained; schema version is 3.
The repository derives `topic_cluster` from the linked title when reading data.
`NULL` is displayed as Unlabeled and `-1` as Outlier, irrespective of its CSV title.

`data/seed.py` validates the three configured CSV sources and inserts topics and
voices in one transaction. Empty databases initialize automatically unless
`BNIVOICE360_AUTO_INITIALIZE=0`. Existing databases are always preserved on app
startup. Only the initialization CLI consumes `BNIVOICE360_INIT_OVERWRITE` or
`--overwrite`; it creates a SQLite backup before replacing data. An immediate
write transaction serializes the final empty check and insertion across sessions.
`data/demo.py` supplies deterministic fixtures for offline tests.

Every repository operation opens a short-lived connection through
`Database.transaction()`. Successful writes commit, exceptions roll back, and
the connection closes in a `finally` block. Values use SQL parameters; query
column names come exclusively from fixed application constants.

Ingestion validates each incoming row before persistence. Duplicate primary keys
are skipped with an explicit conflict clause; other database constraint failures
roll back the batch. Enrichment performs classification outside the write
transaction, then fills missing labels with `COALESCE`, preserving reviewed
values. Topic absence alone does not trigger enrichment. Predictions resolve
titles to topic records only for voices with a null topic ID, preserving existing
assignments and outliers. Escalation is a single idempotent update.

Topic counts, rankings, category/topic charts, topic sentiment charts, and
companion rankings use assigned topics only, excluding null IDs and `-1`.
Overall volume, sentiment, urgency, and escalation metrics retain those voices.
Issue Analytics provides separate assignment counts and empty states for filters
matching only Outlier or Unlabeled records. Selecting all categories leaves
uncategorized voices visible; an explicit category subset filters by its flags.

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
Each card has an inline status selectbox. Its callback commits through
`VoiceRepository.set_escalation_status()` before Streamlit reruns and reloads the
snapshot, allowing Open, Escalated, or Resolved. Resolved voices never enter
urgent Open metrics. The feed sorts by urgency, then Escalated/Open/Resolved,
then newest timestamp, preserving or clamping pagination after edits.

`services/issues.py` supplies negative-only diagnostics independently of UI:
topic catalogs keyed by IDs, top-five named topic tiles per category without tail aggregation,
daily/weekly/monthly zero-filled bins, volume/peak topic selection, and channel
and unit summaries. `pages/issues.py` keeps the existing filter/alert widgets and
applies negative sentiment scope after filtering. Global analytics remain
independent of the deep-dive selection. Treemap events and a selectbox share a
durable topic ID; consumed selection signatures prevent old chart events from
overriding manual choices on unrelated reruns. Only real topic leaves can select
a topic. The None-free `all` selectbox option disables deep-dive and restores
the full negative feed. Filter changes validate selected IDs and fall back to
the highest-volume available topic. Duplicate titles display IDs to disambiguate
them, and every unique topic remains available even when not in the treemap.
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
- `test_initialization.py`: source validation, calendar-month escalation boundary,
  null topics, outliers, backups, overwrite rollback, legacy migration, and safe startup.
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
