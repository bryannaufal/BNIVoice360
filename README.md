# BNIVoice360

BNIVoice360 is a beginner-friendly Streamlit dashboard for exploring customer feedback (Voice of Customer). It stores data in SQLite and provides issue analytics, a searchable voice feed, CSV ingestion, and demo enrichment tools.

This project is a local demo: the role selector is not authentication, crawling is simulated, and classification uses local rules instead of paid APIs.

## 1. Requirements

- Python 3.10 or newer
- Git installed on your system
- Windows PowerShell (the commands below are written for Windows)
- Internet access for installing packages and the first Google CSV initialization

## 2. Clone, install and run

Open PowerShell, clone the repository from your Git provider, and navigate into the project folder:

```powershell
git clone <repository-url>
cd bnivoice360
```

Then, set up the virtual environment and run the app:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

If PowerShell refuses to activate the environment, allow it only for the current terminal window and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Streamlit opens the dashboard at <http://localhost:8501>. Keep the terminal running while using the app. Press `Ctrl+C` in that terminal to stop it.

On first startup, the app creates `bnivoice360.db` and initializes an empty database from the configured Google CSV exports. Existing data is preserved. This database is intentionally ignored by Git because it contains local runtime data.

### Initialize or overwrite the database

The database has two tables: `suara_nasabah` and `topic_detail`. Voice records have a nullable foreign key to topic details. No assignment means Unlabeled (`NULL`); topic ID `-1` means Outlier. Outlier and Unlabeled voices remain in overall metrics and the feed, but are excluded from topic counts, rankings, and topic charts. Issue Analytics displays their counts separately and allows filtering each group.

The default sources are the `df_Clean` Google Sheet, topic descriptions, and document-to-topic map supplied for this project. Google files must be accessible as CSV downloads. Local CSV paths can also be supplied through `BNIVOICE360_VOICE_CSV`, `BNIVOICE360_TOPIC_CSV`, and `BNIVOICE360_MAPPING_CSV`. `Internal_CRM` becomes `Laporan Nasabah`; `Netral` becomes `Neutral`. Author identifiers are preserved. List-like topic fields are retained as CSV text.

Choose whether to replace existing data with this configuration:

```powershell
# 0 = preserve existing voices (default); 1 = replace with the new initialization
$env:BNIVOICE360_INIT_OVERWRITE = "0"
python -m bnivoice360.data.seed
```

For a fresh initialization, stop Streamlit first, then run:

```powershell
$env:BNIVOICE360_INIT_OVERWRITE = "1"
python -m bnivoice360.data.seed
$env:BNIVOICE360_INIT_OVERWRITE = "0"
streamlit run app.py
```

`--overwrite` and `--preserve` override the environment setting for one command. Overwrite validates all sources first, backs up the existing database to `bnivoice360.backup-<timestamp>.db`, then replaces both tables' data in one transaction. A failed import rolls back. Preserve mode skips downloading when voices already exist. The overwrite setting is used **only by this command**, never by Streamlit startup, so reruns cannot reset data. Legacy single-table databases are migrated with a backup while retaining their records and escalation values.

For offline startup or a deliberately empty database, set `$env:BNIVOICE360_AUTO_INITIALIZE = "0"`. Set it to `"1"` (default) to initialize empty databases automatically. Disable automatic initialization after intentionally removing all voices if you want the database to remain empty.

On first startup, the app downloads `topic_id_embedding_mapping.parquet` and `data_id_embedding_mapping.parquet` from the supplied Google Drive sources into `.embeddings/`. Later launches reuse these files; vectors are stored for future use and are not loaded into memory or inserted into SQLite. Downloads use temporary files and validate the Parquet envelope before publishing them to the cache. A failed download displays a warning and retries on the next startup/rerun; the existing pages remain available.

Set `BNIVOICE360_EMBEDDING_DIR` to choose another cache directory, or `BNIVOICE360_DOWNLOAD_EMBEDDINGS=0` for offline startup. Source URLs can be overridden with `BNIVOICE360_TOPIC_EMBEDDING_URL` and `BNIVOICE360_DATA_EMBEDDING_URL`. The Google Drive files must allow downloads without signing in. To refresh an artifact, remove its cached file and restart the app.

Seeding sets `escalation_status = "Escalated"` for records before one calendar month prior to initialization in Jakarta time, and `"Open"` at or after the cutoff. Undated source rows are retained with a null posting date and status `"Escalated"`; the command reports their count. Nonempty invalid dates fail validation. Use `--as-of "2026-10-11"` to reproduce a cutoff of September 11, 2026. Later startups preserve status changes, including `"Resolved"`. Date filters exclude undated voices; overall metrics still include them.

The status field accepts exactly `Open`, `Escalated`, or `Resolved`, defaults to `Open`, and is enforced by SQLite. Existing databases are backed up and migrated automatically: the former binary field maps `0` to `Open` and `1` to `Escalated`, preserving topic links and author information.

To try a separate database without touching the default one:

```powershell
$env:BNIVOICE360_DB_PATH = "$PWD\demo.db"
streamlit run app.py
```

## 3. What to click

Use the role selector in the sidebar:

- **Manager / Executive**: overview, issue analytics, voice feed, graph, and companion pages.
- **Data Admin**: everything above plus CSV upload, source selection, demo crawler, and enrichment actions.

The **Issue Analytics & Voice Feed** page focuses on **Negative** feedback. Its four KPIs show assigned negative issue volume, urgent Open cases awaiting triage, urgent Escalated cases under investigation, and the largest assigned negative topic. The existing filters and alert remain available; a filter excluding Negative produces an empty issue view. Outlier and Unlabeled counts appear separately and never drive topic rankings.

The category treemap shows only the five largest named negative topics per unit, without an aggregated long-tail tile. All remaining topics stay accessible through the deep-dive dropdown and Voice Feed. The adjacent global trend plots the union of four topics by total volume and two by peak volume in the active time bins, with no aggregated Lainnya line. Both charts are 400px tall. Inclusive date ranges use daily bins for up to 30 days, weekly bins for 31–90 days, and monthly bins above 90 days. Weeks start on Monday and months start on the first; partial edge intervals include only records within the selected date range. Missing intervals display zero counts.

Click a treemap topic or use **Pilih Topik untuk Deep-Dive** to inspect its volume, status counts, dominant channel, unit coverage, isolated trend, source breakdown, and department overlap. The largest topic is selected initially; selection persists across navigation and falls back to the largest available topic if filtering removes it. Category nodes do not select a topic. Choose **Semua isu (tanpa deep-dive)** to see the globally filtered negative feed, including Outlier and Unlabeled records. Channel names reflect stored source data. Unit involvement is multi-label, so percentages can exceed 100% in total.

The **Suara Nasabah & Action Center** feed displays full escaped customer quotes, exact Jakarta timestamps, topic/source/unit badges, urgency, and an inline **Open / Escalated / Resolved** status dropdown. Status changes save immediately and re-sort the feed: urgent voices first, then Escalated, Open, Resolved within each urgency group, then newest submissions. Status and topic filters may remove a changed voice from the current view; pagination is preserved or clamped as needed. Only urgent Open voices contribute to the alert and the Home **Urgent Open** metric. Resolved records remain in historical analytics and the feed.

Native treemap selection requires Streamlit 1.55 or newer, reflected in `requirements.txt` ([Streamlit release notes](https://docs.streamlit.io/develop/quick-reference/release-notes/2026)).

For CSV ingestion, the required columns are `content` and `date_posted`. The selected source in the sidebar supplies `source`; mixed-source files may include their own `source` column. Dates should be parseable by pandas, and sentiment must be one of the values accepted by the validation message. Optional `escalation_status` accepts `Open`, `Escalated`, or `Resolved`; blank or missing values default to `Open`. Unknown statuses are rejected during validation. Enrichment preserves status. Companion queries support Open, Escalated, Resolved, and unresolved (Open plus Escalated).

## 4. Project map

```text
app.py                     # Small entry point used by Streamlit
bnivoice360/application.py   # Starts the app and registers pages
bnivoice360/container.py     # Builds shared database/services
bnivoice360/config.py        # Settings and business constants
bnivoice360/domain.py        # Typed domain models
bnivoice360/data/            # SQLite schema, queries, and CSV initialization
bnivoice360/services/        # Analytics, ingestion, crawler, and companion logic
bnivoice360/pages/           # One rendering module per dashboard page
bnivoice360/ui/              # Reusable widgets, charts, filters, and CSS
tests/                       # Automated tests
docs/architecture.md        # Design decisions and extension examples
CONTRIBUTING.md             # Team workflow and coding conventions
```

The application follows a simple flow:

`app.py` → application/page registry → page UI → services → SQLite repository.

Pages should render screens, services should contain business rules, and the repository should contain database queries. Keeping those responsibilities separate makes changes safer for a team.

## 5. Run quality checks

Install development tools, then run these commands before opening a pull request:

```powershell
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m pytest
```

The tests use temporary databases, so they do not modify your local `bnivoice360.db`.

## 6. Basic Git workflow

```powershell
git status
git add .
git commit -m "Describe the change"
git log --oneline -5
```

Before committing, check that local databases, `.venv`, caches, and secrets are not listed by `git status`. Never commit passwords or API keys; use a local `.env` file (already ignored) for development-only settings.

## Conversational complaint investigation

### Graph Analytics: related complaint network

Open **Graph Analytics**, choose a named topic, period, channels, statuses and unit labels, then click
**Bangun / perbarui jaringan**. The default scope is Negative, Open plus Escalated, within the last
30 days ending at the latest posting date in the dataset. All-time includes undated complaints.
To explore across topics, choose **Semua topik** and explicitly enable cross-topic analysis;
this also allows Outlier/Unlabeled. A named topic remains restricted to that topic.

The network uses existing 384-dimensional voice embeddings with cosine similarity. At most 200
eligible records enter pairwise computation; the default is 120. When capped, urgent Open records
come first, followed by other Open, Escalated and Resolved, newest first within each priority.
Missing embeddings and records omitted by the cap are reported separately. Edges are the union
of each node's strongest threshold-qualified neighbors (default five). A node may have more than
five edges when other nodes select it. Groups are connected components, not verified incidents,
duplicates, or shared root causes. Connectivity can be transitive: not every pair in a group has
a direct relationship. Isolated complaints remain visible. Group statistics describe displayed
records only; partial sampling and neighbor limits can change group membership.

Node fill shows channel, border shows status, and larger nodes indicate the stored urgent label.
Clicking a node opens evidence without selecting it for escalation. Group summaries rank urgent
Open count, Open count, then oldest Open posting date. Posting age is not time in status or SLA.
Department flags indicate relevance, not assigned ownership; channel coverage is not a unique
customer count. **Jelaskan kelompok** provides local statistics, or a validated Groq explanation
using at most five screened examples when configured. Full text, authors and ticket identifiers
are not sent as provider metadata; heuristic text screening still does not guarantee PII removal.
Explanation calls are explicitly triggered and metadata-audited.

Select individual Open ticket checkboxes and click **Lanjutkan ke Companion** to prepare a draft.
Nothing is preselected. Handoff supports at most 51 tickets, revalidates the original displayed
row fingerprints, and invalidates any previous Companion draft/approval. It does not change status.
Companion still requires target-team details, draft review and explicit approval. A read-only
graph explanation leaves Companion state untouched. The embedded Voice Feed displays live statuses;
changes require rebuilding the graph before handoff. The graph remains a session snapshot until
explicit refresh, and the displayed scope reflects the built filters rather than unsubmitted edits.

### Data Admin: Companion Evaluation

Select **Data Admin → Companion Evaluation** to inspect Companion telemetry and evaluate its behavior.
Access is checked in navigation, the page, and evaluation service methods using the existing demo role
selector. The page has **Overview**, **Agent Logs**, **Testing & Evaluation**, and **Run Comparison**.

Overview uses the selected Jakarta date range and separates recorded interaction outcomes, p95 response
time, provider time, tool time, tokens, and fallback/validation events. Missing metrics display `—`.
Tokens are counted from provider-call events only, not counted again from analytical run summaries.
Operational events remain in `<database-stem>.agent-audit.db`. New chat, draft, revision and commit
interactions receive IDs linking their events. Historical events remain readable without invented IDs
or durations. Logs support date/event/status/exact-session filters, pagination, event details,
interaction timelines and a CSV download of the current page. Operational prompts/answers are not
retrospectively reconstructed or stored in these metadata logs.

The evaluation suite uses six synthetic tickets and a fresh temporary database for every scenario.
It covers existence/display intents, status/category/channel filters, arbitrary topics, follow-ups,
period clarification, empty results, selection, full originals in drafts, revision cancellation,
approval, guardrails, and tool-grounded counts. No operational tickets or existing conversation state
are used or mutated. Opening or refreshing the page never starts evaluation.

- **Local** runs deterministic search/workflow checks and basic drafts without a provider.
- **Mock** tests model-output contracts and grounding with deterministic simulated responses; it does
  not measure live model quality.
- **Live Groq** uses the configured model/key on synthetic data only, and requires an explicit click.
  Select at most three scenarios per run. Missing keys prevent execution. Provider failures/fallbacks
  are visible in results; draft scenarios require validated AI prose to pass in Mock/Live modes.

Evaluation runs, checks, synthetic transcripts/drafts/traces and manual reviews persist in the separate
`<database-stem>.companion-evaluation.db`. Each run records mode, model, dataset date/digest, source and
prompt-module digests, suite version and configuration. Detail views show expected/actual values and
each failed check, with JSON export. Human ratings are separate from automated pass/fail checks and
do not rewrite them. Interrupted runs retain saved partial results. Comparison aligns scenarios and
flags regressions; mismatched modes, datasets, suites or scenario coverage are explicitly labelled.
The page's scenario pass rate is not a substitute for pytest results or semantic human evaluation.

### Search, investigation and escalation

Open **BNIVoice360 Companion** and start with:

> Apakah ada issue terkait QRIS akhir-akhir ini?

For vague dates such as **akhir-akhir ini**, Companion asks for a concrete period. Reply **1 minggu terakhir**, **30 hari terakhir**,
**semua periode**, or two ISO dates such as **2026-10-01 sampai 2026-10-11**. Relative dates use Jakarta
time: seven days includes today and six preceding calendar days; **minggu ini** starts on Monday.
Searches without a stated period default to the last **30 days**, with the exact scope displayed.
**Apakah ada ...?** checks the database, shows the count/status breakdown, and offers **Tampilkan issue**.
**Tampilkan / lihat / cari ...** immediately opens the list. Reply **ya, tampilkan** to accept an offer;
the database is rechecked before showing records. This only opens results, never approves escalation.
Search matches literal topic phrases in complaint text or topic titles; topics are not limited to QRIS/wondr.
Negative sentiment is the default; explicit positive/neutral/all-sentiment searches are supported.
Combine category, channel, status and urgency filters, even without a keyword: **Tampilkan issue IT yang
masih Escalated**. **Mendesak / perlu segera ditindaklanjuti / urgent** filter the stored urgent label,
not a verified incident severity. **Yang masih Open saja** refines the active search; **semua kanal**,
**semua kategori**, **semua urgensi**, **semua topik** clear the respective restriction. A new topic starts
a new scope. The actual scope is displayed below the conversation.
Chat text is not embedded. Existing count/top-topic questions remain available and calculated locally.
Undated rows are only included in explicit all-time searches.

Companion also answers broader analytical questions through Groq tool calling:

- **Keluhan apa yang paling meningkat minggu ini dibanding periode sebelumnya?**
- **Apakah keluhan QRIS lebih banyak muncul di X atau PlayStore?**
- **Topik mana yang punya backlog Open terbanyak dan paling lama?**
- **Masalah spesifik apa yang dilaporkan dalam keluhan transfer?**
- **Buat briefing keluhan minggu ini untuk tim IT.**

The model chooses from read-only tools: `dataset_coverage`, `volume_breakdown`, `compare_periods`,
`daily_trend`, `backlog_age`, `complaint_examples`, and `similar_complaints`. It can use several tools
for one question, then write a natural answer grounded in their results. **Tools yang digunakan**
shows executed tools and their outcomes; **Sumber dan hasil analisis** exposes the actual data behind
citations such as `[T1]`. Date scope and filters are displayed beside the answer. New analysis defaults
to the last seven days; backlog without dates uses all posting dates. A follow-up such as **Bagaimana
di PlayStore?** can reuse the previous analytical scope. Explicit search commands use the search
workflow and its 30-day default, asking about dates only when vague or unsupported.

Comparisons use immediately preceding periods with equal numbers of days. A zero baseline has no
growth percentage. Outlier/Unlabeled stay in totals but are excluded from named-topic rankings.
Categories can overlap. Backlog age is measured since posting, **not** time in status, resolution time,
or SLA compliance. Examples are recent screened samples, not exhaustive subtopic frequencies.
Similarity uses existing voice vectors, Open candidates and the anchor's named topic; it never embeds
chat or silently broadens an Outlier/Unlabeled query. Missing data/tools or unsupported questions
(such as actual losses, verified technical causes or SOP without a source) are refused. Excerpts and
labels are screened before provider use. Quoted passages must match a cited source; numeric values
are checked against cited tool data. These checks do not guarantee that every interpretation is correct.

To act on an analytical example, explicitly select **Pilih T2.E1** in the analysis panel or chat.
Selection binds the displayed row version; concurrent changes require a fresh analysis. Analytical
questions do not select tickets, create drafts, invalidate existing approval, or change status.

Results show ID, date, source, status and excerpt: ten per page, up to 200 results. The total count
includes matches beyond that limit. **Nothing is selected automatically.** Click **Pilih keluhan**,
write **pilih nomor 2**, or explicitly refer to a visible ID. **Yang saldo terpotong** selects a ticket
only if that phrase uniquely identifies a visible result. Ambiguous references require selection.
Selected IDs remain visible and survive navigation within the browser session. A new search or cancel
clears selection and pending drafts.

With exactly one ticket selected, ask **apakah ada keluhan serupa**. Search uses that ticket's existing
384-dimensional vector and Open negative complaints in the same period/channels. Same-topic search
is the default; Outlier/Unlabeled requires explicit cross-topic confirmation. Missing vectors are
reported; a failed search never masquerades as zero matches. Similarity does not prove a shared
cause. Only the anchor remains selected; choose additional candidates explicitly. Configurable
defaults: `BNIVOICE360_AGENT_SIMILARITY` (0.80, provisional), `BNIVOICE360_AGENT_LIMIT` (20, maximum 50).

Ask **saya ingin escalate keluhan tersebut**. Companion requires a selection, destination unit
(IT/Operasional/Bisnis), and request written by the petugas. Enter the form or use chat:

> IT; permintaan: periksa status transaksi; pemeriksaan: petugas sudah memeriksa mutasi

The examination note is optional and omitted when absent. Drafts contain the **full original complaint
for every selected ticket**, including actual IDs, source and posting date. Originals are assembled
locally, never rewritten or truncated by the model. Groq creates an evidence-linked summary and
proposed verification steps, separately labelled as AI suggestions. The application validates citations,
quote/numeric grounding, selected-ticket coverage and unsupported claims. Technical causes and actual
impact remain unconfirmed; the petugas' request and examination remain verbatim. You can download the
document, revise through the form, or ask **revisi draft agar lebih singkat dan formal**. Each revision
cancels the old preview and requires fresh approval. Drafting does not send messages or change status.
Rejected/failed AI output uses a basic local summary while preserving every full original complaint.
This update does not add personal-information masking; existing remote screening remains in place.

To change status, check approval and click **Konfirmasi eskalasi N tiket**. Only Open tickets can be
committed; Escalated/Resolved tickets can produce a document without a status action. Approval binds
exact IDs, complete row snapshots, session, role and draft/input context, and expires after 15 minutes.
Status writes and the success audit commit together using `<database-stem>.agent-audit.db`; the
business database remains at two tables. DELETE rollback journals are required; WAL is rejected.
Audit failure or concurrent edits abort the whole batch. Repeated submission does not repeat writes.

Companion uses the existing Groq key automatically. Supported secret layouts: `GROQ_API_KEY = "..."`,
`GROQ = "..."`, or `[groq]` with `api_key = "..."`. Environment `GROQ_API_KEY` takes precedence.
Default model: `openai/gpt-oss-20b`. Keys are never saved in history or session state. For local-only
operation set `$env:BNIVOICE360_COMPANION_REMOTE_ENABLED = "0"` before starting Streamlit; this is
separate from the Issue Analytics batch panel's remote-AI setting. Without Groq, common commands,
period buttons, selection, basic drafts and approval remain usable. For unrecognized local terms,
use quotes, for example **cari keluhan "pembayaran digital" 7 hari terakhir**.

Intent extraction uses strict JSON schema and local validation for allowed actions, explicit dates,
selection evidence and user-provided fields. There is no model write action or arbitrary SQL/shell/web
tool. Analytical turns allow at most six sequential planning rounds/tool calls, a 60-second overall
deadline and 20,000 reported tokens. Natural answers use a separate strict JSON response; an invalid
format or unsupported number may be regenerated once within the same budget. Rejected prose is
never displayed or copied into retry prompts. Workflow commands can use an intent call, quote
extraction and a grounded natural reply; the application retains authority over actions. Requests
have at most two HTTP attempts, 1,800 completion tokens (900 for tool selection), a 15-second socket
timeout and a 40-second per-response deadline. Tool calling and strict final responses are separate,
as required by [Groq's tool-use documentation](https://console.groq.com/docs/tool-use/overview).
Only the current question, minimal workflow flags or previous analytical question/scope, and
screened tool results are sent; full chat history is not sent. History is capped at 41 messages and
is session-local. Common statistics also run through these tools in local mode.

The separate audit database logs `analysis_started`, tool start/finish, output rejection/retry,
planner corrections and `analysis_finished`, with run/session IDs, digests, result counts, duration,
reported tokens, model, outcome and safe reason codes. Prompt text, complaint content, generated
answers/drafts, authors and API keys are excluded. Provider/tool failures produce a safe fallback;
successful screened tool results remain inspectable. Audit failure prevents a success response.
The role selector is not authentication.

**PII masking remains deferred.** Author metadata and real ticket IDs are excluded from Groq payloads.
Common structured PII/injection checks are heuristic; names and addresses may still be present.
Exact citations prevent invented quotes, not incorrect real-world claims. Review drafts before use.
The boundaries follow [Groq structured output documentation](https://console.groq.com/docs/structured-outputs)
and [OWASP agent security guidance](https://cheatsheetseries.owasp.org/cheatsheets/AI_Agent_Security_Cheat_Sheet.html).

## Troubleshooting

- **`ModuleNotFoundError`**: activate `.venv`, then run `python -m pip install -r requirements.txt` again.
- **Port 8501 is busy**: run `streamlit run app.py --server.port 8502` and open <http://localhost:8502>.
- **The page looks stale**: stop Streamlit with `Ctrl+C` and start it again.
- **You need fresh source data**: stop Streamlit and run `python -m bnivoice360.data.seed --overwrite`; the command creates a database backup.

For architecture details and extension recipes, read [docs/architecture.md](docs/architecture.md). For team conventions, read [CONTRIBUTING.md](CONTRIBUTING.md).
