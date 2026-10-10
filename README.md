# BNIVoice360

BNIVoice360 is a beginner-friendly Streamlit dashboard for exploring customer feedback (Voice of Customer). It stores data in SQLite and provides issue analytics, a searchable voice feed, CSV ingestion, and demo enrichment tools.

This project is a local demo: the role selector is not authentication, crawling is simulated, and classification uses local rules instead of paid APIs.

## 1. Requirements

- Python 3.10 or newer
- Git installed on your system
- Windows PowerShell (the commands below are written for Windows)
- Internet access only while cloning the repository and installing Python packages

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

On first startup, the app creates `bnivoice360.db` and inserts 96 demo records only when the database table is empty. This database is intentionally ignored by Git because it is local runtime data.

To try a separate database without touching the default one:

```powershell
$env:BNIVOICE360_DB_PATH = "$PWD\demo.db"
streamlit run app.py
```

## 3. What to click

Use the role selector in the sidebar:

- **Manager / Executive**: overview, issue analytics, voice feed, graph, and companion pages.
- **Data Admin**: everything above plus CSV upload, source selection, demo crawler, and enrichment actions.

The **Issue Analytics & Voice Feed** page displays compact cards. Use search, filters, and page buttons to find feedback. Clicking **Escalate** saves the change to SQLite immediately and refreshes the page.

For CSV ingestion, the required columns are `content` and `date_posted`. The selected source in the sidebar supplies `source`; mixed-source files may include their own `source` column. Dates should be parseable by pandas, and sentiment must be one of the values accepted by the validation message.

## 4. Project map

```text
app.py                     # Small entry point used by Streamlit
bnivoice360/application.py   # Starts the app and registers pages
bnivoice360/container.py     # Builds shared database/services
bnivoice360/config.py        # Settings and business constants
bnivoice360/domain.py        # Typed domain models
bnivoice360/data/            # SQLite schema, queries, and demo seed data
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

## Troubleshooting

- **`ModuleNotFoundError`**: activate `.venv`, then run `python -m pip install -r requirements.txt` again.
- **Port 8501 is busy**: run `streamlit run app.py --server.port 8502` and open <http://localhost:8502>.
- **The page looks stale**: stop Streamlit with `Ctrl+C` and start it again.
- **You need fresh demo data**: point `BNIVOICE360_DB_PATH` to a new filename rather than deleting the existing database.

For architecture details and extension recipes, read [docs/architecture.md](docs/architecture.md). For team conventions, read [CONTRIBUTING.md](CONTRIBUTING.md).