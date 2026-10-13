"""
Centralized path configuration for Easy-ML.

All runtime data (database, runs, logs, uploads) is stored in a
platform-standard application-data directory — NOT inside the package
directory.  This ensures data persists across library reinstalls.

Override the location by setting the ``EASYML_DATA_DIR`` environment
variable (used by the Docker image).

Directory layout
----------------
<DATA_DIR>/
├── database/   ← SQLite DB files
├── runs/       ← trained-model artefacts per run
├── logs/       ← application log files
├── uploads/    ← uploaded CSV / image datasets
└── eda/        ← EDA HTML reports
"""

import os
import platform

# ── resolve DATA_DIR ────────────────────────────────────────────────
_env = os.environ.get("EASYML_DATA_DIR")

if _env:
    DATA_DIR = os.path.abspath(_env)
elif platform.system() == "Windows":
    DATA_DIR = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "easy-ml")
elif platform.system() == "Darwin":
    DATA_DIR = os.path.join(os.path.expanduser("~"), "Library", "Application Support", "easy-ml")
else:  # Linux / other Unix
    DATA_DIR = os.path.join(os.environ.get("XDG_DATA_HOME", os.path.join(os.path.expanduser("~"), ".local", "share")), "easy-ml")

# ── sub-directories ─────────────────────────────────────────────────
DB_DIR      = os.path.join(DATA_DIR, "database")
RUNS_DIR    = os.path.join(DATA_DIR, "runs")
LOGS_DIR    = os.path.join(DATA_DIR, "logs")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
EDA_DIR     = os.path.join(DATA_DIR, "eda")

# ── create them eagerly so the rest of the app never has to ─────────
# A read-only home directory must not break ``import easyml``; uploads and
# logs fall back to the temp directory when their folder is not writable.
for _d in (DB_DIR, RUNS_DIR, LOGS_DIR, UPLOADS_DIR, EDA_DIR):
    try:
        os.makedirs(_d, exist_ok=True)
    except OSError:
        pass

# ── package root (for locating bundled static/frontend files) ───────
PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
