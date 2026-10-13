# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Easy-ML Studio is a no-code ML studio: a Flask + Socket.IO backend (import package `easyml`, PyPI distribution `easy-ml-studio`, CLI `easy-ml`) serving a bundled browser UI, covering tabular classification/regression, NLP text classification, and CV image classification. `docs/DEVELOPER_GUIDE.md` walks through every module and endpoint — read the relevant section before changing a subsystem.

## Commands

```bash
pip install -e ".[all,dev]"      # dev install (extras: boost, nlp, cv, eda, all, dev)
easy-ml --no-browser             # run on http://127.0.0.1:5010 (also: python -m easyml)
pytest                           # full suite (tests/; conftest sets a temp EASYML_DATA_DIR)
pytest tests/test_training_flows.py::test_regression_flow    # single test
flake8 easyml tests --select=E9,F63,F7,F82,F401,F811,F841    # CI lint gate
python -m build && twine check --strict dist/*               # packaging check (CI does the same)
docker compose up --build        # Dockerfile targets: base, ml, nlp, cv, full
```

Tests needing optional extras skip themselves; CI also runs the suite on a core-only install to catch optional imports leaking into core.

## Architecture

**Core vs extras.** Core = Flask, scikit-learn, pandas (tabular + TF-IDF NLP). Boosters (`[boost]`), spaCy/sentence-transformers (`[nlp]`), torch (`[cv]`), shap/ydata-profiling (`[eda]`) are optional and must stay lazily imported. `config/GLOBAL_VARS.py` registers booster models only when importable; `easyml/capabilities.py` (`find_spec`, never imports heavy libs) drives `/settings/capabilities`, 503s in routes, and greyed-out UI features. Never add a module-level import of an optional library to a core path.

**App wiring.** `app_factory.create_app()` builds the app: JSON provider, Origin-check `before_request` that 403s cross-site POST/DELETE (no CORS by default; `EASYML_ALLOWED_ORIGINS` adds origins), same-origin Socket.IO, blueprints. `app.py` is only the CLI (binds 127.0.0.1 by default). Library code must not touch root logging or global warning filters — that belongs in the CLI.

**Training flow.** Routes validate input (dataset paths must resolve inside the data dir via `route_helpers.resolve_data_file`; run folders via `create_run_folder`, which sanitizes names), then `route_helpers.start_training_job` runs `MLTrainer` (tabular/NLP) or `CVController` (CV bypasses MLTrainer) in a background task. Controllers emit the success "done" event themselves and *raise* on failure/cancel; `start_training_job` emits the single final error/cancelled event. Each background run is a job (`utils/cancellation.TrainingJob`): start endpoints return its `job_id`, `emit_progress` sends only to the job's Socket.IO room (clients `join_run` and get the job's history replayed, deduped by `seq`), and `/stop_training {job_id}` sets that job's own cancel event, checked via `check_cancelled()`/`is_cancelled()` between models/stages/epochs/batches and between grid-search candidate batches (`core/cancellable_search.py`) (`InterruptedError`). Code running outside a job (e.g. `MLTrainer.run()` in a script) is never cancelled and emits nothing. On failure/cancel the run folder is removed unless a DB row points at it.

**Mutable state carriers.** `ML_RequestBody` (`config/GLOBAL_VARS.py`), `NLP_RequestBody` (`nlp/nlp_config.py`), `CV_RequestBody` (`cv/cv_config.py`) are mutated stage by stage; errors recorded via `status_code`/`error_message`/`error_traceback`. Mutable defaults must use `default_factory` (covered by tests).

**Persistence contract.** Data root from `EASYML_DATA_DIR` or platform app-data dir (`config/paths.py`). Each run folder holds `data.csv`, `config_json.json`, `column_data.json` (incl. `column_dtypes` used to coerce string form input at inference), models (`<Class>.pkl`, CatBoost `<Class>.cbm` — older runs `.pkl.cbm`, CV `.pth`) plus `preprocessing_pipeline.pkl` (tabular) or vectorizer/`fe_config.json` (NLP). `get_record_by_id` raises `RunNotFoundError` (→ 404). Tabular inference, SHAP and export all load artifacts through `core/artifacts.py` (model-name validation, file naming, input coercion) — change saving and loading together there.

**Errors to HTTP.** Use `route_helpers.error_response(exc)` (RunNotFound/FileNotFound→404, PermissionError→403, ValueError/KeyError→400); never return error dicts with status 200.

**Frontend.** `easyml/unified_frontend/dist/index.html` is a hand-written, no-build Vue 3 single page (it *is* the source). Libraries and fonts are vendored in `dist/static/vendor` — do not reintroduce CDN links. `/settings/capabilities` and `/get_run_data` (`column_details`) feed UI gating and inference form widgets.
