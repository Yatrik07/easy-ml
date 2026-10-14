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

**Persistence contract.** Data root from `EASYML_DATA_DIR` or platform app-data dir (`config/paths.py`). Each run folder holds `data.csv`, `config_json.json`, `column_data.json` (incl. `column_dtypes` used to coerce string form input at inference), models (`<Class>.pkl`, CatBoost `<Class>.cbm` — older runs `.pkl.cbm`, CV `.pth`) plus `preprocessing_pipeline.pkl` (tabular) or vectorizer/`fe_config.json` (NLP). New runs also write `evaluation.json` (result chart data, `core/evaluation_charts.py`; a chart error is recorded under `skipped` and never fails a run). Class labels are decoded only through `utils/labels.decoder_from_config` (`target_labels` keeps their type). New DB columns go in `db_runs_columns_config`; `create_database_table` adds missing ones to older databases. `get_record_by_id` raises `RunNotFoundError` (→ 404). Tabular inference, SHAP and export all load artifacts through `core/artifacts.py` (model-name validation, file naming, input coercion) — change saving and loading together there.

**Errors to HTTP.** Use `route_helpers.error_response(exc)` (RunNotFound/FileNotFound→404, PermissionError→403, ValueError/KeyError→400); never return error dicts with status 200.

**Frontend.** `easyml/unified_frontend/dist` is the source: no-build Vue 3 as native ES modules (`static/js/{lib,store,components,views}`, entry `main.js` → `App.js`), CSS split into `static/css/{tokens,base,components,views}.css` with light/dark tokens (`<html data-theme>`). Templates are template strings: never put `${` or backticks inside them. Stores are reactive singletons: never destructure; the training socket lives in `store/training.js`. No inline styles or hard-coded colors; no CDN links (vendored in `static/vendor`). `EASYML_FRONTEND_DEBUG=1` serves Vue's dev build for template warnings. Charts use `lib/charts.js` (`useChart`: build after mount, rebuild on data/theme change, destroy on unmount) with `--chart-*` tokens. `tests/test_frontend_assets.py` checks every import/asset resolves. `/settings/capabilities`, `/runs` summaries (`utils/run_summary.py`) and `/get_run_data` feed the UI; `project_info.py` holds About-page facts and the `REPO_PUBLIC` switch.

**Using models outside the studio.** `easyml.load_model(run_id, model_name=None)` (lazy wrapper → `easyml/predictor.py`, defaults to the best model via `run_summary.rank_models`) and the CLI `easy-ml runs` / `easy-ml predict` (`easyml/cli.py`) reuse the inference controllers' batch methods (`predict_frame`, `predict_texts`, `predict_images`). `easyml.retrain` (`retraining.py`: replays a run's saved config on new data, keeps its input schema, records `parent_run_id`/`origin`) and `easyml.export_run`/`import_run` (`run_archive.py`: zip with manifest + sha256; imports are validated and need explicit trust) back the `retrain`/`export`/`import` CLI commands and the `routes_runs.py` endpoints. The user-facing reference is `docs/PYTHON_API.md` (linked from Home and About via `project_info.python_guide()`); update it when `MLTrainer`, `Predictor` or the CLI change.
