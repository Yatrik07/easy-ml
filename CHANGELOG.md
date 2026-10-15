# Changelog

All notable changes to Easy-ML Studio (`pip install easy-ml-studio`). Versions follow
[semantic versioning](https://semver.org/); existing runs and data keep working across upgrades.

## 0.9.2 (2026-10-15)

Housekeeping and fixes.

- **Docker**: the images now include `waitress`, so `easy-ml serve` inside a container runs on a real
  web server instead of Flask's development server. A test keeps `requirements-core.txt` in sync with
  the package's dependencies. `docker-compose.yml` shows how to add a schedule worker.
- **Logs no longer grow forever**: each log file is rotated at 5 MB, keeping 3 old copies
  (`EASYML_LOG_MAX_MB`, `EASYML_LOG_BACKUPS`). `easy-ml serve` and `easy-ml schedule` write their own
  log files, and routine per-request messages are no longer logged.
- **Old uploads are cleaned up**: Settings shows how much space uploads use and deletes the ones older
  than a day (every run keeps its own copy of its data); uploads older than 30 days are removed when
  the studio starts. Only files Easy-ML created are ever deleted.
- **Clearer errors for broken optional libraries**: when an extra (sentence-transformers, spaCy, PyTorch,
  SHAP) is installed but fails to load, the message shows the real error and suggests reinstalling it,
  instead of saying it is not installed. Image routes answer such failures with a clear message.

## 0.9.1 (2026-10-15)

- **Try a real row** on the prediction page: fills the form (or the text box) with a random row of the
  run's data, preferably from its test set, predicts it and shows the real answer next to the prediction.
- **Predict** is disabled while every input is empty.
- **Readable explanations**: a full-width chart with one bar per input column and the value used, and a
  one-line summary; SHAP's detailed plot opens in a large window with a PNG download.

## 0.9.0 (2026-10-15)

- **Scheduled retraining**: retrain a run when new data arrives or at set times; the new model is used
  only if it scores better on data neither model trained on. Runs in the studio, with
  `easy-ml schedule worker`, or `easy-ml schedule run-due` (Task Scheduler / cron).
- `easy-ml serve --schedule NAME` follows a schedule's current model without a restart.
- New Schedules page, "Schedule retraining" in Run history, "In use by" / "Not promoted" badges.
- Python: `easyml.schedules`; `easyml.prep` and `easyml.schedules` load on first use.

## 0.8.0

- **Data prep**: clean a table step by step with a live preview (16 steps), then train on it, download
  it, or save the steps as a reusable recipe (`easyml.prep`, `easy-ml prep`).
- Runs trained on cleaned data remember their recipe; retraining cleans new files the same way.

## 0.7.0

- **`easy-ml serve`**: any trained run (tables, text, images) as a local REST API on waitress, with a
  help page, optional API key, CORS and request size limit. "Serve as API" tab on the prediction page.

## 0.6.2

- "Export as API" explains what the downloaded package is and how to run it.

## 0.6.1 (2026-10-14)

- Faster start-up, run listing, predictions (models kept in memory) and training, with identical results.

## 0.6.0 (2026-10-14)

- Retrain a run on new data with the same settings (studio, `easyml.retrain`, `easy-ml retrain`).
- Export and import runs as `.zip` archives; compare two runs side by side; result charts for every model.
- Class labels keep their original type in predictions; run lineage ("Retrained from #N").

## 0.5.0 (2026-10-14)

- Redesigned studio (no-build Vue, light and dark themes).
- Use trained models outside the studio: `easyml.load_model`, `easy-ml runs`, `easy-ml predict`.

## 0.4.0 (2026-10-13)

- First release on PyPI as `easy-ml-studio`: the `easy-ml` command, optional extras
  (`boost`, `nlp`, `cv`, `eda`, `all`) and Docker images.
