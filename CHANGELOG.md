# Changelog

All notable changes to Easy-ML Studio (`pip install easy-ml-studio`). Versions follow
[semantic versioning](https://semver.org/); existing runs and data keep working across upgrades.

## 0.12.0 (2026-10-18)

The final model.

- **Train final model**: once cross-validation and the test set show a model is good, train it again on
  all the data, training and test rows, with the same settings and the best hyperparameters the run found.
  It is the model to use. With no rows left to test it on, its scores are the evaluated run's test and
  cross-validation scores, clearly marked as estimates ("Expected R²", "Final model of #5"). In the studio:
  **Train final model** on the Results page and **Final model** in a run's details; also
  `easyml.train_final(5)`, `easy-ml final 5` and `POST /runs/<id>/final`. Image runs are not supported
  (they need their validation images to decide when to stop training).
- Schedules whose model in use is a final model stay final: a better new model is trained again on all the
  new data before it is put in use.
- Retraining a final model trains an evaluated run again (with a test set) and says so.
- Dialogs opened from a run's details (such as **Retrain**) no longer open hidden behind the details panel.
- `easy-ml serve` with an unknown run says so, even when the port is busy.

## 0.11.0 (2026-10-18)

Predict the way you trained.

- **Models trained on cleaned data clean their input themselves.** A run trained on data from Data prep
  now keeps its recipe frozen, with the numbers it learned from the training data (fill values, outlier
  limits, ranges). Predictions take the original columns and clean them exactly like the training data:
  the prediction form, CSV files, `easy-ml predict`, Python, `easy-ml serve`, explanations and **Export as
  API** all do this. The form asks for `area` instead of a range like "area splitted"; **Try a real row**
  fills in the raw row. Steps that remove or reorder rows are skipped, since every row needs a prediction.
  Data that already has the cleaned columns still works. Runs cleaned with older versions are unchanged
  (retrain them to get this).
- **Data check**: does new data look like the training data? Values outside the training range, new
  categories, unusual empty cells and (for 30+ rows) shifted distributions are explained in plain words:
  a warning under a single prediction, a **Data check** card after a CSV prediction, `warnings` in serve
  answers, `easy-ml check`, `model.check(...)`, and a badge in a schedule's history.
- Scheduled retraining compares the old and new model on the new data cleaned each model's own way.
- `MLTrainer(..., recipe="name")` cleans the dataset with a saved recipe before training, like the studio.

## 0.10.1 (2026-10-18)

Easy-ML Studio is open source: the code is on [GitHub](https://github.com/Yatrik07/easy-ml).

- The About page links to GitHub, the documentation and the issue tracker; the "Python guide" links on
  Home and About open the full guide.
- **Report a bug** opens GitHub's bug form with your version, operating system, Python and installed
  optional libraries already filled in (no file paths or data). New bug and feature-idea forms, and a
  [contributing guide](https://github.com/Yatrik07/easy-ml/blob/main/CONTRIBUTING.md).
- **Explain this prediction** no longer disappears when SHAP is not installed: the button is greyed out and
  shows the install command, with a reminder to restart the studio afterwards.

## 0.10.0 (2026-10-15)

Share and organise runs, and trust the scores.

- **Random seed** (Choose models step, `random_seed=`, default 42): the same data, settings and seed always
  give the same models and scores. Random forests, AdaBoost and the MLP used to vary slightly between
  trainings; they are now repeatable, so their scores can differ slightly from earlier versions. The
  train/test split is unchanged with the default seed.
- **Cross-validation** (off / 3 / 5 / 10 folds, `cv_folds=`): every model is also scored on folds of the
  training data, shown as "97.9% ± 3.1" in the leaderboard, the report and the API. Preprocessing and
  resampling are refit inside each fold. When models tie on the test score, the better cross-validation
  score ranks first.
- **Notes, tags and renaming** in a run's details; tags show on runs and as a filter in Run history, and
  the search finds them. Retrained and scheduled runs keep their parent's tags; archives keep both.
  CLI: `easy-ml tag`, `easy-ml runs --tag`.
- **Run report**: one HTML file with a run's scores, charts, settings, notes and tags to send to anyone
  (Report in the run details, Download report on the Results page, `easy-ml report`, `easyml.report`).
- API responses keep the training order of models instead of sorting them A to Z.

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
