# Python guide

Easy-ML Studio is also a Python library. You can train models from a script, then use any trained
model (from a script or from the studio) in your own code or from the command line.

```bash
pip install easy-ml-studio            # tabular + TF-IDF text models
pip install "easy-ml-studio[all]"     # + XGBoost/LightGBM/CatBoost, embeddings, image models, EDA
```

- [Train a model](#train-a-model)
- [Use a trained model](#use-a-trained-model)
- [Clean data with recipes](#clean-data-with-recipes)
- [Retrain on new data](#retrain-on-new-data)
- [Retrain on a schedule](#retrain-on-a-schedule)
- [Organise and share runs](#organise-and-share-runs)
- [Move runs between computers](#move-runs-between-computers)
- [Serve a model as a REST API](#serve-a-model-as-a-rest-api)
- [Command line](#command-line)
- [Where runs are stored](#where-runs-are-stored)

## Train a model

```python
from easyml import MLTrainer

run = MLTrainer(
    task="MLClassification",
    name="churn",
    dataset_path="churn.csv",
    target_column="churned",
    models=["logistic_regression", "random_forest"],
).run()

print(run.run_id)        # the run number, also shown in the studio's Run History
print(run.performance)   # test/validation scores per model
```

`MLTrainer` handles tabular and text tasks. Image models are trained in the studio.

### Common arguments

| Argument | Meaning |
|---|---|
| `task` | `"MLClassification"`, `"MLRegression"` or `"NLPTextClassification"` |
| `name` | Name of the run |
| `dataset_path` | A `.csv` or Excel file |
| `target_column` | The column to predict |
| `models` | Model keys to train and compare (see below) |
| `hyper_parameter_tuning` | Grid-search level: `0` none (default), `1` small, `2` larger |
| `random_seed` | Seed for the split, resampling, cross-validation and every model with randomness (default `42`): the same data, settings and seed always give the same models and scores |
| `cv_folds` | k-fold cross-validation of every model on the training data, e.g. `5` (default `0`: off). Scores land in `run.performance[model]["cv"]` (`mean`, `std`, `scores`); the test set still ranks the models |
| `recipe` | A Data prep recipe (name, id, `.json` file or `prep.Recipe`) to clean `dataset_path` with first, like the studio's "Train with this data". The run then cleans prediction input itself (see [Runs trained on cleaned data](#runs-trained-on-cleaned-data)) |

### Tabular options

| Argument | Default | Meaning |
|---|---|---|
| `imputation` | `False` | Fill in missing values |
| `scaling` | `False` | Scale numeric columns |
| `categorical_columns` | auto | Columns to treat as categories |
| `ignore_columns` | none | Columns to leave out, such as IDs |
| `reduce_features` | `False` | Compress features with PCA (keeps 95% of the variance) |
| `imbalance_handling` | `"none"` | Classification only: `"smote"`, `"tomek"` or `"smotetomek"` |
| `feature_weights` | none | How much models rely on input columns, e.g. `{"area": 2, "age": 0.5}` (0.1 to 10; see below) |

Tabular model keys: `logistic_regression` (classification), `linear_regression` (regression),
`random_forest`, `knn`, `ada_boost`, and with the `[boost]` extra `xg_boost`, `light_gbm`, `cat_boost`.

### Column weights

`feature_weights={"area": 2, "age": 0.5}` tells the models which inputs to rely on more (above 1) or less (below
1), from what you know about the data (Easy-ML 0.14). Models are scored on the column they predict, so weights
can lower the test score; the run records the weights and its importance charts show what each model then used.

| Model | Follows the weights |
|---|---|
| LightGBM | Yes: splits on a weighted column count more |
| K-nearest neighbours | Yes: the column counts more in distances |
| CatBoost, XGBoost | Barely: they get the weights, but measured, what they rely on hardly changes |
| Logistic regression | Barely: only its regularisation changes |
| Random Forest, AdaBoost, linear regression | No (tree results can shift slightly through rounding): leave a column out with `ignore_columns` instead |

Weights cannot be combined with `reduce_features=True` (PCA mixes the columns). Retraining, final models and
schedules keep them.

### Several columns to predict

A table run can predict several columns at once (Easy-ML 0.13): several numbers (regression), or several
categories or 0/1 label columns (classification). Pass a list:

```python
run = MLTrainer(
    task="MLRegression",
    name="houses",
    dataset_path="houses.csv",
    target_column=["price", "rent"],         # or target_columns=[...]
    target_weights={"price": 3},             # optional, 0.1 to 10; the others count 1
    models=["linear_regression", "random_forest", "light_gbm"],
).run()

run.performance["RandomForestRegressor"]["targets"]["rent"]["accuracy_test"]   # R² on rent alone
run.performance["RandomForestRegressor"]["accuracy_test"]                      # weighted average
```

- Each column gets its own copy of every model, trained, tuned and cross-validated on that column
  alone, so the accuracy of one column is never traded for another's.
- `target_weights` only decide how the columns' scores add up into the run's score, which ranks the models;
  they never change a column's predictions.
- Classification runs also report `test_metrics["exact_match"]`, the share of test rows with every column
  right.
- Class balance (`imbalance_handling`) is not available with several columns. All columns must be numbers
  (regression) or all categories (classification).

### Text options

| Argument | Default | Meaning |
|---|---|---|
| `text_column` | | The column that holds the text (required) |
| `feature_mode` | `"vectorizer"` | `"vectorizer"` (TF-IDF and friends) or `"embedding"` (needs `[nlp]`) |
| `vectorizer_type` | `"tfidf"` | `"tfidf"`, `"count"` or `"hashing"` |
| `max_features`, `ngram_range`, `min_df`, `max_df` | | Vectorizer settings |
| `lowercase`, `remove_stopwords`, `lemmatize` | | Text cleaning (lemmatizing needs `[nlp]`) |
| `embedding_model` | `"all-MiniLM-L6-v2"` | Sentence-transformer model for `"embedding"` mode |

Text model keys: `logistic_regression`, `naive_bayes`, `complement_nb`, `svm`, `random_forest`, `mlp`.

```python
run = MLTrainer(
    task="NLPTextClassification",
    name="tickets",
    dataset_path="tickets.csv",
    target_column="category",
    text_column="message",
    models=["logistic_regression", "svm"],
).run()
```

Runs trained from Python appear in the studio's Run History like any other run.

## Use a trained model

```python
import easyml

model = easyml.load_model(5)       # run #5, its best model
print(model)                       # Predictor(run_id=5, task='Tabular Classification', model='Random Forest')
print(model.available_models)      # every model of the run, best first

model = easyml.load_model(5, "LogisticRegression")   # or pick a model by class name
```

`predict` always returns a list with one prediction per input:

| Run type | Accepted input |
|---|---|
| Tabular | a dict (one row), a list of dicts, a pandas `DataFrame`, or a `.csv`/Excel path |
| Text | a string, a list of strings, a pandas `Series`, or a `.csv` path (uses the run's text column) |
| Image | an image path, a list of image paths, or a folder of images (needs `[cv]`) |

```python
model.predict({"age": 45, "city": "NY"})          # -> ["high"]
model.predict("new_customers.csv")                # one prediction per row
model.predict(["great phone", "late delivery"])   # text runs
model.predict("photos/")                          # image runs

predictions, confidences = model.predict_with_confidence("new_customers.csv")
```

`confidences` holds the probability of each predicted class, or `None` for regression and for models
that do not report probabilities (such as `LinearSVC`).

Predicted classes keep the type of the training labels: a target of `0`/`1` predicts the integers `0`
and `1`, a target of `"yes"`/`"no"` predicts strings. (Before 0.6.0, numeric labels came back as
strings such as `"1"`; runs trained with older versions now return numbers too.)

Tabular inputs can be strings or numbers, the same as the studio's form: values are converted to the
types seen during training. Missing columns are treated as missing values.

`model.input_columns` lists the columns to pass in.

A run with several columns to predict answers one `{column: prediction}` dict per row (confidences are
dicts too), and `model.targets` lists the columns. By default it uses the **best model for each column**
(`"best_for_each_target"`): every column is predicted by the model that scored best on it. Load one model
for all columns by name as usual.

```python
model = easyml.load_model(7)                      # model.model_name == "best_for_each_target"
model.predict({"area": 120, "rooms": 3, "city": "NY"})    # -> [{"price": 415.2, "rent": 49.8}]
```

### Runs trained on cleaned data

A run trained on data cleaned with a Data prep recipe (Easy-ML 0.11+) cleans prediction input itself:
give it the data as it originally looks, and it applies the same steps with the numbers learned from
its training data. Fill values, outlier limits and ranges come from the training data, never from the rows
you predict, so one row gets the same treatment as a whole file.

```python
model = easyml.load_model(12)
model.cleans_input        # True
model.input_columns       # the original columns, e.g. ["area", "bedrooms", ...] (no "area splitted")
model.predict({"area": 5800, "bedrooms": 3, "mainroad": "yes"})
model.predict(already_clean_frame)                  # has the cleaned columns: used as it is
model.predict(frame, cleaned=True)                  # or say so explicitly (cleaned=False always cleans)
```

Steps that remove or reorder rows (remove duplicates, filters, sorting) are skipped, because every row
needs a prediction; `predict_with_notes` returns notes about rows a training filter would have removed.
Runs cleaned before 0.11 expect cleaned input, as before.

### Check new data against the training data

```python
report = model.check("new_customers.csv")   # same inputs as predict (tables and text)
report.ok                                   # False when something looks different
print(report.summary)                       # "2 of 12 columns look different from the training data. ..."
report.to_frame()                           # one row per column: status, PSI, empty shares, outside range, unseen
```

Each column is compared with the training data: numbers outside the training range, categories the model
never saw, cells that are empty more often, and (with 30 rows or more) how far the distribution moved
(population stability index: under 0.1 fine, 0.1 to 0.25 slightly different, above 0.25 clearly
different). The check only informs; predictions are made either way.

Errors:

- `easyml.utils.database_operations.RunNotFoundError` when the run does not exist.
- `ValueError` when the model is not part of the run.
- `TypeError` when the input type does not fit the run.

## Clean data with recipes

The studio's **Data prep** page cleans a table step by step with a live preview (remove columns, fill
missing values, filter rows, split dates, ...). Saved as a **recipe**, the same steps can be applied to
new files from Python or the command line:

```python
from easyml import prep

prep.list_recipes()                                  # [{"id": 1, "name": "customers", "steps": 5, ...}]
recipe = prep.Recipe.load("customers")               # by name, id, or a recipe .json file
clean = recipe.apply("october.csv")                  # a pandas DataFrame
recipe.apply("october.csv", output="october_clean.csv")

prep.profile("october.csv")                          # one summary per column: kind, missing, unique, ...
```

Recipes can also be written in code. A step is `{"op": ..., "params": {...}}`; `prep.steps()` lists every
step with its settings:

```python
recipe = prep.Recipe(name="churn cleanup", steps=[
    {"op": "drop_columns", "params": {"columns": ["customer_id"]}},
    {"op": "fill_missing", "params": {"strategy": "median"}},
    {"op": "clean_text", "params": {"columns": ["city"], "trim": True, "case": "upper"}},
    {"op": "date_parts", "params": {"column": "signup", "parts": ["year", "month"]}},
])
recipe.save()                                        # now listed in the studio too
print(recipe.to_json())                              # share it as a file; Recipe.load("recipe.json") reads it
```

| Group | Steps |
|---|---|
| Columns | `drop_columns`, `keep_columns`, `rename_columns`, `change_type` |
| Rows | `remove_duplicates`, `remove_missing`, `filter_rows`, `sort_rows` |
| Values | `fill_missing`, `replace_values`, `clean_text`, `clip_outliers`, `round_numbers` |
| New columns | `date_parts`, `bin_numbers`, `combine_columns` |

- Steps run from top to bottom. A step that cannot run on a file (for example a column it needs is
  missing) makes `apply` raise a `ValueError` that names the step, so a recipe never half-cleans data
  silently.
- Steps that use statistics (median/mean fill, outlier limits, ranges) compute them on the data they are
  applied to, so new data is filled with its own medians. (A *trained model* is different: it keeps the
  numbers from its training data, see [Runs trained on cleaned data](#runs-trained-on-cleaned-data).)
- Nothing in a recipe runs code: steps are a fixed list of pandas operations.

A run trained on cleaned data (the **Train with this data** button, or `MLTrainer(..., recipe=...)`)
remembers its recipe: predictions clean raw input with it, and retraining cleans new files with the same
steps (next section).

## Retrain on new data

Train a run again on newer data. The new run uses exactly the same settings as the original
(models, preprocessing, which columns are categorical, tuning level, text settings, image settings),
records which run it came from, and leaves the original untouched.

```python
import easyml

result = easyml.retrain(5, "customers_october.csv")              # same settings as run #5
result = easyml.retrain(5, "customers_october.csv", name="churn-oct", models=["random_forest"])

print(result)                  # RunResult(run_id=12, parent_run_id=5, task='Tabular Classification', ...)
print(result.best_model, result.performance)
for note in result.warnings:   # e.g. "Ignoring 1 column(s) the original run did not use: notes."
    print(note)
result.load_model().predict("new_customers.csv")
```

- The new data must contain every column the run was trained on (and the target). Otherwise
  `easyml.retraining.DatasetMismatchError` (a `ValueError`) lists the missing columns in `.missing_columns`.
- Columns the original run did not use are ignored, so the retrained model accepts the same input.
- Models that are not installed on this computer are skipped with a warning.
- Image runs take a folder with one sub-folder per class. Image runs trained before 0.6.0 did not save
  their training settings, so defaults are used for those (listed in `result.warnings`).
- A run trained on cleaned data cleans the new file with the same recipe first. If the file is
  already clean (the recipe does not fit it), it is used as it is, with a warning. Choose another
  recipe with `recipe="name"` (or an id, or a `prep.Recipe`), or skip cleaning with `recipe=False`:

  ```python
  easyml.retrain(5, "october_raw.csv", recipe="customers")
  easyml.retrain(5, "october_clean.csv", recipe=False)
  ```

## Train the final model

A run holds back a test set (20 % of the rows) to measure its models honestly. Once the scores (and
cross-validation, if on) show a model is good, train it again on **all** the data for use:

```python
final = easyml.train_final(5)                                  # run #5's best model, on every row
final = easyml.train_final(5, models=["RandomForestClassifier"], name="churn-final")
final.load_model().predict("new_customers.csv")
```

The final run uses the evaluated run's settings and the best hyperparameters its search found (no new
search), and the same seed, so training it twice gives the same model. No rows are left to test it on, so
its `performance` repeats the evaluated run's test and cross-validation scores with
`"estimated_from_run": 5`; `accuracy_train`/`train_metrics` are measured on all the rows. The run has
`origin == "final"` and `parent_run_id == 5`; runs trained on cleaned data keep cleaning input the same way.
`train_final` raises `ValueError` for image runs and for a run that already is a final model. Retraining a
final model on new data trains an evaluated run again (with a test set).

## Retrain on a schedule

A schedule keeps a model fresh: it retrains a run when new data arrives (or at set times) and uses the
new model only if it is better.

```python
from easyml import schedules

schedules.create(5, "C:/exports/customers/", name="churn-nightly", daily="02:00")
schedules.create(5, "C:/exports/customers.csv", name="churn-live", on_new_data=True, min_improvement=0.01)

schedules.list()                              # each schedule with the model it uses now ("champion")
event = schedules.run_now("churn-nightly")    # retrain now, whatever the time
print(event["decision"], event["message"])    # "promoted", "kept champion" or "failed"
schedules.events("churn-nightly")             # what it did, newest first
schedules.disable("churn-nightly")            # enable(), delete() work the same way
```

| Argument | Meaning |
|---|---|
| `source` | A file (retrained on when it changes), a folder (its newest file matching `pattern` is used), or for image runs a folder of class folders |
| `every` / `daily` / `weekly` / `on_new_data` | When to retrain (exactly one): `every="6h"` (or `"30m"`, `"1d"`), `daily="02:00"`, `weekly="monday 02:00"`, `on_new_data=True` |
| `pattern` | File names to use in a folder, e.g. `"orders_*.csv"` (default: CSV and Excel files) |
| `recipe` | Data-prep recipe that cleans each new file; `"none"` uses it as it is; by default the run's own recipe, if any |
| `min_improvement` | How much better the new model must be: `0.01` = one accuracy point (R² for regression); `0` switches on a tie |

How a schedule decides:

1. It retrains the model currently in use, with the same settings, on the new data. Unchanged data is
   skipped ("no new data"); a file counts only once it has not changed for 10 seconds, so a file that is
   still being copied is never used.
2. Both models predict the new run's test rows (data neither trained on) and are scored with the run's
   metric. When the old model cannot score them (for example the columns changed, or image runs), their
   own test scores are compared and the event says so (`compared_on`). Models trained on cleaned data get
   those rows as they arrived and clean them with their own recipe, so each is judged the way it predicts.
3. A better model becomes the one in use (when the model in use is a final model, the better one is
   first trained again on all the new data, so the schedule keeps using a final model). A worse one is still saved in Run History (marked "Not
   promoted"), so nothing is lost.

Each attempt also compares the new data with the data the model in use was trained on (see
[Check new data](#check-new-data-against-the-training-data)): `event["data_check"]` holds
`{"status", "summary", "columns"}`, and the message says when the data looks different. This only informs;
the decision is still made on the scores.

Schedules run while the studio is open, or without it:

```bash
easy-ml schedule worker          # keeps running and checks every 30 s (Ctrl+C stops it)
easy-ml schedule run-due         # runs what is due once and exits
```

`run-due` suits Windows Task Scheduler (action: `easy-ml`, arguments: `schedule run-due`, repeat every
15 minutes) or cron (`*/15 * * * * easy-ml schedule run-due`). The studio, workers and `run-due` can run
at the same time: each schedule is claimed in the database, so it never runs twice at once.

`easy-ml serve --schedule churn-nightly` serves the schedule's current model and switches to each new
one it promotes, without a restart (see below).

## Organise and share runs

```python
easyml.report(5)                          # easyml-run-5-<name>.html: scores, charts and settings in one file
easyml.report(5, "churn-for-the-team.html")
```

The report is a single HTML file without scripts or external links: open it in any browser, attach it to
an email, or print it to PDF. Runs can also be renamed, tagged and given notes in Run history; tags
are passed on to retrained and scheduled runs and kept in run archives.

```bash
easy-ml tag 5 churn production            # add tags (easy-ml tag 5 production --remove removes one)
easy-ml runs --tag churn                  # only runs with this tag
easy-ml report 5 -o churn.html
```

## Move runs between computers

Save a run as a single `.zip` file and load it somewhere else:

```python
path = easyml.export_run(5)                                   # easyml-run-5-churn.zip in the current folder
path = easyml.export_run(5, "churn.zip", include_data=False)  # without the training data

new_id = easyml.import_run("churn.zip", trusted=True)         # on the other computer
easyml.load_model(new_id).predict("new_customers.csv")
```

The archive holds the models, the preprocessing, the run's settings and scores, its result charts and
(unless `include_data=False`) the training data, which explanations and retraining use. It records the
library versions that trained the run; importing warns when this computer has different versions,
because saved models may then behave differently or fail to load.

Model files can run code when they are loaded, so only import archives from people you trust.
`import_run` refuses to import unless you pass `trusted=True`.

## Serve a model as a REST API

One command turns any run (tables, text or images) into a web service that other programs call:

```bash
easy-ml serve 5                      # http://127.0.0.1:8000, the run's best model
easy-ml serve 5 --model RandomForestClassifier --port 9000
easy-ml serve 5 --host 0.0.0.0 --api-key s3cret    # reachable from your network, with a key
```

```python
import easyml

easyml.serve(5, port=8000)           # blocks until stopped (Ctrl+C)
```

Open <http://127.0.0.1:8000/> for the model's inputs and copy-paste examples. Endpoints:

| Endpoint | What it does |
|---|---|
| `POST /predict` | Predictions (formats below) |
| `GET /info` | The model, its task, input columns, classes and an example request |
| `GET /health` | `{"status": "ok"}` while the service is running |

| Run type | Request body | Answer |
|---|---|---|
| Tabular | `{"features": {"age": 45, "city": "NY"}}` | `{"prediction": ..., "confidence": ...}` |
| Tabular, many rows | `{"rows": [{...}, {...}]}` | `{"predictions": [...], "confidences": [...]}` |
| Text | `{"text": "..."}` or `{"texts": ["...", "..."]}` | same as above |
| Image | multipart `file` (one or several), `{"image_base64": "..."}` or `{"images_base64": [...]}` | same as above |

Regression answers have no confidence. The single-row format is the same as the "Export as API"
package, so a program written for one works with the other.

When a request does not look like the training data (a number outside the training range, a category the
model never saw, an empty value that never was empty), table answers include `"warnings": [...]` in plain
words; ordinary answers have no such key. A run trained on cleaned data takes the original columns and
cleans each request itself (its help page lists the recipe); requests that already have the cleaned
columns still work.

```python
import requests

answer = requests.post("http://127.0.0.1:8000/predict", json={"features": {"age": 45, "city": "NY"}}).json()
```

Options: `--threads` (requests handled in parallel, default 4), `--api-key` or the `EASYML_SERVE_KEY`
variable (then send `Authorization: Bearer <key>` or `X-API-Key`), `--cors-origin URL` (let web pages
call it from a browser; repeatable, `*` for any), `--max-mb` (largest request, default 20),
`--access-log`. The model is loaded once at start, so every request is fast; the service runs on the
waitress web server.

`easy-ml serve --schedule NAME` (or `easyml.serve(schedule="NAME")`) serves the model a retraining
schedule uses now and checks every 15 seconds for a newer one. A new model is loaded in the background
and then swapped in, so requests never wait or fail; `/info` shows the schedule and the current run.

By default the service only accepts requests from this computer. With `--host 0.0.0.0` anyone who can
reach the computer can use it, so add `--api-key` on shared networks.

## Command line

```bash
easy-ml runs                                        # list runs with their best model and score
easy-ml runs --task MLRegression --limit 5
easy-ml runs --json                                 # machine-readable
easy-ml runs --tag churn                            # only runs with this tag
easy-ml tag 5 churn production                      # add tags (--remove to take them off)
easy-ml report 5 -o churn.html                      # the run as one HTML file to share

easy-ml predict 5 new_customers.csv -o out.csv      # adds Prediction (and Confidence) columns
easy-ml predict 5 new_customers.csv --model RandomForestClassifier
easy-ml predict 3 "my laptop got a software update" # text runs print the label
easy-ml predict 3 reviews.csv --text-column review  # text CSV with another column name
easy-ml predict 7 photos/ -o labels.csv             # image runs: one row per image
easy-ml check 5 new_customers.csv                   # does the file look like run #5's training data?

easy-ml retrain 5 customers_october.csv             # same settings as run #5, new data
easy-ml retrain 5 october.csv --name churn-oct --models random_forest,light_gbm
easy-ml retrain 7 new_photos/                       # image runs: a folder of class folders
easy-ml retrain 5 october_raw.csv --recipe customers   # clean with a recipe first (--recipe none: as it is)
easy-ml final 5                                     # run #5's best model trained on all its data (--model to choose)

easy-ml prep list                                   # saved data-prep recipes
easy-ml prep show customers                         # a recipe's steps as JSON
easy-ml prep apply customers october.csv -o clean.csv   # clean a file (without -o: print the CSV)

easy-ml serve 5 --port 8000                         # REST API for run #5 (Ctrl+C stops it)
easy-ml serve --schedule churn-nightly              # always the schedule's current model

easy-ml schedule create 5 C:/exports/customers/ --name churn-nightly --daily 02:00
easy-ml schedule create 5 customers.csv --name churn-live --on-new-data --min-improvement 0.01
easy-ml schedule list                               # schedules, the model each uses, last result
easy-ml schedule run churn-nightly                  # retrain now
easy-ml schedule events churn-nightly               # promoted / kept champion / failed, with scores
easy-ml schedule worker                             # run schedules without the studio

easy-ml export 5 -o churn.zip                       # save run #5 as a .zip (add --no-data to leave out data.csv)
easy-ml import churn.zip                            # asks for confirmation; --yes skips the question
```

Without `-o`, the predictions are written to the terminal as CSV, so you can pipe them to another command.
After predicting a file, `easy-ml predict` prints a one-line data check on stderr when the file does not look
like the training data (`--no-check` turns it off); `easy-ml check` shows the details per column (`--json`).
`easy-ml predict`, `retrain`, `prep`, `schedule`, `export` and `import` exit with code 1 when something is wrong
(unknown run, missing columns, damaged archive, ...) and print the reason.

## Where runs are stored

Runs, models and the run database live in the Easy-ML data folder, which `load_model` and the CLI read:

| OS | Default location |
|---|---|
| Windows | `%LOCALAPPDATA%\easy-ml` |
| macOS | `~/Library/Application Support/easy-ml` |
| Linux | `~/.local/share/easy-ml` |

Set `EASYML_DATA_DIR` to use another folder. To use a model on another machine, export the run
(above), or copy the whole data folder and point `EASYML_DATA_DIR` at it.
The studio's **Make predictions** page also shows these snippets pre-filled for each run.
