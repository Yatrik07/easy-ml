# Python guide

Easy-ML Studio is also a Python library. You can train models from a script, then use any trained
model (from a script or from the studio) in your own code or from the command line.

```bash
pip install easy-ml-studio            # tabular + TF-IDF text models
pip install "easy-ml-studio[all]"     # + XGBoost/LightGBM/CatBoost, embeddings, image models, EDA
```

- [Train a model](#train-a-model)
- [Use a trained model](#use-a-trained-model)
- [Retrain on new data](#retrain-on-new-data)
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

### Tabular options

| Argument | Default | Meaning |
|---|---|---|
| `imputation` | `False` | Fill in missing values |
| `scaling` | `False` | Scale numeric columns |
| `categorical_columns` | auto | Columns to treat as categories |
| `ignore_columns` | none | Columns to leave out, such as IDs |
| `reduce_features` | `False` | Compress features with PCA (keeps 95% of the variance) |
| `imbalance_handling` | `"none"` | Classification only: `"smote"`, `"tomek"` or `"smotetomek"` |

Tabular model keys: `logistic_regression` (classification), `linear_regression` (regression),
`random_forest`, `knn`, `ada_boost`, and with the `[boost]` extra `xg_boost`, `light_gbm`, `cat_boost`.

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

Errors:

- `easyml.utils.database_operations.RunNotFoundError` when the run does not exist.
- `ValueError` when the model is not part of the run.
- `TypeError` when the input type does not fit the run.

## Retrain on new data

Train a run's recipe again on newer data. The new run uses exactly the same settings as the original
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

```python
import requests

answer = requests.post("http://127.0.0.1:8000/predict", json={"features": {"age": 45, "city": "NY"}}).json()
```

Options: `--threads` (requests handled in parallel, default 4), `--api-key` or the `EASYML_SERVE_KEY`
variable (then send `Authorization: Bearer <key>` or `X-API-Key`), `--cors-origin URL` (let web pages
call it from a browser; repeatable, `*` for any), `--max-mb` (largest request, default 20),
`--access-log`. The model is loaded once at start, so every request is fast; the service runs on the
waitress web server.

By default the service only accepts requests from this computer. With `--host 0.0.0.0` anyone who can
reach the computer can use it, so add `--api-key` on shared networks.

## Command line

```bash
easy-ml runs                                        # list runs with their best model and score
easy-ml runs --task MLRegression --limit 5
easy-ml runs --json                                 # machine-readable

easy-ml predict 5 new_customers.csv -o out.csv      # adds Prediction (and Confidence) columns
easy-ml predict 5 new_customers.csv --model RandomForestClassifier
easy-ml predict 3 "my laptop got a software update" # text runs print the label
easy-ml predict 3 reviews.csv --text-column review  # text CSV with another column name
easy-ml predict 7 photos/ -o labels.csv             # image runs: one row per image

easy-ml retrain 5 customers_october.csv             # same settings as run #5, new data
easy-ml retrain 5 october.csv --name churn-oct --models random_forest,light_gbm
easy-ml retrain 7 new_photos/                       # image runs: a folder of class folders

easy-ml serve 5 --port 8000                         # REST API for run #5 (Ctrl+C stops it)

easy-ml export 5 -o churn.zip                       # save run #5 as a .zip (add --no-data to leave out data.csv)
easy-ml import churn.zip                            # asks for confirmation; --yes skips the question
```

Without `-o`, the predictions are written to the terminal as CSV, so you can pipe them to another command.
`easy-ml predict`, `retrain`, `export` and `import` exit with code 1 when something is wrong
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
