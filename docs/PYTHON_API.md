# Python guide

Easy-ML Studio is also a Python library. You can train models from a script, then use any trained
model (from a script or from the studio) in your own code or from the command line.

```bash
pip install easy-ml-studio            # tabular + TF-IDF text models
pip install "easy-ml-studio[all]"     # + XGBoost/LightGBM/CatBoost, embeddings, image models, EDA
```

- [Train a model](#train-a-model)
- [Use a trained model](#use-a-trained-model)
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

Tabular inputs can be strings or numbers, the same as the studio's form: values are converted to the
types seen during training. Missing columns are treated as missing values.

Errors:

- `easyml.utils.database_operations.RunNotFoundError` when the run does not exist.
- `ValueError` when the model is not part of the run.
- `TypeError` when the input type does not fit the run.

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
```

Without `-o`, the predictions are written to the terminal as CSV, so you can pipe them to another command.
`easy-ml predict` exits with code 1 when the run, the model or the input file is not found.

## Where runs are stored

Runs, models and the run database live in the Easy-ML data folder, which `load_model` and the CLI read:

| OS | Default location |
|---|---|
| Windows | `%LOCALAPPDATA%\easy-ml` |
| macOS | `~/Library/Application Support/easy-ml` |
| Linux | `~/.local/share/easy-ml` |

Set `EASYML_DATA_DIR` to use another folder. To use your models on another machine, install
`easy-ml-studio` there and copy the whole data folder (or point `EASYML_DATA_DIR` at a copy).
The studio's **Make predictions** page also shows these snippets pre-filled for each run.
