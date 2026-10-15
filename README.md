# Easy-ML Studio

Easy-ML Studio is a no-code machine learning studio that runs on your own machine. Upload a dataset in the browser, train several models side by side, compare their metrics, make predictions, explain them with SHAP, and export a trained model as a ready-to-run FastAPI service.

It covers three workflows:

- **Tabular** classification and regression (CSV / Excel)
- **Text** classification (TF-IDF or sentence embeddings)
- **Image** classification (SimpleCNN, ResNet18, MobileNetV2, or your own CNN)

Everything stays local: the UI, the models and your data never leave your computer.

## Install

```bash
pip install easy-ml-studio
```

The core install covers tabular ML with scikit-learn models and TF-IDF text classification. Add extras for more:

| Extra | Adds |
|---|---|
| `[boost]` | XGBoost, LightGBM, CatBoost models |
| `[nlp]` | Lemmatization (spaCy) and sentence-embedding features |
| `[cv]` | Image classification with PyTorch |
| `[eda]` | SHAP explanations and automatic EDA reports |
| `[all]` | Everything above |

```bash
pip install "easy-ml-studio[all]"
easy-ml download-models   # spaCy English model, needed for lemmatization
```

Python 3.10 – 3.12 is supported. Features whose extra is not installed are greyed out in the UI with the command to enable them.

## Launch

```bash
easy-ml
```

The studio opens in your browser at <http://127.0.0.1:5010>. Useful options:

```bash
easy-ml --port 8080          # use another port
easy-ml --no-browser         # don't open a browser tab
easy-ml --host 0.0.0.0       # expose on your network (see Security below)
python -m easyml             # same as easy-ml
```

## Features

- Train several models in one run and compare accuracy, confusion matrices and classification reports
- Data prep without code: clean a table step by step with a live preview, then train on it or save the steps as a reusable recipe
- Result charts for every model: ROC and precision-recall curves, predicted vs actual, errors, feature importance and top words
- Light or thorough hyperparameter search (GridSearchCV)
- Missing-value imputation, scaling, PCA, one-hot encoding, and SMOTE / Tomek Links for imbalanced classes
- Live progress and training curves while models train; stop a run at any time
- Run history stored in SQLite, so every model can be reused later
- Retrain any run on new data with the same settings, and compare two runs side by side
- Move a run to another computer as a single .zip (download and import)
- Serve any trained model as a REST API with one command (`easy-ml serve`)
- Single-record and batch (CSV) predictions
- SHAP waterfall explanations for tabular predictions
- One-click export of a tabular model as a FastAPI app with Dockerfile
- Use any trained model from Python or the command line (`easyml.load_model`, `easy-ml predict`)
- Automatic EDA reports with ydata-profiling
- Light and dark themes, works fully offline

## Use your models outside the studio

Every run is saved, so you can use its models from any Python program or script on the same machine:

```python
import easyml

model = easyml.load_model(5)                      # run #5, best model by default
model.predict({"age": 45, "city": "NY"})          # -> ["high"]
model.predict("new_customers.csv")                # one prediction per row
model.predict(["great phone", "late delivery"])   # text runs
model.predict("photos/")                          # image runs: a file, a list or a folder
```

Or from the command line:

```bash
easy-ml runs                                        # list runs with their best model
easy-ml predict 5 new_customers.csv -o out.csv      # adds Prediction (and Confidence) columns
easy-ml predict 3 "my laptop got a software update" # text runs print the label
easy-ml predict 7 photos/ -o labels.csv             # image runs: one row per image
easy-ml retrain 5 october.csv                       # run #5's settings on new data
easy-ml export 5 -o churn.zip                       # move run #5 to another computer...
easy-ml import churn.zip                            # ...and load it there
easy-ml serve 5                                     # REST API for run #5 on http://127.0.0.1:8000
easy-ml prep apply customers october.csv -o clean.csv   # clean a file with a saved data-prep recipe
```

Pick a specific model with `model_name=` / `--model` (for example `RandomForestClassifier`). The studio's prediction page shows these snippets pre-filled for each run.

## Train from Python

```python
from easyml import MLTrainer

run = MLTrainer(
    task="MLClassification",          # or "MLRegression", "NLPTextClassification"
    name="churn",
    dataset_path="churn.csv",
    target_column="churned",
    models=["logistic_regression", "random_forest"],
).run()

print(run.run_id, run.performance)
```

Runs trained from Python appear in the studio's Run History and can be used for predictions there. The [Python guide](https://github.com/Yatrik07/easy-ml/blob/main/docs/PYTHON_API.md) lists every option.

## Where data is stored

Uploads, trained models, the run database and logs live outside the package, so upgrading never deletes them:

| OS | Default location |
|---|---|
| Windows | `%LOCALAPPDATA%\easy-ml` |
| macOS | `~/Library/Application Support/easy-ml` |
| Linux | `~/.local/share/easy-ml` |

Set the `EASYML_DATA_DIR` environment variable to use another folder. **Settings → Clear Data** in the studio deletes everything stored there.

## Security

Easy-ML Studio is a local, single-user tool with no login.

- By default it only listens on `127.0.0.1`, so other machines cannot reach it.
- Requests from other websites are rejected, so a page you visit cannot drive the studio.
- Only use `--host 0.0.0.0` on a network you trust: anyone who can reach the port can use the studio and its data.

## Docker

```bash
docker compose up --build
```

The image targets `base`, `ml`, `nlp`, `cv` and `full` (default) select which extras are installed. Data is kept in the `easyml-data` volume.

## Documentation

- [User guide](https://github.com/Yatrik07/easy-ml/blob/main/docs/USER_GUIDE.md)
- [Python guide](https://github.com/Yatrik07/easy-ml/blob/main/docs/PYTHON_API.md): train and use models from code and the command line
- [Developer guide](https://github.com/Yatrik07/easy-ml/blob/main/docs/DEVELOPER_GUIDE.md)

## Development

```bash
git clone https://github.com/Yatrik07/easy-ml.git
cd easy-ml
python -m venv venv
venv\Scripts\activate            # Windows; use `source venv/bin/activate` on macOS/Linux
pip install -e ".[all,dev]"
pytest
easy-ml
```

## License

Apache License 2.0
