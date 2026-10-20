import json
import os
import tempfile
import zipfile
from importlib import metadata

from easyml.core.artifacts import load_target_decoder, model_file_path, validate_model_name
from easyml.utils.database_operations import get_record_by_id
from easyml.utils.logger import write_log

# Model class name -> (distribution name, import needed in the generated app)
_MODEL_PACKAGES = {
    "XGBClassifier": "xgboost", "XGBRegressor": "xgboost",
    "LGBMClassifier": "lightgbm", "LGBMRegressor": "lightgbm",
    "CatBoostClassifier": "catboost", "CatBoostRegressor": "catboost",
}


def create_deployment_package(run_id: int, model_name: str) -> str:
    """
    Creates a ZIP file containing a ready-to-deploy FastAPI application for the specified model.
    Returns the absolute path to the generated ZIP file.
    """
    validate_model_name(model_name)
    write_log(f"Creating deployment package for run {run_id}, model {model_name}", "info", "export_utils.py")
    record = get_record_by_id(run_id)
    folder_path = record["folder_path"]
    task = record["task"]

    if task not in ("MLClassification", "MLRegression"):
        raise ValueError("Export is currently available for tabular classification and regression runs only.")

    from easyml.core.multi_target import BEST_FOR_EACH_TARGET, run_targets

    targets = run_targets(folder_path)  # several columns to predict (0.13): one model per target, pickled
    pipeline_source = os.path.join(folder_path, "preprocessing_pipeline.pkl")
    if not os.path.exists(pipeline_source):
        raise FileNotFoundError("The preprocessing pipeline for this run is missing.")
    combined_file = None
    if targets and model_name == BEST_FOR_EACH_TARGET:
        # Put together from the run's saved models when loaded: written into the package as one file.
        import joblib

        from easyml.core.artifacts import load_tabular_model

        combined = load_tabular_model(folder_path, model_name, task)
        handle, combined_file = tempfile.mkstemp(suffix=".pkl")
        os.close(handle)
        joblib.dump(combined, combined_file, protocol=5)
        model_source, model_classes = combined_file, list(combined.model_names)
    else:
        model_source, model_classes = model_file_path(folder_path, model_name), [model_name]
    try:
        return _write_package(run_id, folder_path, task, model_name, model_source, model_classes, targets,
                              pipeline_source)
    finally:
        if combined_file:
            os.remove(combined_file)


def _write_package(run_id, folder_path, task, model_name, model_source, model_classes, targets, pipeline_source):
    model_filename = "model" + os.path.splitext(model_source)[1]
    copies = {model_filename: model_source, "pipeline.pkl": pipeline_source}
    for name in ("config_json.json", "column_data.json"):
        source = os.path.join(folder_path, name)
        if os.path.exists(source):
            copies["config.json" if name == "config_json.json" else name] = source
    # A run trained on cleaned data: the service cleans each request with the frozen recipe first.
    prep_source = os.path.join(folder_path, "prep.json")
    cleans = os.path.exists(prep_source)
    if cleans:
        copies["prep.json"] = prep_source

    if targets:
        generated = {
            "main.py": generate_fastapi_app(task, model_name, model_filename, cleans_input=cleans, targets=targets),
            "requirements.txt": generate_requirements(model_name, cleans_input=cleans, model_classes=model_classes,
                                                      needs_easyml=True),
            "Dockerfile": generate_dockerfile(next((m for m in model_classes if m.startswith("LGBM")), model_name)),
            "README.md": generate_readme(task, model_name, model_filename, example_features(folder_path),
                                         cleans_input=cleans, targets=targets),
        }
    else:
        generated = {
            "main.py": generate_fastapi_app(task, model_name, model_filename, cleans_input=cleans),
            "requirements.txt": generate_requirements(model_name, cleans_input=cleans),
            "Dockerfile": generate_dockerfile(model_name),
            "README.md": generate_readme(task, model_name, model_filename, example_features(folder_path), cleans_input=cleans),
        }
    if task == "MLClassification" and not targets:
        decoder = load_target_decoder(folder_path)
        generated["labels.json"] = json.dumps([decoder[i] for i in sorted(decoder)])

    zip_path = os.path.join(tempfile.gettempdir(), f"easyml_deploy_{model_name}_{run_id}.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for arcname, source in copies.items():
            zipf.write(source, arcname)
        for arcname, content in generated.items():
            # Same bytes as the text-mode files written before (platform line endings).
            zipf.writestr(arcname, content.replace("\n", os.linesep).encode("utf-8"))

    write_log(f"Deployment package created at {zip_path}", "info", "export_utils.py")
    return zip_path


CLEANING_LINES = [
    "    df = pd.DataFrame([features])",
    "    # The model was trained on cleaned data: clean the request the same way first.",
    "    df, _ = prep.prepare(df)",
    "    df = df.replace(r'^\\s*$', np.nan, regex=True)",
]


SEVERAL_TARGETS_LINES = [
    "        values = np.asarray(model.predict(transformed))[0]",
    "        answer, confidence = {}, {}",
    "        for index, target in enumerate(targets):",
    "            raw = values[index].item() if hasattr(values[index], 'item') else values[index]",
    "            labels = target_classes.get(target)",
    "            if not labels:",
    "                answer[target] = float(raw)",
    "                continue",
    "            code = int(raw)",
    "            answer[target] = labels[code] if 0 <= code < len(labels) else raw",
    "            estimator = model.estimators_[index]",
    "            if hasattr(estimator, 'predict_proba'):",
    "                confidence[target] = float(np.max(estimator.predict_proba(transformed)[0]))",
]


def generate_fastapi_app(task: str, model_name: str, model_filename: str, cleans_input: bool = False,
                         targets=None) -> str:
    is_classification = "classification" in task.lower()
    is_catboost = model_name.startswith("CatBoost") and not targets  # several targets: every model is pickled

    lines = [
        '"""FastAPI service generated by Easy-ML Studio."""',
        "import json",
        "import os",
        "from typing import Any, Dict",
        "",
        "import joblib",
        "import numpy as np",
        "import pandas as pd",
        "from fastapi import FastAPI, HTTPException",
        "from pydantic import BaseModel",
    ]
    if is_catboost:
        lines.append(f"from catboost import {model_name}")
    if cleans_input:
        lines.append("from easyml.dataprep.frozen import FrozenRecipe")
    lines += [
        "",
        "app = FastAPI(title='Easy-ML Exported Model API', version='1.0.0')",
        "",
        "",
        "class InferenceInput(BaseModel):",
        "    features: Dict[str, Any]",
        "",
        "",
        "def _read_json(path):",
        "    if not os.path.exists(path):",
        "        return {}",
        "    with open(path, 'r', encoding='utf-8') as handle:",
        "        return json.load(handle)",
        "",
        "",
        "pipeline = joblib.load('pipeline.pkl')",
    ]
    if is_catboost:
        lines += [f"model = {model_name}()", f"model.load_model('{model_filename}')"]
    else:
        lines.append(f"model = joblib.load('{model_filename}')")
    if targets:
        # One model per column to predict (Easy-ML's PerTargetModel), and each column's classes.
        labels_lines = ["config = _read_json('config.json')",
                        "targets = config.get('target_columns') or []",
                        "target_classes = config.get('target_classes') or {}"]
    else:
        labels_lines = ["labels = _read_json('labels.json')",
                        "target_labels = labels if isinstance(labels, list) else []"]
    lines += [
        "column_dtypes = _read_json('column_data.json').get('column_dtypes', {})",
    ] + (["prep = FrozenRecipe.load('prep.json')"] if cleans_input else []) + labels_lines + [
        "",
        "",
        "def _prepare(features):",
    ] + (CLEANING_LINES if cleans_input else [
        "    df = pd.DataFrame([features]).replace(r'^\\s*$', np.nan, regex=True)",
    ]) + [
        "    for col, dtype in column_dtypes.items():",
        "        if col not in df.columns:",
        "            df[col] = np.nan",
        "        elif dtype.startswith(('int', 'float', 'uint')):",
        "            values = pd.to_numeric(df[col], errors='coerce')",
        "            if dtype.startswith(('int', 'uint')) and values.notna().all():",
        "                values = values.astype('int64')",
        "            df[col] = values",
        "    return df",
        "",
        "",
        "@app.post('/predict')",
        "def predict(data: InferenceInput):",
        "    try:",
        "        transformed = pipeline.transform(_prepare(data.features))",
    ]
    if targets:
        lines += SEVERAL_TARGETS_LINES + (
            ["        return {'prediction': answer, 'confidence': confidence or None}"] if is_classification
            else ["        return {'prediction': answer}"])
    elif is_classification:
        lines += [
            "        prediction = np.asarray(model.predict(transformed)).reshape(-1)[0]",
            "        confidence = None",
            "        if hasattr(model, 'predict_proba'):",
            "            confidence = float(np.max(model.predict_proba(transformed)[0]))",
            "        raw = prediction.item() if hasattr(prediction, 'item') else prediction",
            "        index = int(raw)",
            "        label = target_labels[index] if 0 <= index < len(target_labels) else raw",
            "        return {'prediction': label, 'confidence': confidence}",
        ]
    else:
        lines += [
            "        prediction = np.asarray(model.predict(transformed)).reshape(-1)[0]",
            "        return {'prediction': float(prediction)}",
        ]
    lines += [
        "    except Exception as e:",
        "        raise HTTPException(status_code=400, detail=str(e))",
        "",
        "",
        "@app.get('/health')",
        "def health():",
        "    return {'status': 'ok'}",
        "",
    ]
    return "\n".join(lines)


def installed_version(dist):
    """Installed version of a distribution, or None."""
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _pinned(dist):
    version = installed_version(dist)
    return f"{dist}=={version}" if version else dist


def generate_requirements(model_name: str, cleans_input: bool = False, model_classes=None,
                          needs_easyml: bool = False) -> str:
    """Pin the libraries that unpickle the model to the versions that trained it."""
    reqs = ["fastapi", "uvicorn", "pydantic"]
    reqs += [_pinned(dist) for dist in ("scikit-learn", "pandas", "numpy", "joblib")]
    for dist in dict.fromkeys(_MODEL_PACKAGES[m] for m in (model_classes or [model_name]) if m in _MODEL_PACKAGES):
        reqs.append(_pinned(dist))
    if cleans_input or needs_easyml:  # Easy-ML's own code: the frozen recipe, or the per-target model
        from easyml import __version__

        reqs.append(f"easy-ml-studio=={__version__}")
    return "\n".join(reqs) + "\n"


def generate_dockerfile(model_name: str = "") -> str:
    system_deps = ""
    if model_name.startswith("LGBM"):
        # LightGBM needs the OpenMP runtime, which slim images do not ship.
        system_deps = "RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*\n\n"
    return f"""FROM python:3.11-slim

WORKDIR /app

{system_deps}COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""


def example_features(folder_path, limit=None):
    """One real input row from the training data (target and ignored columns removed), for examples."""
    import math

    import pandas as pd

    from easyml.core.artifacts import load_frozen_recipe, read_json

    prep = load_frozen_recipe(folder_path)
    if prep is not None:  # the run cleans its input: examples use the original columns
        example = dict(prep.data.get("example") or {})
        return {c: example.get(c) for c in (prep.input_columns[:limit] if limit else prep.input_columns)}
    columns = list((read_json(folder_path, "column_data.json").get("column_dtypes") or {}).keys())
    data_path = os.path.join(folder_path, "data.csv")
    row = {}
    if columns and os.path.exists(data_path):
        try:
            first = pd.read_csv(data_path, nrows=1, usecols=lambda c: c in columns)
            if len(first):
                row = {c: first[c].iloc[0] for c in columns if c in first.columns}  # per column: keeps ints as ints
        except (OSError, ValueError):
            row = {}
    example = {}
    for column in columns[:limit] if limit else columns:
        value = row.get(column)
        if hasattr(value, "item"):
            value = value.item()
        if value is None or (isinstance(value, float) and math.isnan(value)):
            value = None
        example[column] = value
    return example


def generate_readme(task: str, model_name: str, model_filename: str = "model.pkl", example=None,
                    cleans_input: bool = False, targets=None) -> str:
    from easyml.utils.run_summary import model_label, task_label

    example = example or {"column_a": 1.5, "column_b": "value"}
    is_classification = "classification" in task.lower()
    body = json.dumps({"features": example}, indent=2)
    one_line = json.dumps({"features": example})
    response = ('{"prediction": "<class>", "confidence": 0.93}' if is_classification
                else '{"prediction": 123.45}')
    columns = "\n".join(f"- `{name}`" for name in example)
    label_file = "| `labels.json` | Class names, in the order the model uses them |\n" if is_classification else ""
    if targets:  # several columns to predict: one answer per column
        if is_classification:
            response = json.dumps({"prediction": {t: "<class>" for t in targets}, "confidence": {t: 0.93 for t in targets}})
        else:
            response = json.dumps({"prediction": {t: 123.45 for t in targets}})
        label_file = ""
        runs_on = ("It predicts " + ", ".join(f"`{t}`" for t in targets) + " with one model per column. It needs no "
                   "running studio: `requirements.txt` installs the `easy-ml-studio` package for the code that "
                   "holds those models together" + (" and cleans each request like the training data"
                                                    if cleans_input else "") + ".")
        if cleans_input:
            label_file = ("| `prep.json` | The data-prep recipe and the numbers it learned from the training data; "
                          "each request is cleaned with it first |\n")
    elif cleans_input:
        label_file += ("| `prep.json` | The data-prep recipe and the numbers it learned from the training data; "
                       "each request is cleaned with it first |\n")
        runs_on = ("It needs no running studio: `requirements.txt` installs the `easy-ml-studio` package only for "
                   "its data-prep code, because the model was trained on cleaned data and every request is "
                   "cleaned the same way. Send the original columns.")
    else:
        runs_on = "It runs on its own, without Easy-ML Studio, on any computer or server with Python 3.10+\n(or Docker)."
    return f"""# {model_label(model_name)} prediction API

Exported from Easy-ML Studio: your trained **{task_label(task)}** model ({model_name}) as a small
web service. {runs_on} Other programs send it one row of data and get the prediction back as JSON.

## What is in this folder

| File | What it is |
|---|---|
| `main.py` | The web service (FastAPI) |
| `{model_filename}` | The trained model |
| `pipeline.pkl` | The preprocessing used in training (missing values, encoding, scaling) |
| `config.json`, `column_data.json` | The run's settings and input columns |
{label_file}| `requirements.txt` | The exact library versions the model was trained with |
| `Dockerfile` | Runs the service in a container |

## Run it

1. Unzip this file and open a terminal in the folder.
2. Create an environment and install the libraries:
   ```bash
   python -m venv .venv
   .venv\\Scripts\\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
   pip install -r requirements.txt
   ```
3. Start the service:
   ```bash
   uvicorn main:app --port 8000
   ```
4. Open <http://127.0.0.1:8000/docs> in a browser: an interactive page where you can send test requests.

## Make a prediction

Send a `POST` request to `/predict` with the input columns under `"features"`:

```json
{body}
```

Python:
```python
import requests  # pip install requests

row = {json.dumps(example)}
print(requests.post("http://127.0.0.1:8000/predict", json={{"features": row}}).json())
```

curl (macOS/Linux):
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{one_line}'
```

PowerShell (Windows):
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body '{one_line}'
```

Answer:
```json
{response}
```

Input columns:
{columns}

Values can be numbers or text, like in the studio's form. A missing column counts as a missing value.
Other endpoints: `GET /health` (is the service up) and `GET /docs` (interactive documentation).

## Run it with Docker

```bash
docker build -t my-model-api .
docker run -p 8000:8000 my-model-api
```
"""
