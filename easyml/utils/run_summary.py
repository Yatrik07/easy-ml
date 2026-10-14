"""
Human-friendly summaries of training runs.

The single place that knows how to label tasks and models, which metric ranks a run's models,
and how to present a run in lists (Run History, ``easy-ml runs``) without leaking server paths.
Pure Python: safe to import from the CLI and the predictor without pulling in heavy libraries.
"""

import json
import re
from datetime import datetime

TASK_LABELS = {
    "MLClassification": "Tabular Classification",
    "MLRegression": "Tabular Regression",
    "NLPTextClassification": "Text Classification",
    "ImageClassification": "Image Classification",
}

MODEL_LABELS = {
    # UI keys
    "logistic_regression": "Logistic Regression", "linear_regression": "Linear Regression",
    "random_forest": "Random Forest", "xg_boost": "XGBoost", "light_gbm": "LightGBM", "cat_boost": "CatBoost",
    "knn": "K-Nearest Neighbors", "ada_boost": "AdaBoost", "naive_bayes": "Multinomial Naive Bayes",
    "complement_nb": "Complement Naive Bayes", "svm": "Linear SVM", "mlp": "Neural Network (MLP)",
    # Saved model class names
    "LogisticRegression": "Logistic Regression", "LinearRegression": "Linear Regression",
    "RandomForestClassifier": "Random Forest", "RandomForestRegressor": "Random Forest",
    "XGBClassifier": "XGBoost", "XGBRegressor": "XGBoost",
    "LGBMClassifier": "LightGBM", "LGBMRegressor": "LightGBM",
    "CatBoostClassifier": "CatBoost", "CatBoostRegressor": "CatBoost",
    "KNeighborsClassifier": "K-Nearest Neighbors", "KNeighborsRegressor": "K-Nearest Neighbors",
    "AdaBoostClassifier": "AdaBoost", "AdaBoostRegressor": "AdaBoost",
    "MultinomialNB": "Multinomial Naive Bayes", "ComplementNB": "Complement Naive Bayes",
    "LinearSVC": "Linear SVM", "MLPClassifier": "Neural Network (MLP)",
    # Image models
    "SimpleCNN": "Simple CNN", "ResNet18": "ResNet-18", "MobileNetV2": "MobileNet V2", "CustomCNN": "Custom CNN",
}

_INTERNAL_UPLOAD_NAME = re.compile(r"^data(_nlp)?_[0-9a-f]{8}\.csv$")


def task_label(task):
    return TASK_LABELS.get(task, task or "")


def model_label(name):
    return MODEL_LABELS.get(name, name)


def primary_metric(task):
    """The metric used to rank a run's models, and how to display it."""
    if task == "MLRegression":
        return {"key": "r2", "name": "R²", "kind": "decimal"}
    return {"key": "accuracy", "name": "Accuracy", "kind": "percent"}


def parse_metrics(metrics_raw):
    if isinstance(metrics_raw, dict):
        return metrics_raw
    if isinstance(metrics_raw, str) and metrics_raw:
        try:
            parsed = json.loads(metrics_raw)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            return {}
    return {}


def _as_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number  # NaN -> None


def rank_models(task, metrics):
    """Models sorted best first by test score (ties broken by test F1); models without a score go last."""
    ranked = []
    for key, values in (metrics or {}).items():
        values = values if isinstance(values, dict) else {}
        ranked.append({
            "key": key,
            "label": model_label(key),
            "test_score": _as_float(values.get("accuracy_test")),
            "train_score": _as_float(values.get("accuracy_train")),
            "f1_test": _as_float(values.get("f1_test")),
        })
    ranked.sort(key=lambda m: (
        m["test_score"] is None,
        -(m["test_score"] or 0.0),
        -(m["f1_test"] or 0.0),
    ))
    return ranked


def display_dataset_name(name):
    """Runs created before 0.5.0 stored the internal upload name; show something readable instead."""
    if not name:
        return ""
    return "Uploaded dataset" if _INTERNAL_UPLOAD_NAME.match(name) else name


def normalize_timestamp(ts):
    """Stored timestamps look like 'YYYY-MM-DDTHH-MM-SS' (or with a space); return ISO 8601 or the input."""
    if not ts:
        return ts
    for fmt in ("%Y-%m-%dT%H-%M-%S", "%Y-%m-%d %H-%M-%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(ts, fmt).isoformat(timespec="seconds")
        except ValueError:
            continue
    return ts


def _strip_paths(metrics):
    clean = {}
    for key, values in metrics.items():
        if isinstance(values, dict):
            clean[key] = {k: v for k, v in values.items() if k not in ("model_path",)}
        else:
            clean[key] = values
    return clean


def summarize_run(row):
    """A compact, path-free description of a run for lists."""
    task = row.get("task")
    ranked = rank_models(task, parse_metrics(row.get("metrics_json")))
    best = ranked[0] if ranked and ranked[0]["test_score"] is not None else None
    return {
        "id": row.get("id"),
        "run_name": row.get("run_name"),
        "task": task,
        "task_label": task_label(task),
        "dataset_name": display_dataset_name(row.get("dataset_name")),
        "timestamp": normalize_timestamp(row.get("timestamp")),
        "models": [{"key": m["key"], "label": m["label"], "score": m["test_score"]} for m in ranked],
        "best_model": best["key"] if best else None,
        "best_model_label": best["label"] if best else None,
        "best_score": best["test_score"] if best else None,
        "metric": primary_metric(task),
    }


def run_detail(row):
    """Everything about one run that the UI shows, without server file paths."""
    detail = summarize_run(row)
    detail["metrics"] = _strip_paths(parse_metrics(row.get("metrics_json")))
    return detail
