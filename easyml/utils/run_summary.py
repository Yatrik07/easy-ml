"""
Human-friendly summaries of training runs.

The single place that knows how to label tasks and models, which metric ranks a run's models,
and how to present a run in lists (Run History, ``easy-ml runs``) without leaking server paths.
Pure Python: safe to import from the CLI and the predictor without pulling in heavy libraries.
"""

import hashlib
import json
import re
import threading
from collections import OrderedDict
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

_INTERNAL_UPLOAD_NAME = re.compile(r"^(data|data_nlp|clean)_[0-9a-f]{8}\.csv$")


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


def _as_int(value):
    # pandas returns nullable integer columns as floats (7.0) or NaN
    number = _as_float(value)
    return None if number is None else int(number)


def rank_models(task, metrics):
    """
    Models sorted best first by test score; ties go to the higher test F1, then the higher cross-validation
    mean, then training order. Models without a score go last.
    """
    ranked = []
    for key, values in (metrics or {}).items():
        values = values if isinstance(values, dict) else {}
        ranked.append({
            "key": key,
            "label": model_label(key),
            "test_score": _as_float(values.get("accuracy_test")),
            "train_score": _as_float(values.get("accuracy_train")),
            "f1_test": _as_float(values.get("f1_test")),
            "cv_mean": _as_float((values.get("cv") or {}).get("mean")) if isinstance(values.get("cv"), dict) else None,
        })
    ranked.sort(key=lambda m: (
        m["test_score"] is None,
        -(m["test_score"] or 0.0),
        -(m["f1_test"] or 0.0),
        -(m["cv_mean"] or 0.0),
    ))
    return ranked


# Ranking a run means parsing its metrics JSON, which holds every model's confusion matrices and
# reports and grows with classes² × models. Run History lists every run on each visit, so rankings
# are cached by a digest of that text: identical text always gives the identical ranking, so the
# cache never needs invalidating, and only the small ranking (not the JSON) is kept in memory.
_RANK_CACHE = OrderedDict()
_RANK_CACHE_SIZE = 4096
_RANK_LOCK = threading.Lock()


def _ranked(task, metrics_raw):
    if not isinstance(metrics_raw, str) or not metrics_raw:
        return rank_models(task, parse_metrics(metrics_raw))
    key = (task, hashlib.blake2b(metrics_raw.encode("utf-8"), digest_size=16).digest())
    with _RANK_LOCK:
        cached = _RANK_CACHE.get(key)
        if cached is not None:
            _RANK_CACHE.move_to_end(key)
    if cached is None:
        cached = tuple(rank_models(task, parse_metrics(metrics_raw)))
        with _RANK_LOCK:
            _RANK_CACHE[key] = cached
            while len(_RANK_CACHE) > _RANK_CACHE_SIZE:
                _RANK_CACHE.popitem(last=False)
    return [dict(m) for m in cached]  # copies: callers may modify their list


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


def parse_tags(raw):
    from easyml.utils.run_meta import parse_tags as _parse

    return _parse(raw)


def summarize_run(row):
    """A compact, path-free description of a run for lists."""
    task = row.get("task")
    ranked = _ranked(task, row.get("metrics_json"))
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
        "parent_run_id": _as_int(row.get("parent_run_id")),
        "origin": row.get("origin") or None,
        "tags": parse_tags(row.get("tags")),
        "notes": row.get("notes") or "",
    }


def run_detail(row):
    """Everything about one run that the UI shows, without server file paths."""
    detail = summarize_run(row)
    detail["metrics"] = _strip_paths(parse_metrics(row.get("metrics_json")))
    return detail


TUNING_LABELS = {0: "Off", 1: "Quick", 2: "Thorough"}
IMBALANCE_LABELS = {"none": "None", "smote": "SMOTE oversampling", "tomek": "Tomek links",
                    "smotetomek": "SMOTE + Tomek"}


def _list(value):
    return ", ".join(str(v) for v in value) if value else "None"


def _yes_no(value):
    return "Yes" if value else "No"


# (key, label, group, formatter) per task; formatters turn saved config values into display text.
_COMMON_SETTINGS = [
    ("recipe", "Data prep recipe", "Data",
     lambda v: f"{v.get('name')} ({len(v.get('steps') or [])} steps)" if isinstance(v, dict) else "None"),
    ("models", "Models", "Models", lambda v: _list([model_label(m) for m in (v or [])])),
    ("hyper_parameter_tuning", "Hyperparameter tuning", "Models", lambda v: TUNING_LABELS.get(v, str(v))),
    ("imbalance_handling", "Class imbalance", "Models", lambda v: IMBALANCE_LABELS.get(v or "none", v)),
    ("cv_folds", "Cross-validation", "Models", lambda v: f"{int(v)} folds" if v else "Off"),
    ("random_seed", "Random seed", "Models", str),
]
_SETTINGS = {
    "tabular": [
        ("target_column", "Target column", "Data", str),
        ("ignore_columns", "Ignored columns", "Data", _list),
        ("categorical_columns", "Categorical columns", "Data", _list),
        ("imputation", "Missing values", "Preprocessing", lambda v: "Impute" if v else "Drop rows"),
        ("scaling", "Scale numeric features", "Preprocessing", _yes_no),
        ("reduce_features", "Reduce features (PCA)", "Preprocessing", _yes_no),
    ],
    "text": [
        ("target_column", "Target column", "Data", str),
        ("text_column", "Text column", "Data", str),
        ("lowercase", "Lowercase", "Text cleaning", _yes_no),
        ("remove_stopwords", "Remove stopwords", "Text cleaning", _yes_no),
        ("lemmatize", "Lemmatize", "Text cleaning", _yes_no),
        ("regex_clean", "Strip punctuation", "Text cleaning", _yes_no),
        ("feature_mode", "Features", "Features", lambda v: "Sentence embeddings" if v == "embedding" else "Word counts"),
        ("vectorizer_type", "Vectorizer", "Features", lambda v: {"tfidf": "TF-IDF", "count": "Counts",
                                                                 "hashing": "Hashing"}.get(v, v)),
        ("max_features", "Max features", "Features", str),
        ("ngram_range", "N-grams", "Features", lambda v: "-".join(str(x) for x in v) if v else "None"),
        ("embedding_model", "Embedding model", "Features", str),
    ],
    "image": [
        ("classes", "Classes", "Data", _list),
        ("image_size", "Image size", "Data", lambda v: " x ".join(str(x) for x in v) if v else "None"),
        ("augment", "Augmentation", "Data", _yes_no),
        ("validation_split", "Validation split", "Data", lambda v: f"{round(v * 100)}%"),
        ("test_split", "Test split", "Data", lambda v: f"{round(v * 100)}%"),
        ("epochs", "Epochs", "Training", str),
        ("batch_size", "Batch size", "Training", str),
        ("learning_rate", "Learning rate", "Training", str),
        ("early_stopping_patience", "Early stopping patience", "Training", str),
    ],
}


def _settings_kind(task):
    if task in ("MLClassification", "MLRegression"):
        return "tabular"
    if task == "NLPTextClassification":
        return "text"
    return "image"


def settings_for_display(task, config):
    """
    A run's saved settings as ``[{key, label, group, value}]`` with readable values, for the run's
    config page, the compare view and the CLI. Settings an older run did not save are left out.
    """
    config = config or {}
    rows = []
    specs = _SETTINGS[_settings_kind(task)] + _COMMON_SETTINGS
    for key, label, group, fmt in specs:
        if key not in config:
            continue
        if key == "imbalance_handling" and task in ("MLRegression", "ImageClassification"):
            continue
        if key == "embedding_model" and config.get("feature_mode") != "embedding":
            continue
        if key in ("vectorizer_type", "max_features", "ngram_range") and config.get("feature_mode") == "embedding":
            continue
        try:
            value = fmt(config[key])
        except (TypeError, ValueError):
            value = str(config[key])
        rows.append({"key": key, "label": label, "group": group, "value": value})
    return rows
