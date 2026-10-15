"""
Data for the result charts: ROC and precision-recall curves, predicted-vs-actual and residuals,
feature importance and top terms.

Computed once per model at evaluation time (while the fitted model and the test set are in memory)
and written to ``evaluation.json`` in the run folder. Kept small: curves are thinned to at most
200 points and scatter plots to 500, so the file stays a few hundred KB even for big datasets.
A chart that cannot be computed is reported under ``skipped``; it never fails the training run.
"""

import json
import os
import time

import numpy as np

from easyml.utils.cancellation import check_cancelled
from easyml.utils.logger import write_log

FORMAT_VERSION = 1
FILE_NAME = "evaluation.json"
MAX_CURVE_POINTS = 200
MAX_SCATTER_POINTS = 500
MAX_CURVE_CLASSES = 10
TOP_FEATURES = 20
TOP_TERMS = 15
PERMUTATION_ROWS = 500
PERMUTATION_COLUMNS = 60
PERMUTATION_REPEATS = 3
PERMUTATION_SECONDS = 20


def _r(value, digits=4):
    value = float(value)
    return round(value, digits) if np.isfinite(value) else None


def _sig(value, digits=6):
    value = float(value)
    return float(f"{value:.{digits}g}") if np.isfinite(value) else None


def _thin(xs, ys, max_points=MAX_CURVE_POINTS):
    """Keep at most ``max_points`` evenly spaced points, always including both ends."""
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if len(xs) > max_points:
        index = np.unique(np.linspace(0, len(xs) - 1, max_points).round().astype(int))
        xs, ys = xs[index], ys[index]
    return [_r(x) for x in xs], [_r(y) for y in ys]


# --------------------------------------------------------------------------- classification

def class_scores(model, X, n_classes):
    """
    One score column per class code, from ``predict_proba`` or else ``decision_function``.

    Returns ``(scores, source)`` with source ``"probability"`` or ``"decision"``, or ``(None, reason)``.
    """
    classes = [int(c) for c in getattr(model, "classes_", range(n_classes))]
    if hasattr(model, "predict_proba"):
        try:
            raw, source = np.asarray(model.predict_proba(X), dtype=float), "probability"
        except Exception:  # e.g. SVC without probability=True
            raw, source = None, None
    else:
        raw, source = None, None
    if raw is None and hasattr(model, "decision_function"):
        raw, source = np.asarray(model.decision_function(X), dtype=float), "decision"
        if raw.ndim == 1:
            raw = np.column_stack([-raw, raw])
    if raw is None:
        return None, "This model does not report scores, so curves cannot be drawn."
    scores = np.full((raw.shape[0], n_classes), -np.inf if source == "decision" else 0.0)
    for column, code in enumerate(classes[:raw.shape[1]]):
        if 0 <= code < n_classes:
            scores[:, code] = raw[:, column]
    return scores, source


def classification_curves(y_codes, scores, labels, source):
    """ROC and precision-recall curves (one-vs-rest beyond two classes) plus AUC / average precision."""
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    y = np.asarray(y_codes, dtype=int)
    present = sorted(set(y.tolist()))
    if len(present) < 2:
        return None, None, "The test set contains a single class, so curves cannot be drawn."
    scores = np.where(np.isfinite(scores), scores, np.nanmin(scores[np.isfinite(scores)]) - 1)

    if len(labels) == 2:
        chosen = [1]
    else:
        support = {c: int((y == c).sum()) for c in present}
        chosen = sorted(present, key=lambda c: -support[c])[:MAX_CURVE_CLASSES]

    roc_curves, pr_curves, all_aucs = [], [], []
    for code in (present if len(labels) > 2 else chosen):
        truth = (y == code).astype(int)
        fpr, tpr, _ = roc_curve(truth, scores[:, code])
        area = auc(fpr, tpr)
        all_aucs.append(area)
        if code not in chosen:
            continue
        precision, recall, _ = precision_recall_curve(truth, scores[:, code])
        fx, fy = _thin(fpr, tpr)
        rx, ry = _thin(recall[::-1], precision[::-1])
        label = str(labels[code])
        roc_curves.append({"label": label, "auc": _r(area), "fpr": fx, "tpr": fy})
        # scikit-learn's average_precision_score, computed from the curve above instead of rebuilding it
        average_precision = max(0.0, -np.sum(np.diff(recall) * np.array(precision)[:-1]))
        pr_curves.append({"label": label, "ap": _r(average_precision),
                          "prevalence": _r(truth.mean()), "recall": rx, "precision": ry})

    roc = {"score_source": source, "curves": roc_curves, "macro_auc": _r(np.mean(all_aucs))}
    if len(labels) > 2:
        from sklearn.preprocessing import label_binarize

        truth = label_binarize(y, classes=list(range(len(labels))))
        fpr, tpr, _ = roc_curve(truth.ravel(), scores.ravel())
        fx, fy = _thin(fpr, tpr)
        roc["micro"] = {"fpr": fx, "tpr": fy}
        roc["micro_auc"] = _r(auc(fpr, tpr))
    pr = {"score_source": source, "curves": pr_curves}
    return roc, pr, None


# --------------------------------------------------------------------------- regression

def regression_points(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    n = len(y_true)
    index = np.arange(n)
    if n > MAX_SCATTER_POINTS:
        index = np.sort(np.random.default_rng(0).choice(n, MAX_SCATTER_POINTS, replace=False))
    residuals = y_true - y_pred
    bins = int(np.clip(int(np.sqrt(n)), 5, 30))
    counts, edges = np.histogram(residuals, bins=bins)
    return {
        "scatter": {"actual": [_sig(v) for v in y_true[index]], "predicted": [_sig(v) for v in y_pred[index]],
                    "n": n, "sampled": n > MAX_SCATTER_POINTS},
        "residual_histogram": {"edges": [_sig(e) for e in edges], "counts": counts.astype(int).tolist(),
                               "mean": _sig(residuals.mean()), "std": _sig(residuals.std())},
    }


# --------------------------------------------------------------------------- feature importance

def _top(values, method, metric, note=None, total=None):
    ranked = sorted(values.items(), key=lambda item: -item[1])
    features = [{"name": str(name), "value": _r(value)} for name, value in ranked[:TOP_FEATURES]]
    return {"method": method, "metric": metric, "features": features, "total_features": total or len(values),
            "note": note}


def _output_columns(pipeline, raw_columns):
    """Original column for each output column of the fitted ColumnTransformer (None when unknown)."""
    transformer = pipeline.named_steps["column_transformer"]
    mapping = []
    for name, step, columns in transformer.transformers_:
        if step == "drop" or name not in getattr(transformer, "output_indices_", {}):
            continue
        part = transformer.output_indices_[name]
        width = part.stop - part.start
        if width == 0:
            continue
        columns = [raw_columns[c] if isinstance(c, (int, np.integer)) else c for c in list(columns)]
        if width == len(columns):
            mapping.extend(columns)
            continue
        try:
            outputs = list(step.get_feature_names_out(columns))
        except Exception:
            return None
        by_length = sorted(columns, key=lambda c: -len(str(c)))
        for out in outputs:
            out = str(out).split("__")[-1]
            owner = next((c for c in by_length if out == str(c) or out.startswith(f"{c}_")), None)
            mapping.append(owner)
    return mapping


def _native_importance(model, pipeline, X_train, raw_columns):
    if "pca" in pipeline.named_steps:
        return None
    if hasattr(model, "feature_importances_"):
        weights = np.asarray(model.feature_importances_, dtype=float).reshape(-1)
    elif hasattr(model, "coef_"):
        coef = np.abs(np.asarray(model.coef_, dtype=float))
        coef = coef.mean(axis=0) if coef.ndim > 1 else coef
        # Coefficients only compare across features of the same scale.
        spread = np.asarray(np.std(np.asarray(X_train, dtype=float), axis=0)).reshape(-1)
        weights = coef.reshape(-1) * spread if spread.shape == coef.reshape(-1).shape else coef.reshape(-1)
    else:
        return None
    owners = _output_columns(pipeline, raw_columns)
    if owners is None or len(owners) != len(weights):
        return None
    totals = {}
    for owner, weight in zip(owners, weights):
        if owner is not None:
            totals[owner] = totals.get(owner, 0.0) + float(weight)
    norm = sum(totals.values())
    if not norm:
        return None
    return {k: v / norm for k, v in totals.items()}


def permutation_importance_raw(predict, X_raw, y, metric):
    """
    Score drop when one original column is shuffled, on at most 500 test rows and 60 columns,
    within a time budget; checks for cancellation between columns.
    """
    from sklearn.metrics import accuracy_score, r2_score

    score = accuracy_score if metric == "accuracy" else r2_score
    y = np.asarray(y)
    rng = np.random.default_rng(0)
    if len(X_raw) > PERMUTATION_ROWS:
        rows = np.sort(rng.choice(len(X_raw), PERMUTATION_ROWS, replace=False))
        X_raw, y = X_raw.iloc[rows].reset_index(drop=True), y[rows]
    else:
        X_raw = X_raw.reset_index(drop=True)
    columns = list(X_raw.columns)
    if len(columns) > PERMUTATION_COLUMNS:
        return None, f"Feature importance is computed for up to {PERMUTATION_COLUMNS} columns."
    base = score(y, predict(X_raw))
    started, drops, note = time.monotonic(), {}, None
    for column in columns:
        check_cancelled()
        if time.monotonic() - started > PERMUTATION_SECONDS:
            note = "Stopped early to keep training fast; some columns are not shown."
            break
        losses = []
        for _ in range(PERMUTATION_REPEATS):
            shuffled = X_raw.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            losses.append(base - score(y, predict(shuffled)))
        drops[column] = max(float(np.mean(losses)), 0.0)
    return drops, note


def tabular_importance(model, request_body, task):
    pipeline = request_body.pipeline
    raw = request_body.X_test_raw
    if pipeline is None or raw is None:
        return None, "The preprocessing pipeline was not available."
    native = _native_importance(model, pipeline, request_body.X_train, list(raw.columns))
    if native:
        return _top(native, "model", None, total=len(native)), None

    def predict(frame):
        return np.asarray(model.predict(pipeline.transform(frame))).reshape(-1)

    metric = "accuracy" if task == "classification" else "r2"
    drops, note = permutation_importance_raw(predict, raw, request_body.y_test, metric)
    if drops is None:
        return None, note
    return _top(drops, "permutation", metric, note=note, total=len(raw.columns)), None


def nlp_top_terms(model, vectorizer, labels):
    if vectorizer is None or not hasattr(vectorizer, "get_feature_names_out"):
        return None, "Top terms need a word-count or TF-IDF vectorizer."
    try:
        terms = np.asarray(vectorizer.get_feature_names_out())
    except Exception:
        return None, "Top terms need a word-count or TF-IDF vectorizer."

    def pick(weights, largest=True):
        order = np.argsort(weights)
        order = order[::-1] if largest else order
        return [{"term": str(terms[i]), "weight": _r(abs(weights[i]))} for i in order[:TOP_TERMS]]

    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_.toarray() if hasattr(model.coef_, "toarray") else model.coef_, dtype=float)
        if coef.shape[-1] != len(terms):
            return None, "Top terms are not available for this model."
        if coef.shape[0] == 1 and len(labels) == 2:
            by_class = [{"label": str(labels[1]), "terms": pick(coef[0])},
                        {"label": str(labels[0]), "terms": pick(coef[0], largest=False)}]
        else:
            by_class = [{"label": str(labels[int(code)]), "terms": pick(coef[row])}
                        for row, code in enumerate(getattr(model, "classes_", range(len(coef))))][:MAX_CURVE_CLASSES]
        return {"method": "coef", "by_class": by_class}, None
    if hasattr(model, "feature_importances_"):
        weights = np.asarray(model.feature_importances_, dtype=float)
        if len(weights) != len(terms):
            return None, "Top terms are not available for this model."
        return {"method": "importance", "overall": pick(weights)}, None
    return None, "Top terms are not available for this model."


# --------------------------------------------------------------------------- assembly and storage

def build_model_charts(parts):
    """
    Run each ``name -> callable`` in ``parts``; callables return ``(data, skip_reason)`` or a dict of
    several charts. Errors become skip reasons, except cancellation, which must stop the run.
    """
    charts, skipped = {}, {}
    for name, compute in parts.items():
        try:
            data = compute()
        except InterruptedError:
            raise
        except Exception as exc:
            write_log(f"Could not compute the {name} chart: {exc}", "warning", os.path.basename(__file__))
            skipped[name] = "Could not be computed for this model."
            continue
        if isinstance(data, tuple):
            value, reason = data
            if value is None:
                skipped[name] = reason
            else:
                charts[name] = value
        elif isinstance(data, dict):
            charts.update(data)
    if skipped:
        charts["skipped"] = skipped
    return charts


def classification_parts(model, X_test, y_test, labels):
    """Chart builders for a classifier; ``labels`` are the class names in code order."""
    cache = {}

    def curves():
        if "roc" not in cache:
            scores, source = class_scores(model, X_test, len(labels))
            if scores is None:
                cache["roc"] = (None, None, source)
            else:
                cache["roc"] = classification_curves(y_test, scores, labels, source)
        return cache["roc"]

    return {
        "roc": lambda: (curves()[0], curves()[2]),
        "pr": lambda: (curves()[1], curves()[2]),
    }


def write_evaluation(folder, task, models, meta=None):
    payload = {"format_version": FORMAT_VERSION, "task": task, "models": models}
    payload.update(meta or {})
    try:
        from easyml import __version__

        payload["easyml_version"] = __version__
        path = os.path.join(folder, FILE_NAME)
        temp = path + ".tmp"
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, allow_nan=False, default=str)
        os.replace(temp, path)
    except Exception as exc:  # charts are a bonus; never fail the run over them
        write_log(f"Could not save result charts: {exc}", "warning", os.path.basename(__file__))


def read_evaluation(folder):
    path = os.path.join(folder or "", FILE_NAME)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
