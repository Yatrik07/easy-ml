"""
Is a newly trained run better than the one in use?

Both runs' best models predict the new run's test rows (``holdout.json``): data the new run never
trained on, and newer than anything the current run saw. Each is scored with the task's metric
(accuracy, or R² for regression). When the current run cannot predict those rows (other columns,
an image run, a run without ``holdout.json``), the runs' own stored test scores are compared instead.
"""

import os

import numpy as np

SAME_DATA = "same new data"
OWN_TEST_SETS = "own test sets"


def _record(run_id):
    from easyml.utils.database_operations import get_record_by_id

    return get_record_by_id(int(run_id))


def stored_score(run_id):
    """The run's best model's stored test score (what Run History shows)."""
    from easyml.utils.run_summary import summarize_run

    return summarize_run(_record(run_id))["best_score"]


def _normalise(values):
    import pandas as pd

    series = pd.Series(list(values))
    numbers = pd.to_numeric(series, errors="coerce")
    if numbers.notna().all():
        return numbers.astype(float).to_numpy()
    return series.astype(str).str.strip().to_numpy()


def score(task, truth, predictions):
    if task == "MLRegression":
        from sklearn.metrics import r2_score

        return float(r2_score(np.asarray(truth, dtype=float), np.asarray(predictions, dtype=float)))
    truth, predictions = _normalise(truth), _normalise(predictions)
    if truth.dtype.kind != predictions.dtype.kind:  # e.g. numbers vs text labels: compare as text
        truth, predictions = truth.astype(str), predictions.astype(str)
    return float(np.mean(truth == predictions)) if len(truth) else float("nan")


def holdout_inputs(run_id):
    """``(task, inputs, truth)`` for the run's test rows, or None when the run has no usable holdout."""
    from easyml.core.artifacts import read_json
    from easyml.retraining import IMAGE_TASK, TEXT_TASK
    from easyml.utils.data_loader import load_data_from_path
    from easyml.utils.holdout import read_holdout

    record = _record(run_id)
    task, folder = record["task"], record.get("folder_path") or ""
    rows = read_holdout(folder)
    if task == IMAGE_TASK or not rows:
        return None
    frame = load_data_from_path(os.path.join(folder, "data.csv"))
    if max(rows) >= len(frame):
        return None
    config = read_json(folder, "config_json.json")
    target = config["target_column"]
    part = frame.iloc[rows]
    if task == TEXT_TASK:
        return task, part[config["text_column"]].astype(str).tolist(), part[target].tolist()
    return task, part.drop(columns=[target]), part[target].tolist()


def holdout_raw_inputs(run_id):
    """
    The raw rows behind the run's test rows (runs trained on cleaned data keep them in ``raw_data.csv``), so a
    model that cleans its own input gets them the way it was trained: a DataFrame (tables), a list of texts
    (text runs), or None.
    """
    from easyml.core.artifacts import _raw_rows, read_json
    from easyml.retraining import IMAGE_TASK, TEXT_TASK
    from easyml.utils.holdout import read_holdout

    record = _record(run_id)
    folder = record.get("folder_path") or ""
    rows, raw = read_holdout(folder), _raw_rows(folder) if record["task"] != IMAGE_TASK else None
    if not rows or raw is None or max(rows) >= len(raw[1]):
        return None
    prep, frame = raw
    part = frame.iloc[rows]
    if record["task"] == TEXT_TASK:
        column = read_json(folder, "config_json.json").get("text_column")
        if column not in prep.input_columns:
            return None
        return ["" if v != v else str(v) for v in part[column]]
    return part[prep.input_columns]


def _predict(run_id, data, raw):
    """A run's predictions: raw rows for a model that cleans its input (when there are any), else ``data``."""
    from easyml.predictor import Predictor

    predictor = Predictor(run_id)
    if raw is not None and predictor.cleans_input:
        return predictor.predict(raw, cleaned=False)
    return predictor.predict(data)


def compare(champion_id, challenger_id):
    """
    Score both runs. Returns ``{"metric", "champion_score", "challenger_score", "compared_on", "rows", "note"}``.
    """
    from easyml.utils.run_summary import primary_metric

    task = _record(challenger_id)["task"]
    metric = primary_metric(task)["name"]
    note = ""
    inputs = None
    try:
        inputs = holdout_inputs(challenger_id)
    except Exception as exc:  # unreadable data: fall back below
        note = f"The new run's test rows could not be read ({exc})."
    if inputs is not None:
        task, data, truth = inputs
        try:
            raw = holdout_raw_inputs(challenger_id)
        except Exception:  # raw rows are a refinement: fall back to the cleaned rows
            raw = None
        try:
            challenger = score(task, truth, _predict(challenger_id, data, raw))
        except Exception as exc:
            note = f"The new model could not score its test rows ({exc})."
        else:
            try:
                champion = score(task, truth, _predict(champion_id, data, raw))
            except Exception as exc:
                note = (f"Run #{champion_id} could not predict the new data ({exc}), "
                        "so the runs' own test scores were compared.")
            else:
                return {"metric": metric, "champion_score": champion, "challenger_score": challenger,
                        "compared_on": SAME_DATA, "rows": len(truth), "note": ""}
    elif not note:
        note = "Compared the runs' own test scores."
    return {"metric": metric, "champion_score": stored_score(champion_id), "challenger_score": stored_score(challenger_id),
            "compared_on": OWN_TEST_SETS, "rows": None, "note": note}


def is_better(result, min_improvement=0.0):
    """Promote when the new run scores at least ``min_improvement`` more (ties promote the newer model)."""
    new, old = result.get("challenger_score"), result.get("champion_score")
    if new is None or (isinstance(new, float) and np.isnan(new)):
        return False
    if old is None or (isinstance(old, float) and np.isnan(old)):
        return True
    return new >= old + float(min_improvement or 0.0) - 1e-12
