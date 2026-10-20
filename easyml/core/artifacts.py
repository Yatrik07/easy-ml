"""
Helpers for loading the artifacts a tabular training run leaves on disk.

Shared by inference, explainability and export so they agree on file naming and input handling.
"""

import json
import os
import re
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from easyml.config.constants import CATBOOST_MODEL_NAMES
from easyml.utils.frames import is_text

_MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


def validate_model_name(model_name):
    """Reject anything that is not a plain class-like name, so it can never escape the run folder."""
    if not isinstance(model_name, str) or not _MODEL_NAME_PATTERN.match(model_name):
        raise ValueError(f"Invalid model name: {model_name!r}")
    return model_name


def model_file_path(folder_path, model_name):
    """
    Return the path of a saved tabular model.

    CatBoost models are stored natively as ``<Model>.cbm``; runs created before 0.4.0
    used ``<Model>.pkl.cbm``, which is still accepted. Runs with several targets (0.13) pickle every
    model, CatBoost included, as ``<Model>.pkl``.
    """
    validate_model_name(model_name)
    if model_name in CATBOOST_MODEL_NAMES:
        candidates = [f"{model_name}.cbm", f"{model_name}.pkl.cbm", f"{model_name}.pkl"]
    else:
        candidates = [f"{model_name}.pkl"]

    for candidate in candidates:
        path = os.path.join(folder_path, candidate)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Model '{model_name}' was not found for this run.")


def load_tabular_model(folder_path, model_name, task):
    """The saved model (shared, read-only: reused between calls while the file is unchanged)."""
    from easyml.core.model_cache import cached_load
    from easyml.core.multi_target import BEST_FOR_EACH_TARGET, best_for_each_target

    if model_name == BEST_FOR_EACH_TARGET:
        # Put together from the run's saved models, so it is cached under the run's config file.
        config_path = os.path.join(folder_path, "config_json.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError("This run has no best model for each target.")
        return cached_load("tabular_model", config_path, lambda: best_for_each_target(folder_path, task),
                           extra=(BEST_FOR_EACH_TARGET,))
    path = model_file_path(folder_path, model_name)
    is_regression = "regression" in (task or "").lower()
    return cached_load("tabular_model", path, lambda: _load_tabular_model(path, model_name, task),
                       extra=(is_regression,))


def _load_tabular_model(path, model_name, task):
    if model_name in CATBOOST_MODEL_NAMES and not path.endswith(".pkl"):
        if "regression" in task.lower():
            from catboost import CatBoostRegressor
            model = CatBoostRegressor()
        else:
            from catboost import CatBoostClassifier
            model = CatBoostClassifier()
        model.load_model(path)
        return model
    return joblib.load(path)


def load_pipeline(folder_path):
    """The fitted preprocessing pipeline (shared, read-only: reused while the file is unchanged)."""
    from easyml.core.model_cache import cached_load

    path = os.path.join(folder_path, "preprocessing_pipeline.pkl")
    return cached_load("pipeline", path, lambda: joblib.load(path))


def load_frozen_recipe(folder_path):
    """
    The run's frozen data-prep recipe (``prep.json``: runs trained on cleaned data since Easy-ML 0.11), or
    None. Shared and read-only like a model.
    """
    from easyml.dataprep.frozen import PREP_FILE, FrozenRecipe

    path = os.path.join(folder_path or "", PREP_FILE)
    if not folder_path or not os.path.exists(path):
        return None
    from easyml.core.model_cache import cached_load

    return cached_load("frozen_recipe", path, lambda: FrozenRecipe.load(path), size=os.path.getsize(path) * 4)


def read_json(folder_path, filename):
    path = os.path.join(folder_path, filename)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_target_decoder(folder_path):
    """Map encoded class index -> original label."""
    from easyml.utils.labels import decoder_from_config

    return decoder_from_config(read_json(folder_path, "config_json.json"))


def _blank_strings(series):
    """True where a value is a string made only of whitespace (non-strings are never blank)."""
    try:
        return series.str.strip().eq("").fillna(False).astype(bool)
    except AttributeError:  # no strings at all in the column
        return series.map(lambda v: isinstance(v, str) and not v.strip()).astype(bool)


@lru_cache(maxsize=256)
def _column_kinds(dtype_items):
    """(column, "b" | "i" | "f" | None) for each recorded training dtype; cached per run's column list."""
    kinds = []
    for col, dtype in dtype_items:
        try:
            resolved = pd.api.types.pandas_dtype(dtype)
        except TypeError:
            kinds.append((col, None))
            continue
        if pd.api.types.is_bool_dtype(resolved):
            kinds.append((col, "b"))
        elif pd.api.types.is_integer_dtype(resolved):
            kinds.append((col, "i"))
        elif pd.api.types.is_float_dtype(resolved):
            kinds.append((col, "f"))
        else:
            kinds.append((col, None))
    return tuple(kinds)


def coerce_feature_frame(df, column_data):
    """
    Cast raw inference input (often strings from an HTML form) to the dtypes seen at training time.

    Blank strings become NaN so the fitted imputers handle them. Integer columns stay integers when
    possible so one-hot encoders fitted on ints recognise the categories.
    """
    df = df.copy()
    for col in df.columns:
        if is_text(df[col]):
            df[col] = df[col].where(~_blank_strings(df[col]), np.nan)

    dtypes = column_data.get("column_dtypes") or {}
    if not dtypes:
        # Runs created before 0.4.0 only recorded the numeric column list.
        dtypes = {col: "float64" for col in column_data.get("numerical_columns", [])}

    for col, kind in _column_kinds(tuple(dtypes.items())):
        if col not in df.columns:
            df[col] = np.nan
            continue
        if kind is None:
            continue
        current = df[col].dtype
        # Already the type the conversion below would produce: nothing to do (saves time per request).
        if (kind == "f" and current == np.float64) or (kind == "i" and current == np.int64):
            continue
        if kind in "if":
            values = pd.to_numeric(df[col], errors="coerce")
            if kind == "i" and values.notna().all() and (values % 1 == 0).all():
                values = values.astype("int64")
            df[col] = values
        else:
            df[col] = df[col].map(
                lambda v: v if isinstance(v, bool) or pd.isna(v) else str(v).strip().lower() in ("true", "1", "yes")
            )
    return df


def flatten_predictions(predictions):
    """CatBoost returns (n, 1) arrays for multiclass; normalise every model to a flat 1-D array."""
    return np.asarray(predictions).reshape(-1)


def _python_value(value):
    """A plain JSON-friendly value (numpy scalars unwrapped, missing values as None)."""
    if hasattr(value, "item"):
        value = value.item()
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return value


def sample_row(folder_path, task, rng=None):
    """
    A random real row of a run's data, to try the model with: a row of its test set when the run saved
    one (``holdout.json``: the model never trained on it), otherwise any row of its training data.

    Returns ``{"values" | "text", "actual", "row", "source": "test" | "training"}``.
    """
    import random

    from easyml.core.model_cache import cached_load
    from easyml.utils.holdout import read_holdout

    rng = rng or random.Random()
    data_path = os.path.join(folder_path, "data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError("This run has no saved copy of its data to take a row from.")
    # Shared and cached like a model: read rows only, never modify the frame.
    frame = cached_load("sample_frame", data_path, lambda: pd.read_csv(data_path),
                        size=os.path.getsize(data_path) * 3)
    if not len(frame):
        raise ValueError("The run's data has no rows.")
    holdout = [r for r in (read_holdout(folder_path) or []) if 0 <= r < len(frame)]
    row = rng.choice(holdout) if holdout else rng.randrange(len(frame))
    config = read_json(folder_path, "config_json.json")
    target = config.get("target_column")
    result = {"row": int(row), "source": "test" if holdout else "training",
              "actual": _python_value(frame[target].iloc[row]) if target in frame.columns else None}
    if len(config.get("target_columns") or []) > 1:  # several columns to predict: {target: value}
        result["actual"] = {t: _python_value(frame[t].iloc[row]) if t in frame.columns else None
                            for t in config["target_columns"]}
    raw = _raw_rows(folder_path)
    raw_row = raw[1].iloc[row] if raw is not None and row < len(raw[1]) else None
    if task == "NLPTextClassification":
        column = config["text_column"]
        if raw_row is not None and column in raw[0].input_columns:
            text = _json_value(raw_row.get(column))  # the raw text: the model cleans it the same way
        else:
            text = _python_value(frame[column].iloc[row])
        result["text"] = "" if text is None else str(text)
        return result
    if raw_row is not None:
        # The run cleans its input: the form gets the raw row behind this cleaned row.
        result["values"] = {c: _json_value(raw_row.get(c)) for c in raw[0].input_columns}
        return result
    columns = list((read_json(folder_path, "column_data.json").get("column_dtypes") or {}).keys())
    result["values"] = {c: _python_value(frame[c].iloc[row]) if c in frame.columns else None for c in columns}
    return result


def _json_value(value):
    """Like _python_value, also for raw rows: dates become text, NaT/NaN become None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value))
    return _python_value(value)


def _raw_rows(folder_path):
    """``(frozen recipe, raw rows)`` for runs that clean their input (row i matches data.csv row i), else None."""
    from easyml.core.model_cache import cached_load

    prep = load_frozen_recipe(folder_path)
    path = os.path.join(folder_path, "raw_data.csv")
    if prep is None or not os.path.exists(path):
        return None
    from easyml.dataprep.frozen import read_raw_rows

    rows = cached_load("raw_rows", path, lambda: read_raw_rows(path, prep), size=os.path.getsize(path) * 3)
    return prep, rows
