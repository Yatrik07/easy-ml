"""
Helpers for loading the artifacts a tabular training run leaves on disk.

Shared by inference, explainability and export so they agree on file naming and input handling.
"""

import json
import os
import re

import joblib
import numpy as np
import pandas as pd

from easyml.config.GLOBAL_VARS import CATBOOST_MODEL_NAMES

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
    used ``<Model>.pkl.cbm``, which is still accepted.
    """
    validate_model_name(model_name)
    if model_name in CATBOOST_MODEL_NAMES:
        candidates = [f"{model_name}.cbm", f"{model_name}.pkl.cbm"]
    else:
        candidates = [f"{model_name}.pkl"]

    for candidate in candidates:
        path = os.path.join(folder_path, candidate)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Model '{model_name}' was not found for this run.")


def load_tabular_model(folder_path, model_name, task):
    path = model_file_path(folder_path, model_name)
    if model_name in CATBOOST_MODEL_NAMES:
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
    return joblib.load(os.path.join(folder_path, "preprocessing_pipeline.pkl"))


def read_json(folder_path, filename):
    path = os.path.join(folder_path, filename)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_target_decoder(folder_path):
    """Map encoded class index -> original label."""
    encodings = read_json(folder_path, "config_json.json").get("target_encodings") or {}
    return {int(v): k for k, v in encodings.items()}


def coerce_feature_frame(df, column_data):
    """
    Cast raw inference input (often strings from an HTML form) to the dtypes seen at training time.

    Blank strings become NaN so the fitted imputers handle them. Integer columns stay integers when
    possible so one-hot encoders fitted on ints recognise the categories.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            blank = df[col].map(lambda v: isinstance(v, str) and not v.strip())
            df[col] = df[col].where(~blank, np.nan)

    dtypes = column_data.get("column_dtypes") or {}
    if not dtypes:
        # Runs created before 0.4.0 only recorded the numeric column list.
        dtypes = {col: "float64" for col in column_data.get("numerical_columns", [])}

    for col, dtype in dtypes.items():
        if col not in df.columns:
            df[col] = np.nan
            continue
        try:
            resolved = pd.api.types.pandas_dtype(dtype)
        except TypeError:
            continue
        if pd.api.types.is_bool_dtype(resolved):
            kind = "b"
        elif pd.api.types.is_integer_dtype(resolved):
            kind = "i"
        elif pd.api.types.is_float_dtype(resolved):
            kind = "f"
        else:
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
