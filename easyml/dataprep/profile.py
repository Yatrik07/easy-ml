"""A short description of every column: kind, missing values, distinct values and a summary."""

import numpy as np
import pandas as pd


def column_kind(series):
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "category"
    non_null = series.dropna()
    if len(non_null) and non_null.nunique() <= max(20, int(len(non_null) * 0.05)):
        return "category"
    return "text"


def _plain(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value))
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    return value


def profile_column(series):
    rows = len(series)
    missing = int(series.isna().sum())
    non_null = series.dropna()
    kind = column_kind(series)
    info = {
        "name": str(series.name),
        "kind": kind,
        "dtype": str(series.dtype),
        "missing": missing,
        "missing_pct": round(missing / rows * 100, 1) if rows else 0.0,
        "unique": int(non_null.nunique()),
        "samples": [str(v) for v in non_null.head(5).tolist()],
    }
    if kind == "number" and len(non_null):
        info.update(min=_plain(non_null.min()), max=_plain(non_null.max()),
                    mean=_plain(round(float(non_null.mean()), 4)), median=_plain(round(float(non_null.median()), 4)))
    elif kind == "date" and len(non_null):
        info.update(min=_plain(non_null.min()), max=_plain(non_null.max()))
    elif len(non_null):
        top = non_null.astype(str).value_counts().head(5)
        info["top"] = [{"value": str(v), "count": int(c)} for v, c in top.items()]
    return info


def profile(frame):
    """``[profile_column(col) ...]`` for a DataFrame, or for a CSV/Excel path."""
    if not isinstance(frame, pd.DataFrame):
        from easyml.utils.data_loader import load_data_from_path

        frame = load_data_from_path(str(frame))
    return [profile_column(frame[c]) for c in frame.columns]
