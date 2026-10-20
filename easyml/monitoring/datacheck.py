"""
Data check: does new data look like the data a model was trained on?

Every run keeps a small profile of its input columns (``data_profile.json``): for numbers the range,
deciles and how often they are empty; for categories how often each value occurs. New data is then
checked in two ways:

* **Row by row** (any number of rows, even one): numbers outside the training range, categories the
  model never saw, and empty cells where training had none.
* **As a whole** (30 rows or more): how far each column's distribution moved, measured with the
  population stability index (PSI: below 0.1 nothing to worry about, 0.1-0.25 slightly different,
  above 0.25 clearly different), plus changes in how often cells are empty. Because PSI is noisy for
  small files, a column only counts as moved when the shift is also statistically clear for the number
  of rows checked (chi-square test, 0.1 % level).

The check only informs: predictions are made either way. Only pandas and numpy are used.
"""

import json
import math
import os

import numpy as np
import pandas as pd

from easyml.utils.frames import is_text

PROFILE_FILE = "data_profile.json"
PROFILE_VERSION = 1
MIN_ROWS = 30          # fewer rows: only row-by-row checks
TOP_CATEGORIES = 50    # categories kept per column; more different values are "free text"
EPSILON = 1e-4

PSI_MINOR, PSI_MAJOR = 0.1, 0.25
MISSING_MINOR, MISSING_MAJOR = 0.05, 0.2
SHARE_MINOR, SHARE_MAJOR = 0.05, 0.1   # share of values outside the training range or never seen
RANK = {"ok": 0, "minor": 1, "major": 2}
CHI2_Z = 3.09  # a shift must also be clear at the 0.1 % level for the number of rows checked


# --------------------------------------------------------------------------- formatting

def _fmt(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "empty"
    if isinstance(value, (int, np.integer)) or (isinstance(value, float) and float(value).is_integer() and abs(value) < 1e15):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.4g}" if abs(value) >= 1e4 or abs(value) < 1e-3 else f"{value:,.4f}".rstrip("0").rstrip(".")
    return repr(str(value)) if not isinstance(value, str) else f"'{value}'"


def _pct(share):
    return f"{share * 100:.0f}%" if share >= 0.01 or share == 0 else "under 1%"


def _is_number(series):
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


# --------------------------------------------------------------------------- the reference profile

def build_profile(frame, columns, text_columns=()):
    """Profile ``columns`` of the training data ``frame`` (``text_columns`` only get their empty share)."""
    profile = {"version": PROFILE_VERSION, "rows": int(len(frame)), "columns": {}}
    for column in columns:
        if column not in frame.columns:
            continue
        series = frame[column]
        missing = float(series.isna().mean()) if len(series) else 0.0
        if column in text_columns:
            profile["columns"][column] = {"kind": "text", "missing": missing}
            continue
        if _is_number(series):
            values = series.dropna().astype(float).to_numpy()
            entry = {"kind": "number", "missing": missing, "min": None, "max": None, "median": None,
                     "edges": [], "shares": []}
            if len(values):
                quantiles = np.quantile(values, np.linspace(0, 1, 11))
                edges = sorted(set(float(q) for q in quantiles[1:-1]))
                entry.update(min=float(values.min()), max=float(values.max()), median=float(np.median(values)),
                             edges=edges, shares=_bucket_shares(values, edges))
            profile["columns"][column] = entry
        else:
            values = series.dropna().astype(str)
            counts = values.value_counts(normalize=True)
            top = counts.head(TOP_CATEGORIES)
            profile["columns"][column] = {
                "kind": "category", "missing": missing, "unique": int(values.nunique()),
                "complete": bool(values.nunique() <= TOP_CATEGORIES),
                "shares": {str(k): float(v) for k, v in top.items()},
                "other": float(max(0.0, 1.0 - top.sum())),
            }
    return profile


def _bucket_shares(values, edges):
    """Share of ``values`` in the ranges cut by ``edges`` (one more range than edges)."""
    if not len(values):
        return []
    codes = np.searchsorted(np.asarray(edges, dtype=float), values, side="left")
    counts = np.bincount(codes, minlength=len(edges) + 1)
    return [float(c) / len(values) for c in counts]


def write_profile(folder, frame, columns, text_columns=()):
    """Save the training data's profile with a run. Never fails a run: returns None on any error."""
    try:
        profile = build_profile(frame, columns, text_columns)
        with open(os.path.join(folder, PROFILE_FILE), "w", encoding="utf-8") as handle:
            json.dump(profile, handle)
        return profile
    except Exception as exc:  # optional metadata
        from easyml.utils.logger import write_log

        write_log(f"Could not write {PROFILE_FILE}: {exc}", "warning", os.path.basename(__file__))
        return None


def reference_profile(folder):
    """
    The profile of a run's training inputs: ``data_profile.json``, or for runs from before Easy-ML 0.11
    one computed from ``data.csv`` (cached in memory, never written into the run).
    """
    path = os.path.join(folder, PROFILE_FILE)
    from easyml.core.model_cache import cached_load

    if os.path.exists(path):
        return cached_load("data_profile", path, lambda: _read_json(path), size=64 * 1024)
    data_path = os.path.join(folder, "data.csv")
    if not os.path.exists(data_path):
        return None
    return cached_load("data_profile_old", data_path, lambda: _profile_of_old_run(folder), size=64 * 1024)


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _profile_of_old_run(folder):
    from easyml.core.artifacts import read_json

    config = read_json(folder, "config_json.json")
    frame = pd.read_csv(os.path.join(folder, "data.csv"))
    if config.get("text_column"):
        return build_profile(frame, [config["text_column"]], text_columns=[config["text_column"]])
    columns = list((read_json(folder, "column_data.json").get("column_dtypes") or {}).keys())
    return build_profile(frame, columns)


# --------------------------------------------------------------------------- checks

def _numbers(series):
    return pd.to_numeric(series, errors="coerce") if not _is_number(series) else series.astype(float)


def _texts(series):
    return series.astype(str).where(series.notna(), np.nan)


def check_rows(profile, frame, limit=50):
    """
    Row-by-row findings, at most ``limit``: ``[{"row", "column", "issue", "value", "message"}]`` with
    issue "above", "below", "unseen" or "missing". Rows are numbered from 0.
    """
    findings = []
    if not profile:
        return findings
    for column, info in profile["columns"].items():
        if column not in frame.columns:
            continue
        series = frame[column].reset_index(drop=True)
        if is_text(series):
            blank = series.astype(str).str.strip().eq("") & series.notna()
            series = series.where(~blank, np.nan)
        empty = series.isna()
        if info.get("missing", 0) == 0:
            for row in np.flatnonzero(empty.to_numpy())[:limit]:
                findings.append({"row": int(row), "column": column, "issue": "missing", "value": None,
                                 "message": f"'{column}' is empty, but it was never empty in the training data."})
        if info["kind"] == "number" and info.get("min") is not None:
            values = _numbers(series)
            for row in np.flatnonzero((values > info["max"]).fillna(False).to_numpy())[:limit]:
                findings.append({"row": int(row), "column": column, "issue": "above", "value": float(values[row]),
                                 "message": f"'{column}' is {_fmt(float(values[row]))}, higher than anything in the "
                                            f"training data (highest {_fmt(info['max'])})."})
            for row in np.flatnonzero((values < info["min"]).fillna(False).to_numpy())[:limit]:
                findings.append({"row": int(row), "column": column, "issue": "below", "value": float(values[row]),
                                 "message": f"'{column}' is {_fmt(float(values[row]))}, lower than anything in the "
                                            f"training data (lowest {_fmt(info['min'])})."})
        elif info["kind"] == "category" and info.get("complete"):
            known = set(info.get("shares") or {})
            texts = _texts(series)
            unseen = texts.notna() & ~texts.isin(known)
            for row in np.flatnonzero(unseen.to_numpy())[:limit]:
                findings.append({"row": int(row), "column": column, "issue": "unseen", "value": texts[row],
                                 "message": f"'{column}' is {_fmt(texts[row])}, a value the model never saw in training."})
    findings.sort(key=lambda f: (f["row"], f["column"]))
    return findings[:limit]


def _psi(expected, actual):
    total = 0.0
    for e, a in zip(expected, actual):
        e, a = max(e, EPSILON), max(a, EPSILON)
        total += (a - e) * math.log(a / e)
    return float(total)


def _clearly_moved(expected, actual, n):
    """
    Is the difference in shares more than chance for ``n`` values? PSI alone is noisy for small files (a
    random sample of the training data itself often scores above 0.25 with 30-100 rows), so a column only
    counts as moved when a chi-square test also rejects "same distribution" at the 0.1 % level. Ranges
    holding under 2 % of the training data are pooled so rare values do not dominate.
    """
    pooled_e, pooled_a, rare_e, rare_a = [], [], 0.0, 0.0
    for e, a in zip(expected, actual):
        if e < 0.02:
            rare_e, rare_a = rare_e + e, rare_a + a
        else:
            pooled_e.append(e)
            pooled_a.append(a)
    if rare_e > 0:
        pooled_e.append(rare_e)
        pooled_a.append(rare_a)
    df = len(pooled_e) - 1
    if df < 1 or n <= 0:
        return False
    statistic = n * sum((a - e) ** 2 / max(e, EPSILON) for e, a in zip(pooled_e, pooled_a))
    # 99.9 % quantile of chi-square(df), Wilson-Hilferty approximation (z = 3.09)
    critical = df * (1 - 2 / (9 * df) + CHI2_Z * math.sqrt(2 / (9 * df))) ** 3
    return statistic > critical


def _worse(current, status):
    return status if RANK[status] > RANK[current] else current


def _check_column(column, info, series, enough):
    rows = len(series)
    entry = {"column": column, "kind": info["kind"], "status": "ok", "psi": None, "missing_train": info.get("missing", 0.0),
             "missing_new": float(series.isna().mean()) if rows else 0.0, "outside": None, "unseen": None,
             "unseen_values": [], "message": ""}
    reasons = []
    gap = entry["missing_new"] - entry["missing_train"]
    empty_moved = _clearly_moved([entry["missing_train"], 1 - entry["missing_train"]],
                                 [entry["missing_new"], 1 - entry["missing_new"]], rows)
    if enough and abs(gap) >= MISSING_MINOR and (empty_moved or entry["missing_train"] == 0):
        entry["status"] = "major" if abs(gap) >= MISSING_MAJOR else "minor"
        reasons.append(f"empty in {_pct(entry['missing_new'])} of rows (training: {_pct(entry['missing_train'])})")
    elif not enough and entry["missing_train"] == 0 and entry["missing_new"] > 0:
        entry["status"] = "minor"
        reasons.append("has empty cells, which the training data never had")

    if info["kind"] == "number" and info.get("min") is not None:
        values = _numbers(series).dropna().to_numpy(dtype=float)
        if len(values):
            outside = float(((values < info["min"]) | (values > info["max"])).mean())
            entry["outside"] = outside
            if outside >= SHARE_MAJOR or (outside > 0 and not enough):
                entry["status"] = _worse(entry["status"], "major" if outside >= SHARE_MAJOR else "minor")
            elif outside >= SHARE_MINOR:
                entry["status"] = _worse(entry["status"], "minor")
            if outside > 0 and entry["status"] != "ok":
                reasons.append(f"{_pct(outside)} of values outside the training range "
                               f"({_fmt(info['min'])} to {_fmt(info['max'])})")
            if enough and info.get("shares"):
                shares = _bucket_shares(values, info["edges"])
                entry["psi"] = _psi(info["shares"], shares)
                if entry["psi"] >= PSI_MINOR and _clearly_moved(info["shares"], shares, len(values)):
                    entry["status"] = _worse(entry["status"], "major" if entry["psi"] >= PSI_MAJOR else "minor")
                    median = float(np.median(values))
                    direction = "higher" if median > info["median"] else "lower" if median < info["median"] else "spread differently"
                    word = "much " if entry["psi"] >= PSI_MAJOR and direction != "spread differently" else ""
                    reasons.append(f"values are {word}{direction}" + (
                        f" (median {_fmt(median)}, training {_fmt(info['median'])})" if direction != "spread differently" else ""))
    elif info["kind"] == "category":
        texts = _texts(series).dropna()
        if len(texts):
            shares = info.get("shares") or {}
            if info.get("complete"):
                unseen_mask = ~texts.isin(set(shares))
                entry["unseen"] = float(unseen_mask.mean())
                entry["unseen_values"] = [str(v) for v in texts[unseen_mask].value_counts().index[:5]]
                if entry["unseen"] > 0:
                    status = "major" if entry["unseen"] >= SHARE_MAJOR else "minor"
                    entry["status"] = _worse(entry["status"], status)
                    shown = ", ".join(f"'{v}'" for v in entry["unseen_values"])
                    reasons.append(f"new values {shown} ({_pct(entry['unseen'])} of rows)")
            if enough:
                new_shares = texts.value_counts(normalize=True)
                expected = list(shares.values()) + [info.get("other", 0.0)]
                actual = [float(new_shares.get(k, 0.0)) for k in shares]
                actual.append(float(max(0.0, 1.0 - sum(actual))))
                entry["psi"] = _psi(expected, actual)
                if entry["psi"] >= PSI_MINOR and _clearly_moved(expected, actual, len(texts)):
                    entry["status"] = _worse(entry["status"], "major" if entry["psi"] >= PSI_MAJOR else "minor")
                    if not reasons or entry["unseen"] in (None, 0):
                        reasons.append("its values occur in different proportions")
    entry["message"] = "; ".join(reasons)
    return entry


def check_frame(profile, frame):
    """
    Compare ``frame`` with the training profile. Returns::

        {"rows", "status": "ok" | "minor" | "major", "summary", "columns": [...], "missing_columns": [...],
         "rows_checked_as_whole": bool}
    """
    report = {"rows": int(len(frame)), "status": "ok", "summary": "", "columns": [], "missing_columns": [],
              "rows_checked_as_whole": len(frame) >= MIN_ROWS}
    if not profile:
        report["summary"] = "This run has no record of its training data to compare with."
        return report
    enough = len(frame) >= MIN_ROWS
    for column, info in profile["columns"].items():
        if column not in frame.columns:
            report["missing_columns"].append(column)
            continue
        series = frame[column]
        if is_text(series):
            series = series.where(~(series.astype(str).str.strip().eq("") & series.notna()), np.nan)
        entry = _check_column(column, info, series, enough)
        if not enough and entry["status"] == "major":
            entry["status"] = "minor"  # a handful of rows cannot show that a whole column moved
        report["columns"].append(entry)
        report["status"] = _worse(report["status"], entry["status"])
    report["summary"] = summarize(report)
    return report


class DataCheckReport(dict):
    """The result of :func:`check_frame` (a dict) with shortcuts for Python users."""

    @property
    def ok(self):
        """True when nothing looks different from the training data."""
        return self.get("status") == "ok"

    @property
    def summary(self):
        return self.get("summary", "")

    def to_frame(self):
        """One row per checked column (status, PSI, empty shares, values outside the range or never seen)."""
        columns = ["column", "status", "message", "psi", "missing_train", "missing_new", "outside", "unseen"]
        return pd.DataFrame([{k: c.get(k) for k in columns} for c in self.get("columns", [])], columns=columns)

    def __repr__(self):
        return f"DataCheckReport(status={self.get('status')!r}, summary={self.summary!r})"


def summarize(report):
    flagged = [c for c in report["columns"] if c["status"] != "ok"]
    if not report["columns"]:
        return "None of the run's input columns are in this data."
    if not flagged:
        extra = "" if report["rows_checked_as_whole"] else " (checked row by row: too few rows to compare distributions)"
        return "The data looks like the training data" + extra + "."
    flagged.sort(key=lambda c: -RANK[c["status"]])
    parts = [f"{c['column']}: {c['message']}." for c in flagged[:4]]
    if len(flagged) > 4:
        parts.append(f"And {len(flagged) - 4} more.")
    total = len(report["columns"])
    word = "looks" if len(flagged) == 1 else "look"
    return (f"{len(flagged)} of {total} column{'s' if total != 1 else ''} {word} different from the training data. "
            + " ".join(parts))


def warnings_for(profile, frame, limit=10):
    """Short plain-language warnings for a prediction request (row findings, or column findings for big batches)."""
    if not profile:
        return []
    if len(frame) >= MIN_ROWS:
        report = check_frame(profile, frame)
        return [f"{c['column']}: {c['message']}." for c in report["columns"] if c["status"] != "ok"][:limit]
    findings = check_rows(profile, frame, limit=limit)
    if len(frame) == 1:
        return [f["message"] for f in findings]
    return [f"Row {f['row'] + 1}: {f['message']}" for f in findings]
