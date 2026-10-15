"""
Running steps: full applies, and fast previews for the studio.

The preview works on the first ``PREVIEW_ROWS`` rows and keeps the result after each step prefix in
a small LRU, keyed by the file (path, size, modification time) and the steps so far. Editing step k
then only recomputes from step k. Steps never modify their input, so cached frames stay valid.
"""

import hashlib
import json
import os
import threading
from collections import OrderedDict

import pandas as pd

from easyml.dataprep.profile import profile
from easyml.dataprep.steps import STEP_SPECS, StepError, run_step, validate_step

PREVIEW_ROWS = 5000
PREVIEW_SHOWN = 50
_CACHE_SIZE = 48
_cache = OrderedDict()
_lock = threading.Lock()


def read_table(path, nrows=None):
    path = str(path)
    if path.lower().endswith((".xlsx", ".xls")):
        frame = pd.read_excel(path)
        return frame.head(nrows) if nrows else frame
    if not path.lower().endswith(".csv"):
        raise ValueError("Use a .csv or Excel file.")
    return pd.read_csv(path, nrows=nrows)


def _describe(frame):
    return {"rows": int(len(frame)), "cols": int(len(frame.columns)), "columns": [str(c) for c in frame.columns]}


def apply_steps(frame, steps):
    """
    Apply ``steps`` in order. Returns ``(frame, reports)``; a step that fails is skipped and its
    report carries the error, so one bad step never hides the others' results.
    """
    reports = []
    current = frame
    for index, raw in enumerate(steps):
        current, report = _step(current, raw, index)
        reports.append(report)
    return current, reports


def _step(current, raw, index):
    """Run one step; a step that is invalid or not filled in yet is skipped and reported, not fatal."""
    report = {"index": index, "op": raw.get("op") if isinstance(raw, dict) else None, "enabled": True,
              "columns_in": [str(c) for c in current.columns], "rows_in": int(len(current)), "note": "", "error": None}
    try:
        step = validate_step(raw)
    except ValueError as exc:
        spec = STEP_SPECS.get(report["op"]) or {}
        report.update(label=spec.get("label", str(report["op"])), error=str(exc), incomplete=True)
    else:
        spec = STEP_SPECS[step["op"]]
        report.update(label=spec["label"], enabled=step["enabled"])
        if step["enabled"]:
            current, report = _run_one(current, step, report)
    report.update(rows_out=int(len(current)), cols_out=int(len(current.columns)))
    return current, report


def _run_one(current, step, report):
    try:
        result, note = run_step(current, step)
        report["note"] = note
        return result, report
    except (StepError, ValueError, TypeError, KeyError) as exc:
        report["error"] = str(exc)
        return current, report


def _file_key(path):
    stat = os.stat(path)
    return (os.path.normcase(os.path.abspath(path)), stat.st_size, stat.st_mtime_ns)


def _steps_key(steps):
    return hashlib.blake2b(json.dumps(steps, sort_keys=True, default=str).encode("utf-8"), digest_size=16).digest()


def _cached(key):
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    return None


def _store(key, value):
    with _lock:
        _cache[key] = value
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)


def preview(path, steps, rows=PREVIEW_ROWS, shown=PREVIEW_SHOWN):
    """
    Apply ``steps`` to the first ``rows`` rows of ``path`` and describe the result:
    ``{"rows", "cols", "columns", "head", "profile", "steps", "sample_rows", "sampled"}``.
    """
    file_key = _file_key(path)
    base_key = (file_key, rows, b"")
    cached = _cached(base_key)
    if cached is None:
        base = read_table(path, nrows=rows + 1)  # one extra row tells whether the file is longer
        sampled = len(base) > rows
        base = base.head(rows)
        _store(base_key, (base, sampled))
    else:
        base, sampled = cached

    # Find the longest cached prefix, then compute the rest step by step (caching each prefix).
    frame, reports, start = base, [], 0
    for k in range(len(steps), 0, -1):
        hit = _cached((file_key, rows, _steps_key(steps[:k])))
        if hit is not None:
            frame, reports, start = hit[0], list(hit[1]), k
            break
    for index in range(start, len(steps)):
        frame, report = _step(frame, steps[index], index)
        reports.append(report)
        _store((file_key, rows, _steps_key(steps[:index + 1])), (frame, list(reports)))

    return {**_describe(frame), "head": _json_rows(frame.head(shown)), "profile": profile(frame),
            "steps": reports, "sample_rows": int(len(base)), "sampled": bool(sampled)}


def _json_rows(frame):
    """Rows for the preview table: dates written as in the cleaned CSV, missing values as None."""
    frame = frame.copy()
    for column in frame.columns:
        values = frame[column]
        if pd.api.types.is_datetime64_any_dtype(values):
            present = values.dropna()
            midnight = present.empty or bool((present == present.dt.normalize()).all())
            frame[column] = values.dt.strftime("%Y-%m-%d" if midnight else "%Y-%m-%d %H:%M:%S")
    return frame.astype(object).where(frame.notna(), None).to_dict(orient="records")


def apply_file(path, steps, output):
    """Apply ``steps`` to the whole file at ``path`` and write ``output`` (CSV). Returns ``(frame, reports)``."""
    frame, reports = apply_steps(read_table(path), steps)
    frame.to_csv(output, index=False)
    return frame, reports


def clear_cache():
    with _lock:
        _cache.clear()
