"""
``holdout.json``: which rows of a run's ``data.csv`` were its test set.

Scheduled retraining scores the current model and a newly trained one on the same rows the new one
never trained on, so the two are compared fairly. Training writes the file next to the models; it
is only an extra file and changes nothing else about a run. Writing it never fails a run.
"""

import json
import os

import numpy as np

from easyml.utils.logger import write_log

HOLDOUT_FILE = "holdout.json"


def kept_rows(source_rows, kept_index):
    """Compose row positions after rows were dropped: ``kept_index`` are positions in the previous frame."""
    kept = np.asarray(kept_index, dtype=np.int64)
    return kept if source_rows is None else np.asarray(source_rows, dtype=np.int64)[kept]


def write_holdout(folder, test_positions, source_rows=None):
    """Record the test rows (positions in the training frame, mapped back to ``data.csv`` rows)."""
    try:
        rows = kept_rows(source_rows, test_positions)
        with open(os.path.join(folder, HOLDOUT_FILE), "w", encoding="utf-8") as handle:
            json.dump({"version": 1, "file": "data.csv", "rows": sorted(int(r) for r in rows)}, handle)
    except Exception as exc:  # optional metadata: a run is still complete without it
        write_log(f"Could not write {HOLDOUT_FILE}: {exc}", "warning", os.path.basename(__file__))


def read_holdout(folder):
    """The test rows of a run as a list of ``data.csv`` row positions, or None for runs without the file."""
    try:
        with open(os.path.join(folder, HOLDOUT_FILE), encoding="utf-8") as handle:
            data = json.load(handle)
        rows = data.get("rows")
        return [int(r) for r in rows] if isinstance(rows, list) else None
    except (OSError, ValueError, TypeError):
        return None
