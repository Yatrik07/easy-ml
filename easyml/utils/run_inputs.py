"""
What a run records about its inputs once training is done (Easy-ML 0.11+):

* runs trained on cleaned data get ``prep.json`` (the frozen data-prep recipe) and ``raw_data.csv`` (the
  raw rows behind ``data.csv``), copied from the side files that cleaning wrote next to the cleaned upload;
* every table and text run gets ``data_profile.json``, the profile the data check compares new data with.

Both are extra files: writing them never fails a run.
"""

import os

from easyml.utils.logger import write_log


def recipe_from_sidecars(dataset_path):
    """``{"name", "steps"}`` of the recipe that cleaned ``dataset_path``, when its side files exist."""
    from easyml.dataprep.frozen import FrozenRecipe, has_sidecars, sidecar_paths

    if not has_sidecars(dataset_path):
        return None
    try:
        frozen = FrozenRecipe.load(sidecar_paths(dataset_path)[0])
    except (OSError, ValueError):
        return None
    steps = [{"op": s["op"], "params": s.get("params") or {}} for s in frozen.steps if not s.get("skipped")]
    return {"name": frozen.data.get("name") or "Data prep steps", "steps": steps}


def record_inputs(folder, source_path, frame, features, target, recipe=None, text_column=None):
    """
    Write ``prep.json`` + ``raw_data.csv`` (when ``source_path`` was cleaned by Data prep) and
    ``data_profile.json``. ``frame`` is the training data as loaded (``data.csv``). Returns the
    FrozenRecipe or None.
    """
    from easyml.dataprep.frozen import PREP_FILE, RAW_DATA_FILE, attach_to_run, read_raw_rows
    from easyml.monitoring.datacheck import write_profile

    frozen = None
    try:
        frozen = attach_to_run(folder, source_path, features, target, name=(recipe or {}).get("name"))
    except Exception as exc:
        write_log(f"Could not save the run's data-prep recipe for predictions: {exc!r}", "warning",
                  os.path.basename(__file__))
        for name in (PREP_FILE, RAW_DATA_FILE):
            try:
                os.remove(os.path.join(folder, name))
            except OSError:
                pass
        frozen = None

    text_columns = [text_column] if text_column else []
    try:
        if frozen is not None:
            rows = read_raw_rows(os.path.join(folder, RAW_DATA_FILE), frozen)
            columns = frozen.input_columns
        else:
            rows, columns = frame, list(features)
        write_profile(folder, rows, columns, text_columns=[c for c in text_columns if c in columns])
    except Exception as exc:
        write_log(f"Could not save the data profile: {exc!r}", "warning", os.path.basename(__file__))
    return frozen
