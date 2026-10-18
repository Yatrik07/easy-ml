"""
Frozen recipes: a data-prep recipe plus the numbers it learned from the training data.

A run trained on cleaned data keeps ``prep.json`` (its recipe, frozen) and ``raw_data.csv`` (the raw row
behind each row of ``data.csv``, in the same order). At prediction time :meth:`FrozenRecipe.prepare`
cleans raw input with the same steps and the same learned values: fill values, outlier limits and bin
edges come from the training data, never from the rows being predicted. Steps that remove or reorder
rows are skipped, because every input row needs its prediction; rows that such a filter would have
removed from the training data are reported instead.

    prep = FrozenRecipe.load("runs/.../prep.json")
    clean, notes = prep.prepare(raw_frame)      # the columns the model was trained on

Only pandas and numpy are needed, so an exported API can use this module without the studio.
"""

import io
import json
import os

import numpy as np
import pandas as pd

FORMAT_VERSION = 1
PREP_FILE = "prep.json"
RAW_DATA_FILE = "raw_data.csv"
SIDECAR_PREP = ".prep.json"   # next to a cleaned upload: clean_<hex>.prep.json
SIDECAR_RAW = ".raw.csv"      # and clean_<hex>.raw.csv

ROW_STEPS = {"remove_duplicates", "remove_missing", "filter_rows", "sort_rows"}
STRUCTURE_STEPS = {"drop_columns", "keep_columns"}
IN_PLACE_STEPS = {"change_type", "fill_missing", "replace_values", "clean_text", "clip_outliers", "round_numbers"}
MAX_CHOICES = 50  # a text column with more different values is a free-text field in the form

LOOKS_CLEAN = "The input already has the cleaned columns, so it was used as it is."


class MissingInputError(ValueError):
    """Prediction input lacks raw columns the run's recipe needs."""


# --------------------------------------------------------------------------- values and column kinds

def _encode(value):
    """A JSON-safe copy of a learned value (fill values can be numbers, text, booleans or dates)."""
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        stamp = pd.Timestamp(value)
        return None if pd.isna(stamp) else {"timestamp": stamp.isoformat()}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _decode(value):
    if isinstance(value, dict) and "timestamp" in value:
        return pd.Timestamp(value["timestamp"])
    return value


def column_kinds(frame):
    """
    How each column was read: ``int``, ``float``, ``bool``, ``datetime``, ``boolobj`` (True/False with
    empty cells, which pandas keeps as Python booleans) or ``text``.
    """
    kinds = {}
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_bool_dtype(series):
            kinds[str(column)] = "bool"
        elif pd.api.types.is_integer_dtype(series):
            kinds[str(column)] = "int"
        elif pd.api.types.is_float_dtype(series):
            kinds[str(column)] = "float"
        elif pd.api.types.is_datetime64_any_dtype(series):
            kinds[str(column)] = "datetime"
        else:
            present = series.dropna()
            boolean = len(present) and pd.api.types.infer_dtype(present, skipna=True) == "boolean"
            kinds[str(column)] = "boolobj" if boolean else "text"
    return kinds


_TRUE = {"true", "1", "yes", "y", "t"}
_FALSE = {"false", "0", "no", "n", "f"}


def _to_bool(text):
    lowered = text.str.strip().str.lower()
    out = pd.Series(np.nan, index=text.index, dtype=object)
    out[lowered.isin(_TRUE).fillna(False).to_numpy(bool)] = True
    out[lowered.isin(_FALSE).fillna(False).to_numpy(bool)] = False
    return out


def restore_kinds(frame, kinds, blank_is_missing=False):
    """
    Give each column the kind recorded at training. Values may arrive as anything (form strings, numbers,
    a CSV read as text); they are turned into text first and then read the way training read them.
    """
    out = frame.copy()
    for column, kind in kinds.items():
        if column not in out.columns:
            continue
        series = out[column]
        text = series.astype(str).where(series.notna(), np.nan)
        if blank_is_missing:
            text = text.where(~text.str.strip().eq("").fillna(False).astype(bool), np.nan)
        if kind in ("int", "float"):
            values = pd.to_numeric(text, errors="coerce")
            if kind == "int" and values.notna().all() and (values % 1 == 0).all():
                values = values.astype("int64")
            out[column] = values
        elif kind in ("bool", "boolobj"):
            values = _to_bool(text)
            out[column] = values.astype(bool) if kind == "bool" and values.notna().all() else values
        elif kind == "datetime":
            out[column] = pd.to_datetime(text, errors="coerce", format="mixed")
        else:
            out[column] = text
    return out


def _date_format(series):
    """The format the cleaned CSV used for a date column (date only when every value is at midnight)."""
    present = series.dropna()
    if present.empty or bool((present == present.dt.normalize()).all()):
        return "%Y-%m-%d"
    if bool((present.dt.microsecond != 0).any()):
        return "%Y-%m-%d %H:%M:%S.%f"
    return "%Y-%m-%d %H:%M:%S"


# --------------------------------------------------------------------------- freezing one step

def _number_columns(frame):
    from easyml.dataprep.steps import _is_number

    return [c for c in frame.columns if _is_number(frame[c])]


def _freeze(op, before, params):
    """The resolved settings and learned values of a step that ran on ``before`` (the training data)."""
    from easyml.dataprep.steps import _as_number, _is_number

    params = dict(params)
    if op == "fill_missing":
        chosen = list(params.get("columns") or [])
        strategy = params.get("strategy") or "median"
        columns = chosen or list(before.columns)
        if strategy in ("mean", "median") and not chosen:
            columns = _number_columns(before[columns])
        values = {}
        if strategy == "value":
            raw = params.get("value", "")
            for column in columns:
                numeric = _is_number(before[column]) and str(raw).strip() != ""
                values[column] = _encode(_as_number(raw) if numeric else raw)
        elif strategy in ("mean", "median"):
            for column in columns:
                values[column] = _encode(getattr(before[column], strategy)()) if _is_number(before[column]) else None
        elif strategy == "most_frequent":
            for column in columns:
                modes = before[column].mode(dropna=True)
                values[column] = _encode(modes.iloc[0]) if len(modes) else None
        return {"strategy": strategy, "columns": columns, "values": values}
    if op == "clip_outliers":
        columns = list(params.get("columns") or []) or _number_columns(before)
        method = params.get("method") or "iqr"
        limits = {}
        for column in columns:
            series = before[column]
            if method == "iqr":
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                spread = (q3 - q1) * float(params.get("factor", 1.5))
                low, high = q1 - spread, q3 + spread
            else:
                low = series.quantile(float(params.get("lower", 1.0)) / 100)
                high = series.quantile(float(params.get("upper", 99.0)) / 100)
            limits[column] = [_encode(low), _encode(high)]
        return {"columns": columns, "limits": limits}
    if op == "bin_numbers":
        series = before[params["column"]]
        bins = max(2, int(params.get("bins", 5)))
        if (params.get("method") or "equal_width") == "quantile":
            binned = pd.qcut(series, q=bins, duplicates="drop")
        else:
            binned = pd.cut(series, bins=bins)
        categories = binned.cat.categories
        name = str(params.get("new_column") or "").strip() or f"{params['column']}_bin"
        return {"column": params["column"], "new_column": name,
                "rights": [float(iv.right) for iv in categories],
                "labels": [str(iv) for iv in categories]}
    if op == "clean_text":
        columns = list(params.get("columns") or []) or [c for c in before.columns if before[c].dtype == object]
        return {"params": {**params, "columns": columns}}
    if op == "round_numbers":
        columns = list(params.get("columns") or []) or _number_columns(before)
        return {"params": {**params, "columns": columns}}
    if op in ("remove_duplicates", "remove_missing"):
        return {"params": {**params, "columns": list(params.get("columns") or []) or list(before.columns)}}
    if op == "rename_columns":
        mapping = {k: str(v).strip() for k, v in (params.get("mapping") or {}).items() if str(v).strip()}
        return {"params": {**params, "mapping": mapping}}
    return {"params": params}


# --------------------------------------------------------------------------- columns a step reads and writes

def _new_columns(step):
    """Columns a step creates: date parts, a bin column or a combined column."""
    op, params = step["op"], step.get("params") or {}
    frozen = step.get("frozen") or {}
    if op == "date_parts":
        return {f"{params['column']}_{part}" for part in (params.get("parts") or ("year", "month", "day"))}
    if op == "bin_numbers":
        return {frozen.get("new_column") or str(params.get("new_column") or "").strip() or f"{params['column']}_bin"}
    if op == "combine_columns":
        symbol = {"add": "+", "subtract": "-", "multiply": "x", "divide": "/", "concat": "&"}.get(params.get("operation"), "+")
        name = str(params.get("new_column") or "").strip() or f"{params['left']}_{symbol}_{params['right']}".replace("/", "per")
        return {name}
    return set()


def _sources(step):
    op, params = step["op"], step.get("params") or {}
    if op in ("date_parts", "bin_numbers"):
        return {params["column"]}
    if op == "combine_columns":
        return {params["left"], params["right"]}
    return set()


def _step_columns(step):
    """The columns an in-place step changes (resolved at training)."""
    op, frozen = step["op"], step.get("frozen") or {}
    if op in ("fill_missing", "clip_outliers"):
        return list(frozen.get("columns") or [])
    params = frozen.get("params") or step.get("params") or {}
    if op == "replace_values":
        return [params.get("column")]
    return list(params.get("columns") or [])


# --------------------------------------------------------------------------- replaying one step

def _replay(frame, step, flags):
    from easyml.dataprep.steps import STEP_SPECS, _row_mask

    op = step["op"]
    frozen = step.get("frozen") or {}
    params = dict(frozen.get("params") or step.get("params") or {})

    if op in ROW_STEPS:
        # Every input row keeps its prediction; rows a training filter would have removed are reported.
        if op == "filter_rows" and params.get("column") in frame.columns:
            try:
                mask = _row_mask(frame[params["column"]], params.get("operator") or "eq", params.get("value", ""))
            except (ValueError, TypeError):
                return frame
            removed = ~mask if (params.get("action") or "keep") == "keep" else mask
            if bool(removed.any()):
                flags.append({"step": step.get("label") or op, "rows": [int(i) for i in np.flatnonzero(removed.to_numpy())]})
        elif op == "remove_missing":
            columns = [c for c in params.get("columns") or [] if c in frame.columns]
            if columns:
                empty = frame[columns].isna()
                removed = empty.all(axis=1) if params.get("how") == "all" else empty.any(axis=1)
                if bool(removed.any()):
                    flags.append({"step": step.get("label") or op,
                                  "rows": [int(i) for i in np.flatnonzero(removed.to_numpy())]})
        return frame

    if op == "fill_missing":
        strategy = frozen.get("strategy")
        columns = [c for c in frozen.get("columns") or [] if c in frame.columns]
        if strategy == "previous":
            return frame.assign(**{c: frame[c].ffill() for c in columns})
        if strategy == "next":
            return frame.assign(**{c: frame[c].bfill() for c in columns})
        changes = {}
        for column in columns:
            value = (frozen.get("values") or {}).get(column)
            if value is not None and frame[column].isna().any():
                changes[column] = frame[column].fillna(_decode(value))
        return frame.assign(**changes) if changes else frame

    if op == "clip_outliers":
        changes = {}
        for column, (low, high) in (frozen.get("limits") or {}).items():
            if column in frame.columns:
                series = frame[column]
                if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                    series = pd.to_numeric(series, errors="coerce")
                changes[column] = series.clip(_decode(low), _decode(high))
        return frame.assign(**changes) if changes else frame

    if op == "bin_numbers":
        column = frozen.get("column") or params.get("column")
        if column not in frame.columns:
            return frame
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        rights = np.asarray(frozen["rights"], dtype=float)
        labels = np.asarray(frozen["labels"], dtype=object)
        # Ranges are closed on the right: the first edge >= x names the range. Values beyond the
        # training range go into the lowest or highest range.
        codes = np.clip(np.searchsorted(rights, values, side="left"), 0, len(rights) - 1)
        result = np.where(np.isnan(values), np.nan, labels[codes])
        return frame.assign(**{frozen["new_column"]: pd.Series(result, index=frame.index, dtype=object)})

    # Steps without learned values: run the step itself, limited to the columns this input has.
    if op in ("drop_columns", "keep_columns", "change_type", "clean_text", "round_numbers"):
        present = [c for c in params.get("columns") or [] if c in frame.columns]
        if not present and op != "keep_columns":
            return frame
        params["columns"] = present
    elif op == "rename_columns":
        params["mapping"] = {k: v for k, v in (params.get("mapping") or {}).items() if k in frame.columns}
        if not params["mapping"]:
            return frame
    elif op == "replace_values" and params.get("column") not in frame.columns:
        return frame
    elif op in ("date_parts", "bin_numbers", "combine_columns") and not _sources(step) <= set(frame.columns):
        return frame
    spec = STEP_SPECS[op]
    kwargs = {f["name"]: params.get(f["name"], f.get("default")) for f in spec["fields"]}
    result, _ = spec["fn"](frame, **{k: v for k, v in kwargs.items() if v is not None})
    return result


# --------------------------------------------------------------------------- the recipe

class FrozenRecipe:
    """A recipe with everything it learned from the training data (see the module docstring)."""

    def __init__(self, data):
        self.data = data

    # ------------------------------------------------------------------ properties

    @property
    def name(self):
        return self.data.get("name") or "data-prep recipe"

    @property
    def steps(self):
        return self.data.get("steps") or []

    @property
    def input_columns(self):
        """The raw columns prediction input needs (the target and unused columns are not among them)."""
        return list(self.data.get("input_columns") or [])

    @property
    def features(self):
        return list(self.data.get("features") or [])

    @property
    def inputs(self):
        """Form details per input column: ``{"type": "numeric" | "categorical" | "text", "values": [...]}``."""
        return dict(self.data.get("inputs") or {})

    def summary(self):
        """``{"name", "steps": [labels], "applied": n}`` for the studio and the report."""
        used = [s for s in self.steps if not s.get("skipped")]
        return {"name": self.name, "steps": [s.get("label") or s["op"] for s in used],
                "applied": sum(1 for s in used if s.get("needed"))}

    # ------------------------------------------------------------------ training side

    @classmethod
    def fit(cls, raw, steps, name=""):
        """
        Apply ``steps`` to the training frame ``raw`` exactly like the Data prep page does and remember
        what each step learned. Returns ``(cleaned_frame, reports, frozen_recipe)``.
        """
        from easyml.dataprep.engine import _step
        from easyml.dataprep.steps import validate_step

        current, reports, frozen_steps = raw, [], []
        for index, raw_step in enumerate(steps):
            before = current
            current, report = _step(current, raw_step, index)
            reports.append(report)
            entry = {"index": index, "op": report.get("op"), "label": report.get("label") or report.get("op")}
            if report.get("error") or report.get("incomplete") or not report.get("enabled", True):
                entry.update(skipped=True, params=dict((raw_step or {}).get("params") or {})
                             if isinstance(raw_step, dict) else {})
            else:
                step = validate_step(raw_step)
                entry.update(params=step["params"], frozen=_freeze(step["op"], before, step["params"]))
            frozen_steps.append(entry)

        date_formats = {str(c): _date_format(current[c]) for c in current.columns
                        if pd.api.types.is_datetime64_any_dtype(current[c])}
        data = {"format_version": FORMAT_VERSION, "name": name or "", "steps": frozen_steps,
                "raw_columns": [str(c) for c in raw.columns], "raw_kinds": column_kinds(raw),
                "date_formats": date_formats, "clean_kinds": {}, "features": [], "input_columns": [],
                "target": None, "inputs": {}, "example": {}}
        return current, reports, cls(data)

    def set_clean_kinds(self, cleaned_as_read):
        """Record how the cleaned file reads back (``pd.read_csv`` of it), which is what the model saw."""
        self.data["clean_kinds"] = column_kinds(cleaned_as_read)

    def finalize(self, features, target=None, raw_rows=None):
        """
        Work out which raw columns the model's ``features`` come from, and describe them for the form.
        ``raw_rows`` (the training rows as read from ``raw_data.csv``) gives the choices and an example.
        """
        needed = {str(f) for f in features}
        for step in reversed(self.steps):
            step["needed"] = False
            op = step.get("op")
            if step.get("skipped") or op in ROW_STEPS or op in STRUCTURE_STEPS:
                continue
            if op == "rename_columns":
                mapping = (step.get("frozen") or {}).get("params", {}).get("mapping") or {}
                back = {new: old for old, new in mapping.items()}
                hit = needed & set(back)
                step["needed"] = bool(hit)
                needed = (needed - hit) | {back[n] for n in hit}
            elif op in IN_PLACE_STEPS:
                step["needed"] = any(c in needed for c in _step_columns(step))
            else:  # date parts, bins and combined columns
                created = _new_columns(step)
                if created & needed:
                    step["needed"] = True
                    needed = (needed - created) | _sources(step)
        raw_order = self.data.get("raw_columns") or []
        inputs = [c for c in raw_order if c in needed and c != target]
        self.data.update(features=[str(f) for f in features], target=target,
                         input_columns=inputs + sorted(c for c in needed - set(inputs) if c != target))
        if raw_rows is not None:
            self.data["inputs"], self.data["example"] = self._describe_inputs(raw_rows)
        return self

    def _describe_inputs(self, raw_rows):
        kinds = self.data.get("raw_kinds") or {}
        rows = restore_kinds(raw_rows, kinds)
        inputs, example = {}, {}
        for column in self.input_columns:
            kind = kinds.get(column, "text")
            series = rows[column] if column in rows.columns else pd.Series(dtype=object)
            if kind in ("int", "float"):
                inputs[column] = {"type": "numeric"}
            elif kind in ("bool", "boolobj"):
                inputs[column] = {"type": "categorical", "values": [True, False]}
            else:
                values = series.dropna()
                distinct = values.unique().tolist() if kind != "datetime" else []
                if 0 < len(distinct) <= MAX_CHOICES:
                    inputs[column] = {"type": "categorical", "values": [_encode(v) for v in distinct]}
                else:
                    inputs[column] = {"type": "text"}
            first = series.dropna()
            example[column] = _encode(first.iloc[0]) if len(first) else None
            if kind == "datetime" and isinstance(example[column], dict):
                example[column] = str(first.iloc[0])
        return inputs, example

    # ------------------------------------------------------------------ prediction side

    def prepare(self, df, cleaned=None):
        """
        Clean prediction input the way the training data was cleaned.

        ``cleaned=None`` decides from the columns: raw input (every needed raw column present) is cleaned;
        input that already has the model's columns but not the raw ones is used as it is (a note says so).
        ``cleaned=True`` uses the input as it is; ``cleaned=False`` always cleans it.

        Returns ``(frame, notes)``: the model's columns, and a list of plain-language notes.
        Raises MissingInputError when needed raw columns are missing.
        """
        notes = []
        if cleaned is True:
            return df, notes
        present = set(map(str, df.columns))
        missing = [c for c in self.input_columns if c not in present]
        if missing:
            if cleaned is None and all(f in present for f in self.features):
                return df, [LOOKS_CLEAN]
            raise MissingInputError(
                f"The input is missing {len(missing)} column(s): {', '.join(missing)}. This model cleans its input "
                f"with the data-prep recipe '{self.name}', so it needs the original columns: "
                f"{', '.join(self.input_columns)}.")
        frame = restore_kinds(df[self.input_columns].reset_index(drop=True), self.data.get("raw_kinds") or {},
                              blank_is_missing=True)
        flags = []
        for step in self.steps:
            if step.get("skipped"):
                continue
            if step.get("needed") or step.get("op") in ("filter_rows", "remove_missing"):
                frame = _replay(frame, step, flags)
        frame = self._as_written(frame)
        absent = [f for f in self.features if f not in frame.columns]
        if absent:
            raise MissingInputError(f"Cleaning the input did not produce the column(s) {', '.join(absent)}.")
        for flag in flags:
            rows = flag["rows"]
            shown = ", ".join(str(r + 1) for r in rows[:5]) + (" ..." if len(rows) > 5 else "")
            notes.append(f"{len(rows)} row(s) ({shown}) would have been removed by the step '{flag['step']}' "
                         "when the training data was cleaned; the model has not seen rows like them.")
        frame = frame[self.features]
        frame.index = df.index
        return frame, notes

    def _as_written(self, frame):
        """Values as the cleaned CSV holds them, read back the way training read ``data.csv``."""
        out = frame.copy()
        for column, fmt in (self.data.get("date_formats") or {}).items():
            if column in out.columns and pd.api.types.is_datetime64_any_dtype(out[column]):
                out[column] = out[column].dt.strftime(fmt)
        buffer = io.StringIO()
        out.to_csv(buffer, index=False)
        buffer.seek(0)
        back = pd.read_csv(buffer, dtype=str, keep_default_na=True)
        bools = {c: k for c, k in (self.data.get("clean_kinds") or {}).items() if k == "boolobj"}
        return restore_kinds(back, bools) if bools else back

    # ------------------------------------------------------------------ storage

    def to_dict(self):
        return self.data

    def save(self, path):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.data, handle, indent=1)
        return path

    @classmethod
    def from_dict(cls, data):
        """Check a stored frozen recipe (e.g. from an imported archive) before using it."""
        from easyml.dataprep.steps import validate_step

        if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
            raise ValueError("Not a frozen data-prep recipe.")
        if int(data.get("format_version", 1)) > FORMAT_VERSION:
            raise ValueError("This run was cleaned by a newer Easy-ML. Upgrade easy-ml-studio to use it.")
        for step in data["steps"]:
            if not isinstance(step, dict):
                raise ValueError("A step of the frozen recipe is not valid.")
            if not step.get("skipped"):
                validate_step({"op": step.get("op"), "params": step.get("params") or {}})
        return cls(data)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


# --------------------------------------------------------------------------- files

def sidecar_paths(cleaned_path):
    """``(prep, raw)`` side files next to a cleaned upload ``clean_<hex>.csv``."""
    stem = os.path.splitext(str(cleaned_path))[0]
    return stem + SIDECAR_PREP, stem + SIDECAR_RAW


def write_sidecars(cleaned_path, frozen, raw_rows):
    """Save the frozen recipe and the raw rows behind each cleaned row next to the cleaned file."""
    prep_path, raw_path = sidecar_paths(cleaned_path)
    frozen.set_clean_kinds(pd.read_csv(cleaned_path))
    raw_rows.to_csv(raw_path, index=False)
    frozen.save(prep_path)
    return prep_path, raw_path


def has_sidecars(path):
    return bool(path) and all(os.path.isfile(p) for p in sidecar_paths(path))


def read_raw_rows(path, frozen):
    """``raw_data.csv`` (or a raw side file) as text, restored to the kinds the raw training data had."""
    rows = pd.read_csv(path, dtype=str, keep_default_na=True)
    return restore_kinds(rows, frozen.data.get("raw_kinds") or {})


def attach_to_run(folder, cleaned_path, features, target=None, name=None):
    """
    Give a run trained on ``cleaned_path`` its ``prep.json`` and ``raw_data.csv``, from the side files that
    cleaning wrote. Returns the FrozenRecipe, or None when the file has no side files.
    """
    import shutil

    if not has_sidecars(cleaned_path):
        return None
    prep_path, raw_path = sidecar_paths(cleaned_path)
    frozen = FrozenRecipe.load(prep_path)
    if name:
        frozen.data["name"] = name
    target_path = os.path.join(folder, RAW_DATA_FILE)
    shutil.copyfile(raw_path, target_path)
    frozen.finalize(features, target, read_raw_rows(target_path, frozen))
    frozen.save(os.path.join(folder, PREP_FILE))
    return frozen


def remove_sidecars(cleaned_path):
    for path in sidecar_paths(cleaned_path):
        try:
            os.remove(path)
        except OSError:
            pass
