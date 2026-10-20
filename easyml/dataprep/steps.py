"""
The data-prep steps: each is a vectorised pandas function ``fn(frame, **params) -> (frame, note)``.

Every step returns a new frame and never modifies its input (prefix results are cached for the live
preview). ``STEP_SPECS`` describes each step's parameters; the studio builds its forms from it and
:func:`validate_step` checks recipes against it. No step evaluates user-written code.
"""

import numpy as np
import pandas as pd

from easyml.utils.frames import is_text

TRUE_WORDS = {"true", "yes", "y", "1", "t"}
FALSE_WORDS = {"false", "no", "n", "0", "f"}


class StepError(ValueError):
    """A step cannot run on this data (e.g. a column it needs is missing)."""


# --------------------------------------------------------------------------- helpers

def _need(frame, columns):
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise StepError("Column(s) not found: " + ", ".join(map(str, missing)))


def _columns(frame, params, key="columns", default_all=False):
    columns = list(params.get(key) or [])
    if not columns and default_all:
        return list(frame.columns)
    _need(frame, columns)
    return columns


def _is_number(series):
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _plural(n, word):
    return f"{n:,} {word}{'' if n == 1 else 's'}"


def _as_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise StepError(f"{value!r} is not a number.")


# --------------------------------------------------------------------------- columns

def drop_columns(frame, columns=()):
    _need(frame, columns)
    return frame.drop(columns=list(columns)), f"Removed {_plural(len(columns), 'column')}"


def keep_columns(frame, columns=()):
    _need(frame, columns)
    removed = len(frame.columns) - len(columns)
    return frame[list(columns)], f"Kept {_plural(len(columns), 'column')}, removed {removed}"


def rename_columns(frame, mapping=None):
    mapping = {k: str(v).strip() for k, v in (mapping or {}).items() if str(v).strip()}
    _need(frame, list(mapping))
    new_names = [mapping.get(c, c) for c in frame.columns]
    if len(set(new_names)) != len(new_names):
        raise StepError("Two columns would get the same name.")
    return frame.rename(columns=mapping), f"Renamed {_plural(len(mapping), 'column')}"


def _to_boolean(series):
    text = series.astype("string").str.strip().str.lower()
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    result[text.isin(TRUE_WORDS).fillna(False).to_numpy(bool)] = True
    result[text.isin(FALSE_WORDS).fillna(False).to_numpy(bool)] = False
    if pd.api.types.is_bool_dtype(series):
        result = series.astype("boolean")
    return result


def change_type(frame, columns=(), to="number"):
    _need(frame, columns)
    changes, failed = {}, 0
    for column in columns:
        series = frame[column]
        if to == "number":
            # plain str (not pandas' nullable "string") so the result is an ordinary float64/int64 column
            source = series.astype(str).str.strip() if is_text(series) else series
            new = pd.to_numeric(source, errors="coerce")
        elif to == "date":
            new = pd.to_datetime(series, errors="coerce", format="mixed")
        elif to == "boolean":
            new = _to_boolean(series)
        elif to == "category":
            new = series.astype("category")
        elif to == "text":
            new = series.astype("string").astype(object).where(series.notna(), np.nan)
        else:
            raise StepError(f"Unknown type {to!r}.")
        failed += int((series.notna() & pd.isna(new)).sum())
        changes[column] = new
    note = f"Changed {_plural(len(columns), 'column')} to {to}"
    if failed:
        note += f"; {_plural(failed, 'value')} could not be converted and became empty"
    return frame.assign(**changes), note


# --------------------------------------------------------------------------- rows

def remove_duplicates(frame, columns=(), keep="first"):
    subset = _columns(frame, {"columns": columns}) or None
    result = frame.drop_duplicates(subset=subset, keep=keep if keep in ("first", "last") else "first")
    return result, f"Removed {_plural(len(frame) - len(result), 'duplicate row')}"


def remove_missing(frame, columns=(), how="any"):
    subset = _columns(frame, {"columns": columns}) or None
    result = frame.dropna(subset=subset, how="all" if how == "all" else "any")
    return result, f"Removed {_plural(len(frame) - len(result), 'row')} with missing values"


OPERATORS = ("eq", "ne", "gt", "ge", "lt", "le", "contains", "starts_with", "ends_with", "is_empty", "not_empty",
             "in")


def _row_mask(series, operator, value):
    if operator == "is_empty":
        empty = series.isna()
        if is_text(series):
            empty |= series.astype("string").str.strip().eq("").fillna(False)
        return empty
    if operator == "not_empty":
        return ~_row_mask(series, "is_empty", value)
    if operator in ("contains", "starts_with", "ends_with"):
        text = series.astype("string")
        needle = str(value)
        method = {"contains": lambda s: s.str.contains(needle, case=False, regex=False),
                  "starts_with": lambda s: s.str.lower().str.startswith(needle.lower()),
                  "ends_with": lambda s: s.str.lower().str.endswith(needle.lower())}[operator]
        return method(text).fillna(False).astype(bool)
    if operator == "in":
        options = [v.strip() for v in str(value).split(",") if v.strip()]
        if _is_number(series):
            return series.isin([_as_number(v) for v in options])
        return series.astype("string").isin(options).fillna(False).astype(bool)
    if _is_number(series):
        target = _as_number(value)
    elif pd.api.types.is_datetime64_any_dtype(series):
        target = pd.to_datetime(value, errors="coerce")
        if pd.isna(target):
            raise StepError(f"{value!r} is not a date.")
    else:
        series, target = series.astype("string"), str(value)
    compare = {"eq": series.eq, "ne": series.ne, "gt": series.gt, "ge": series.ge, "lt": series.lt, "le": series.le}
    if operator not in compare:
        raise StepError(f"Unknown condition {operator!r}.")
    return compare[operator](target).fillna(False).astype(bool)


def filter_rows(frame, column=None, operator="eq", value="", action="keep"):
    _need(frame, [column])
    mask = _row_mask(frame[column], operator, value)
    result = frame[mask] if action == "keep" else frame[~mask]
    return result, f"Removed {_plural(len(frame) - len(result), 'row')}, kept {len(result):,}"


def sort_rows(frame, column=None, ascending=True):
    _need(frame, [column])
    return frame.sort_values(column, ascending=bool(ascending), kind="stable"), f"Sorted by {column}"


# --------------------------------------------------------------------------- values

def fill_missing(frame, columns=(), strategy="value", value=""):
    chosen = _columns(frame, {"columns": columns})
    columns = chosen or list(frame.columns)
    if strategy in ("mean", "median") and not chosen:
        columns = [c for c in columns if _is_number(frame[c])]  # "all columns" means all number columns here
    if strategy == "value" and str(value).strip() == "":
        raise StepError("Enter the value to fill in.")
    changes, filled = {}, 0
    for column in columns:
        series = frame[column]
        missing = int(series.isna().sum())
        if not missing:
            continue
        if strategy == "value":
            fill = value
            if _is_number(series) and str(value).strip() != "":
                fill = _as_number(value)
            new = series.fillna(fill)
        elif strategy in ("mean", "median"):
            if not _is_number(series):
                raise StepError(f"'{column}' is not a number column; use 'most frequent' or a fixed value.")
            new = series.fillna(getattr(series, strategy)())
        elif strategy == "most_frequent":
            modes = series.mode(dropna=True)
            new = series.fillna(modes.iloc[0]) if len(modes) else series
        elif strategy == "previous":
            new = series.ffill()
        elif strategy == "next":
            new = series.bfill()
        else:
            raise StepError(f"Unknown fill strategy {strategy!r}.")
        filled += missing - int(new.isna().sum())
        changes[column] = new
    return frame.assign(**changes), f"Filled {_plural(filled, 'missing value')}"


def replace_values(frame, column=None, mapping=None):
    _need(frame, [column])
    mapping = dict(mapping or {})
    series = frame[column]
    if _is_number(series):
        numeric = {}
        for old, new in mapping.items():
            try:
                numeric[float(old)] = float(new) if str(new).strip() != "" else np.nan
            except (TypeError, ValueError):
                raise StepError(f"'{column}' holds numbers; {old!r} -> {new!r} is not a number replacement.")
        new_series = series.replace(numeric)
    else:
        # an empty replacement means "make it empty"
        new_series = series.replace({k: (np.nan if str(v) == "" else v) for k, v in mapping.items()})
    changed = int((series.ne(new_series) & ~(series.isna() & new_series.isna())).sum())
    return frame.assign(**{column: new_series}), f"Replaced {_plural(changed, 'value')}"


def clean_text(frame, columns=(), trim=True, case="none", collapse_spaces=False):
    columns = _columns(frame, {"columns": columns}, default_all=False)
    if not columns:
        columns = [c for c in frame.columns if is_text(frame[c])]
    changes = {}
    for column in columns:
        series = frame[column]
        try:
            # .str gives NaN for anything that is not a string, so numbers in a mixed column stay as they were
            text = series.str.strip() if trim else series.str.slice(0)
        except AttributeError:
            continue  # no text in this column
        if collapse_spaces:
            text = text.str.replace(r"\s+", " ", regex=True)
        if case in ("lower", "upper", "title"):
            text = getattr(text.str, case)()
        changes[column] = text.where(text.notna(), series)
    return frame.assign(**changes), f"Cleaned text in {_plural(len(changes), 'column')}"


def clip_outliers(frame, columns=(), method="iqr", factor=1.5, lower=1.0, upper=99.0):
    columns = _columns(frame, {"columns": columns}, default_all=False) or \
        [c for c in frame.columns if _is_number(frame[c])]
    changes, clipped = {}, 0
    for column in columns:
        series = frame[column]
        if not _is_number(series):
            raise StepError(f"'{column}' is not a number column.")
        if method == "iqr":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            spread = (q3 - q1) * float(factor)
            low, high = q1 - spread, q3 + spread
        else:
            low, high = series.quantile(float(lower) / 100), series.quantile(float(upper) / 100)
        new = series.clip(low, high)
        clipped += int((series.ne(new) & series.notna()).sum())
        changes[column] = new
    return frame.assign(**changes), f"Limited {_plural(clipped, 'extreme value')}"


def round_numbers(frame, columns=(), decimals=2):
    columns = _columns(frame, {"columns": columns}, default_all=False) or \
        [c for c in frame.columns if _is_number(frame[c])]
    for column in columns:
        if not _is_number(frame[column]):
            raise StepError(f"'{column}' is not a number column.")
    decimals = int(decimals)
    return frame.assign(**{c: frame[c].round(decimals) for c in columns}), \
        f"Rounded {_plural(len(columns), 'column')} to {decimals} decimals"


# --------------------------------------------------------------------------- new columns

DATE_PARTS = ("year", "month", "day", "weekday", "hour", "minute", "dayofyear", "quarter")


def date_parts(frame, column=None, parts=("year", "month", "day"), drop_original=False):
    _need(frame, [column])
    series = frame[column]
    dates = series if pd.api.types.is_datetime64_any_dtype(series) else \
        pd.to_datetime(series, errors="coerce", format="mixed")
    new = {}
    for part in parts or ():
        if part not in DATE_PARTS:
            raise StepError(f"Unknown date part {part!r}.")
        values = dates.dt.dayofweek if part == "weekday" else getattr(dates.dt, part)
        new[f"{column}_{part}"] = values.astype("Int64")
    result = frame.assign(**new)
    if drop_original:
        result = result.drop(columns=[column])
    unparsed = int((series.notna() & dates.isna()).sum())
    note = f"Added {_plural(len(new), 'column')}"
    if unparsed:
        note += f"; {_plural(unparsed, 'value')} {'was not a date' if unparsed == 1 else 'were not dates'}"
    return result, note


def bin_numbers(frame, column=None, bins=5, method="equal_width", new_column=""):
    _need(frame, [column])
    series = frame[column]
    if not _is_number(series):
        raise StepError(f"'{column}' is not a number column.")
    bins = max(2, int(bins))
    if method == "quantile":
        binned = pd.qcut(series, q=bins, duplicates="drop")
    else:
        binned = pd.cut(series, bins=bins)
    name = str(new_column).strip() or f"{column}_bin"
    labels = binned.astype("string").astype(object).where(binned.notna(), np.nan)
    return frame.assign(**{name: labels}), f"Added '{name}' with {binned.cat.categories.size} ranges"


COMBINE_OPERATIONS = ("add", "subtract", "multiply", "divide", "concat")


def combine_columns(frame, left=None, right=None, operation="add", new_column="", separator=" "):
    _need(frame, [left, right])
    a, b = frame[left], frame[right]
    if operation == "concat":
        values = a.astype("string").fillna("") + str(separator) + b.astype("string").fillna("")
        values = values.astype(object).where(a.notna() | b.notna(), np.nan)
    else:
        if not (_is_number(a) and _is_number(b)):
            raise StepError("Both columns must hold numbers for arithmetic; use 'join as text' otherwise.")
        if operation == "add":
            values = a + b
        elif operation == "subtract":
            values = a - b
        elif operation == "multiply":
            values = a * b
        elif operation == "divide":
            values = (a / b.where(b != 0)).astype(float)
        else:
            raise StepError(f"Unknown operation {operation!r}.")
    symbol = {"add": "+", "subtract": "-", "multiply": "x", "divide": "/", "concat": "&"}[operation]
    name = str(new_column).strip() or f"{left}_{symbol}_{right}".replace("/", "per")
    return frame.assign(**{name: values}), f"Added '{name}'"


# --------------------------------------------------------------------------- registry

# field types: columns (several), column (one), choice, text, number, bool, mapping, parts
STEP_SPECS = {
    "drop_columns": {"label": "Remove columns", "category": "Columns", "fn": drop_columns,
                     "help": "Leave out columns you don't need, such as IDs.",
                     "fields": [{"name": "columns", "type": "columns", "label": "Columns", "required": True}]},
    "keep_columns": {"label": "Keep only some columns", "category": "Columns", "fn": keep_columns,
                     "help": "Keep the chosen columns and remove all the others.",
                     "fields": [{"name": "columns", "type": "columns", "label": "Columns", "required": True}]},
    "rename_columns": {"label": "Rename columns", "category": "Columns", "fn": rename_columns,
                       "help": "Give columns clearer names.",
                       "fields": [{"name": "mapping", "type": "mapping", "label": "Old name → new name",
                                   "key": "column", "required": True}]},
    "change_type": {"label": "Change column type", "category": "Columns", "fn": change_type,
                    "help": "Read text as numbers, dates or yes/no values. Values that don't fit become empty.",
                    "fields": [{"name": "columns", "type": "columns", "label": "Columns", "required": True},
                               {"name": "to", "type": "choice", "label": "New type", "default": "number",
                                "options": [["number", "Number"], ["date", "Date"], ["boolean", "Yes / no"],
                                            ["category", "Category"], ["text", "Text"]]}]},
    "remove_duplicates": {"label": "Remove duplicate rows", "category": "Rows", "fn": remove_duplicates,
                          "help": "Keep one copy of rows that repeat (optionally only judged by some columns).",
                          "fields": [{"name": "columns", "type": "columns", "label": "Judge by columns (all if empty)"},
                                     {"name": "keep", "type": "choice", "label": "Keep", "default": "first",
                                      "options": [["first", "First copy"], ["last", "Last copy"]]}]},
    "remove_missing": {"label": "Remove rows with missing values", "category": "Rows", "fn": remove_missing,
                       "help": "Drop rows that have empty cells.",
                       "fields": [{"name": "columns", "type": "columns", "label": "Look at columns (all if empty)"},
                                  {"name": "how", "type": "choice", "label": "Remove a row when", "default": "any",
                                   "options": [["any", "Any of them is empty"], ["all", "All of them are empty"]]}]},
    "filter_rows": {"label": "Filter rows", "category": "Rows", "fn": filter_rows,
                    "help": "Keep or remove rows that match a condition.",
                    "fields": [{"name": "column", "type": "column", "label": "Column", "required": True},
                               {"name": "operator", "type": "choice", "label": "Condition", "default": "eq",
                                "options": [["eq", "is equal to"], ["ne", "is not equal to"], ["gt", "is greater than"],
                                            ["ge", "is at least"], ["lt", "is less than"], ["le", "is at most"],
                                            ["contains", "contains"], ["starts_with", "starts with"],
                                            ["ends_with", "ends with"], ["in", "is one of (comma separated)"],
                                            ["is_empty", "is empty"], ["not_empty", "is not empty"]]},
                               {"name": "value", "type": "text", "label": "Value"},
                               {"name": "action", "type": "choice", "label": "Matching rows", "default": "keep",
                                "options": [["keep", "Keep them"], ["remove", "Remove them"]]}]},
    "sort_rows": {"label": "Sort rows", "category": "Rows", "fn": sort_rows,
                  "help": "Order the rows by a column.",
                  "fields": [{"name": "column", "type": "column", "label": "Column", "required": True},
                             {"name": "ascending", "type": "bool", "label": "Smallest first", "default": True}]},
    "fill_missing": {"label": "Fill missing values", "category": "Values", "fn": fill_missing,
                     "help": "Replace empty cells with a fixed value, the column's mean/median/most frequent "
                             "value, or the neighbouring row's value.",
                     "fields": [{"name": "columns", "type": "columns", "label": "Columns (all if empty)"},
                                {"name": "strategy", "type": "choice", "label": "Fill with", "default": "median",
                                 "options": [["value", "A fixed value"], ["mean", "Mean"], ["median", "Median"],
                                             ["most_frequent", "Most frequent value"],
                                             ["previous", "Value from the row above"],
                                             ["next", "Value from the row below"]]},
                                {"name": "value", "type": "text", "label": "Fixed value", "show_if": ["strategy", "value"]}]},
    "replace_values": {"label": "Replace values", "category": "Values", "fn": replace_values,
                       "help": "Change specific values, e.g. 'N/A' to empty or 'Y' to 'Yes'.",
                       "fields": [{"name": "column", "type": "column", "label": "Column", "required": True},
                                  {"name": "mapping", "type": "mapping", "label": "Value → replacement",
                                   "key": "text", "required": True}]},
    "clean_text": {"label": "Clean text", "category": "Values", "fn": clean_text,
                   "help": "Trim spaces and unify letter case so 'NY ' and 'ny' count as the same value.",
                   "fields": [{"name": "columns", "type": "columns", "label": "Columns (all text columns if empty)"},
                              {"name": "trim", "type": "bool", "label": "Trim spaces at both ends", "default": True},
                              {"name": "collapse_spaces", "type": "bool", "label": "Collapse repeated spaces",
                               "default": False},
                              {"name": "case", "type": "choice", "label": "Letter case", "default": "none",
                               "options": [["none", "Leave as is"], ["lower", "lower case"], ["upper", "UPPER CASE"],
                                           ["title", "Title Case"]]}]},
    "clip_outliers": {"label": "Limit extreme values", "category": "Values", "fn": clip_outliers,
                      "help": "Pull values far outside the usual range back to its edges (outliers).",
                      "fields": [{"name": "columns", "type": "columns", "label": "Columns (all number columns if empty)"},
                                 {"name": "method", "type": "choice", "label": "Range", "default": "iqr",
                                  "options": [["iqr", "Interquartile range (IQR)"], ["percentile", "Percentiles"]]},
                                 {"name": "factor", "type": "number", "label": "IQR factor", "default": 1.5,
                                  "show_if": ["method", "iqr"]},
                                 {"name": "lower", "type": "number", "label": "Lower percentile", "default": 1,
                                  "show_if": ["method", "percentile"]},
                                 {"name": "upper", "type": "number", "label": "Upper percentile", "default": 99,
                                  "show_if": ["method", "percentile"]}]},
    "round_numbers": {"label": "Round numbers", "category": "Values", "fn": round_numbers,
                      "help": "Round to a number of decimals.",
                      "fields": [{"name": "columns", "type": "columns", "label": "Columns (all number columns if empty)"},
                                 {"name": "decimals", "type": "number", "label": "Decimals", "default": 2}]},
    "date_parts": {"label": "Split a date", "category": "New columns", "fn": date_parts,
                   "help": "Add year, month, weekday... columns from a date column.",
                   "fields": [{"name": "column", "type": "column", "label": "Date column", "required": True},
                              {"name": "parts", "type": "parts", "label": "Parts", "default": ["year", "month", "day"],
                               "options": [[p, p] for p in DATE_PARTS]},
                              {"name": "drop_original", "type": "bool", "label": "Remove the original column",
                               "default": False}]},
    "bin_numbers": {"label": "Group numbers into ranges", "category": "New columns", "fn": bin_numbers,
                    "help": "Add a column that puts a number column into ranges (bins).",
                    "fields": [{"name": "column", "type": "column", "label": "Number column", "required": True},
                               {"name": "bins", "type": "number", "label": "Number of ranges", "default": 5},
                               {"name": "method", "type": "choice", "label": "Ranges", "default": "equal_width",
                                "options": [["equal_width", "Equal width"], ["quantile", "Equal number of rows"]]},
                               {"name": "new_column", "type": "text", "label": "New column name (optional)"}]},
    "combine_columns": {"label": "Combine two columns", "category": "New columns", "fn": combine_columns,
                        "help": "Add a column from two others: add, subtract, multiply, divide or join as text.",
                        "fields": [{"name": "left", "type": "column", "label": "First column", "required": True},
                                   {"name": "operation", "type": "choice", "label": "Operation", "default": "add",
                                    "options": [["add", "+ add"], ["subtract", "− subtract"], ["multiply", "× multiply"],
                                                ["divide", "÷ divide"], ["concat", "join as text"]]},
                                   {"name": "right", "type": "column", "label": "Second column", "required": True},
                                   {"name": "separator", "type": "text", "label": "Separator", "default": " ",
                                    "show_if": ["operation", "concat"]},
                                   {"name": "new_column", "type": "text", "label": "New column name (optional)"}]},
}
CATEGORIES = ("Columns", "Rows", "Values", "New columns")


def public_specs():
    """STEP_SPECS without the functions, for the studio's forms."""
    return [{"op": op, **{k: v for k, v in spec.items() if k != "fn"}} for op, spec in STEP_SPECS.items()]


def default_params(op):
    return {f["name"]: f.get("default", [] if f["type"] in ("columns", "parts") else ({} if f["type"] == "mapping"
                                                                                      else ""))
            for f in STEP_SPECS[op]["fields"]}


def validate_step(step):
    """Raise ValueError if ``step`` is not a well-formed recipe step; return a clean copy."""
    if not isinstance(step, dict) or step.get("op") not in STEP_SPECS:
        raise ValueError(f"Unknown step: {step.get('op') if isinstance(step, dict) else step!r}")
    spec = STEP_SPECS[step["op"]]
    params = dict(step.get("params") or {})
    known = {f["name"] for f in spec["fields"]}
    unknown = set(params) - known
    if unknown:
        raise ValueError(f"{spec['label']}: unknown setting(s) {', '.join(sorted(unknown))}")
    for field in spec["fields"]:
        value = params.get(field["name"], field.get("default"))
        if field.get("required") and (value is None or value == "" or value == [] or value == {}):
            raise ValueError(f"{spec['label']}: '{field['label']}' is required.")
        if field["type"] == "choice" and value not in (None, "") and value not in {o[0] for o in field["options"]}:
            raise ValueError(f"{spec['label']}: {value!r} is not a valid choice for '{field['label']}'.")
        if field["type"] in ("columns", "parts") and value is not None and not isinstance(value, list):
            raise ValueError(f"{spec['label']}: '{field['label']}' must be a list.")
        if field["type"] == "mapping" and value is not None and not isinstance(value, dict):
            raise ValueError(f"{spec['label']}: '{field['label']}' must be a mapping.")
    return {"op": step["op"], "params": params, "enabled": bool(step.get("enabled", True))}


def run_step(frame, step):
    spec = STEP_SPECS[step["op"]]
    params = {f["name"]: step["params"].get(f["name"], f.get("default")) for f in spec["fields"]}
    params = {k: v for k, v in params.items() if v is not None}
    return spec["fn"](frame, **params)
