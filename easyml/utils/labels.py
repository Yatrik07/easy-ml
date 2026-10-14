"""
Class labels of a classification run, in their original type.

Training encodes the target as 0..K-1 (``target_encodings``: label -> code). JSON object keys are
always strings, so a 0/1 target used to come back from inference as ``"0"``/``"1"``. New runs also
save ``target_labels`` (a JSON list in code order, which keeps ints, floats and bools); older runs
fall back to guessing the type from the string keys, but only when that guess is exact.
"""

import math


def labels_in_code_order(target_encodings):
    """``{label: code}`` -> ``[label for code 0, label for code 1, ...]`` with plain Python types."""
    if not target_encodings:
        return None
    ordered = sorted(target_encodings.items(), key=lambda item: int(item[1]))
    return [_plain(label) for label, _ in ordered]


def decoder_from_config(config):
    """Map encoded class index -> original label for a run's ``config_json.json`` contents."""
    labels = config.get("target_labels")
    if isinstance(labels, list) and labels:
        return dict(enumerate(labels))
    encodings = config.get("target_encodings") or {}
    if not encodings:
        return {}
    kind = _infer_legacy_kind(list(encodings))
    return {int(code): _cast(label, kind) for label, code in encodings.items()}


def _plain(value):
    # numpy scalars -> Python scalars so json keeps the type
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (AttributeError, ValueError):
            return value
    return value


def _infer_legacy_kind(keys):
    if all(k in ("true", "false", "True", "False") for k in keys):
        return "bool"
    if all(_round_trips(k, int) for k in keys):
        return "int"
    if all(_round_trips(k, float) for k in keys):
        return "float"
    return "str"


def _round_trips(text, kind):
    try:
        value = kind(text)
    except (TypeError, ValueError):
        return False
    if kind is float:
        return math.isfinite(value) and repr(value) == text
    return str(value) == text


def _cast(text, kind):
    if kind == "bool":
        return text in ("true", "True")
    if kind == "int":
        return int(text)
    if kind == "float":
        return float(text)
    return text
