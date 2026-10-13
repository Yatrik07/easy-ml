from flask.json.provider import DefaultJSONProvider
import numpy as np
import json
import math


def convert_numpy(obj):
    """
    Converts basic NumPy data types to native Python types.

    Single-element arrays (e.g. CatBoost's (1,) / (1, 1) predictions) are unwrapped to a scalar.

    Args:
        obj (any): Object to convert.

    Returns:
        any: Converted object (int, float, bool, list, or original).
    """

    if isinstance(obj, np.ndarray):
        return convert_numpy(obj.item()) if obj.size == 1 else obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return None if math.isnan(value) else value
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _default(obj):
    if isinstance(obj, (np.ndarray, np.integer, np.floating, np.bool_)):
        return convert_numpy(obj)
    raise TypeError


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to serialize NumPy arrays and scalar types.
    """

    def default(self, obj):
        try:
            return _default(obj)
        except TypeError:
            return super().default(obj)


class NumpyJSONProvider(DefaultJSONProvider):
    """
    Custom Flask JSON provider for handling NumPy types in Flask responses.
    Falls back to Flask's default handling (dates, UUIDs, dataclasses, ...).
    """

    @staticmethod
    def default(obj):
        try:
            return _default(obj)
        except TypeError:
            return DefaultJSONProvider.default(obj)
