from flask.json.provider import DefaultJSONProvider
import numpy as np
import json

def convert_numpy(obj):
    """
    Converts basic NumPy data types to native Python types.

    Args:
        obj (any): Object to convert.

    Returns:
        any: Converted object (int, float, or original).
    """

    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    # elif isinstance(obj, np.ndarray):
    #     return obj.tolist()
    return obj

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to serialize NumPy arrays and integer types.
    """

    def default(self, obj):
        """
        Overrides default serialization for NumPy types.

        Args:
            obj (any): Object to serialize.

        Returns:
            any: Serialized object.
        """

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

class NumpyJSONProvider(DefaultJSONProvider):
    """
    Custom Flask JSON provider for handling NumPy types in Flask responses.
    """

    def __init__(self, app):
        """
        Initializes the custom JSON provider.

        Args:
            app (Flask): Flask app instance.
        """

        super().__init__(app)

    def default(self, obj):
        """
        Overrides default JSON serialization for NumPy types.

        Args:
            obj (any): Object to serialize.

        Returns:
            any: Serialized object.
        """

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)
