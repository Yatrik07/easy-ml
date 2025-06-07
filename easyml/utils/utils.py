from flask.json.provider import DefaultJSONProvider
import numpy as np
import json

def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    # elif isinstance(obj, np.ndarray):
    #     return obj.tolist()
    return obj
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

class NumpyJSONProvider(DefaultJSONProvider):
    def __init__(self, app):
        super().__init__(app)

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

#
# class PreprocessingError(EasyAIException):
#     def __init__(self, message="Preprocessing failed", details=None):
#         super().__init__(message, error_code="500", details=details)
#
#
# class ModelTrainingError(EasyAIException):
#     def __init__(self, message="Model training failed", details=None):
#         super().__init__(message, error_code="520", details=details)