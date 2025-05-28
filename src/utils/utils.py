import numpy as np

def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    # elif isinstance(obj, np.ndarray):
    #     return obj.tolist()
    return obj
#
# class PreprocessingError(EasyAIException):
#     def __init__(self, message="Preprocessing failed", details=None):
#         super().__init__(message, error_code="500", details=details)
#
#
# class ModelTrainingError(EasyAIException):
#     def __init__(self, message="Model training failed", details=None):
#         super().__init__(message, error_code="520", details=details)