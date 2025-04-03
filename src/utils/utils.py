from typing import Optional, List, Tuple, Any
import pandas as pd
import numpy as np


def custom_encoder(obj: Any):
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")  # Convert DataFrame to a list of dictionaries
    elif isinstance(obj, pd.Series):
        return obj.to_dict()  # Convert Series to a dictionary
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert ndarray to a list
    elif isinstance(obj, (np.int32, np.int64)):  # Handle NumPy integers
        return int(obj)  # Convert to standard Python int
    elif isinstance(obj, (np.float32, np.float64)):  # Handle NumPy floats
        return float(obj)  # Convert to standard Python float
    else:
        return "A non-serializable object"
    raise TypeError(f"Type {type(obj)} not serializable")
