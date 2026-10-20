import json
import os
import shutil
import traceback

import numpy as np
import pandas as pd

from easyml.config.constants import error_code_mappings
from easyml.utils.holdout import kept_rows
from easyml.utils.logger import write_log
from easyml.utils.utils import NumpyEncoder

def get_data_info(data, dataset_path=None):
    """
    Generates metadata and sample data from a given DataFrame.

    Args:
        data (pd.DataFrame): The dataset to analyze.
        dataset_path (str, optional): The path to the saved CSV file.

    Returns:
        dict: Contains sample data, shape, column names, and file size (in KB).
    """

    write_log("entered get_data_info", "debug", os.path.basename(__file__))
    try:
        rows, cols = data.shape

        # NaN becomes None for strict JSON (null). Only the rows shown are converted, not the whole dataset.
        def records(frame):
            return frame.replace({np.nan: None}).to_dict(orient="records")

        if rows < 5:
            data_sample = data_head = data_tail = records(data)
        else:
            data_sample = records(data.sample(5))
            data_head = records(data.head(5))
            data_tail = records(data.tail(5))
        
        columns = data.columns.tolist()

        # Get dataset file size in bytes
        dataset_size = 0
        if dataset_path and os.path.exists(dataset_path):
            dataset_size = os.path.getsize(dataset_path)

        dataset_size_kb = round(dataset_size / 1024, 2)  # Convert to KB

        data_info = {"data_sample": data_sample, "data_head": data_head,
                     "data_tail": data_tail, "rows": rows, "cols": cols, 
                     "column_names": columns,"dataset_size_kb": dataset_size_kb}

        write_log("successfully generated data info", "info", os.path.basename(__file__))
        return data_info
    except Exception as e:
        write_log(f"error generating data info: {str(e)}", "error", os.path.basename(__file__))
        raise e

def load_data_from_path(path):
    """
    Loads a dataset from a given file path (.csv or .xlsx supported).

    Args:
        path (str): File path to load.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        ValueError: If file extension is not supported.
    """

    write_log("entered to load_data_from_path", "debug", os.path.basename(__file__))

    lowered = path.lower()
    if lowered.endswith(".csv"):
        return pd.read_csv(path)
    elif lowered.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")

def move_to_runs(request_body):
    """
    Moves the saved dataset to the `artifacts/runs/` directory.

    Args:
        request_body (ML_RequestBody): The request context containing save paths.
    """

    # Copy the dataset into artifacts/runs/ instead of moving it
    # This ensures concurrent training tasks can use the same base uploaded data
    write_log("entered move_to_runs", "debug", os.path.basename(__file__))
    try:
        original_path = request_body.dataset_path
        if not original_path or not os.path.exists(original_path):
            raise FileNotFoundError(f"Dataset '{original_path}' was not found. Please upload it again.")

        run_specific_path = os.path.join(request_body.path_to_save, "data.csv")
        shutil.copyfile(original_path, run_specific_path)
        write_log(f"successfully copied dataset to {run_specific_path}", "info", os.path.basename(__file__))

        # Keep the user's own file name when the caller provided it (uploads are stored under internal names).
        request_body.original_dataset_name = request_body.original_dataset_name or os.path.basename(original_path)
        request_body.dataset_path = run_specific_path
    except Exception as e:
        write_log(f"error copying dataset: {str(e)}", "error", os.path.basename(__file__))
        raise e


def data_loader(request_body):
    """
    Loads data, prepares features and target, and encodes labels for classification tasks.

    Steps:
    - Moves data to runs folder
    - Loads it into a DataFrame
    - Handles ignored and target columns
    - Encodes target for classification

    Args:
        request_body (ML_RequestBody): Contains dataset path, task, and other configs.

    Raises:
        Exception: Logs and re-raises errors during data ingestion or preprocessing.
    """

    write_log("entered data_loader()", "debug", os.path.basename(__file__))

    move_to_runs(request_body)

    if len(getattr(request_body, "target_columns", None) or []) > 1:
        try:
            _load_several_targets(request_body)
        except Exception:
            if request_body.status_code == 200:
                request_body.error_message = "DataIngestionError"
                request_body.status_code = error_code_mappings[request_body.error_message]
                request_body.error_traceback = traceback.format_exc()
                write_log("error occurred- {} - {}".format(request_body.error_message, request_body.error_traceback),
                          "error", os.path.basename(__file__))
            raise
        return

    try:
        request_body.df = load_data_from_path(request_body.dataset_path)

        if request_body.target_column not in request_body.df.columns:
            raise ValueError(f"Target column '{request_body.target_column}' not found in dataset. "
                             f"Available columns: {request_body.df.columns.tolist()}")
        
        if request_body.ignore_columns is None:
            request_body.ignore_columns = []

        # Drop rows where y is null (skipped when there are none: it would only copy the data)
        if request_body.df[request_body.target_column].isna().any():
            request_body.df = request_body.df.dropna(subset=[request_body.target_column])
            request_body.source_rows = kept_rows(request_body.source_rows, request_body.df.index)
            request_body.df.reset_index(drop=True, inplace=True)
        
        # If user explicitly disabled imputation, drop all rows with any NaN
        if not getattr(request_body, 'imputation', True):
            write_log("dropping all rows with missing data (imputation=False)", "info", os.path.basename(__file__))
            used_columns = [c for c in request_body.df.columns if c not in request_body.ignore_columns]
            if request_body.df[used_columns].isna().any().any():
                request_body.df = request_body.df.dropna(subset=used_columns)
                request_body.source_rows = kept_rows(request_body.source_rows, request_body.df.index)
                request_body.df.reset_index(drop=True, inplace=True)
            
        drop_columns = [request_body.target_column] + request_body.ignore_columns
        request_body.X = request_body.df.drop(columns=drop_columns, errors='ignore')
        request_body.y = request_body.df[request_body.target_column]

        write_log("read dataframe in python", "info", os.path.basename(__file__))

        # Encode target if it's a classification task
        if "classification" in request_body.task.lower():
            request_body.target_dtype = str(request_body.y.dtype)
            request_body.target_encodings, encoded = encode_categorical_list(request_body.y.tolist())
            # An array (not a list) is shared with the grid-search workers instead of pickled for each fit.
            request_body.y = np.asarray(encoded)
            write_log(f"classification task - encoded target variable : {json.dumps(request_body.target_encodings, cls=NumpyEncoder)}", "info",
                      os.path.basename(__file__))

    except Exception as e:
        if request_body.status_code==200:
            request_body.error_message = "DataIngestionError"
            request_body.status_code = error_code_mappings[request_body.error_message]
            request_body.error_traceback = traceback.format_exc()

            write_log("error occurred- {} - {}".format(
                request_body.error_message, request_body.error_traceback),
                "error", os.path.basename(__file__))

        raise e



def _load_several_targets(request_body):
    """
    ``data_loader`` for a run with several target columns (0.13): ``y`` is one column per target, and each
    classification target is encoded on its own (``target_encodings_by_column``).
    """
    targets = list(request_body.target_columns)
    request_body.df = load_data_from_path(request_body.dataset_path)
    missing = [t for t in targets if t not in request_body.df.columns]
    if missing:
        raise ValueError(f"Column to predict '{missing[0]}' not found in dataset. "
                         f"Available columns: {request_body.df.columns.tolist()}")
    request_body.ignore_columns = request_body.ignore_columns or []

    if request_body.df[targets].isna().any().any():
        request_body.df = request_body.df.dropna(subset=targets)
        request_body.source_rows = kept_rows(request_body.source_rows, request_body.df.index)
        request_body.df.reset_index(drop=True, inplace=True)
    if not getattr(request_body, "imputation", True):
        used_columns = [c for c in request_body.df.columns if c not in request_body.ignore_columns]
        if request_body.df[used_columns].isna().any().any():
            request_body.df = request_body.df.dropna(subset=used_columns)
            request_body.source_rows = kept_rows(request_body.source_rows, request_body.df.index)
            request_body.df.reset_index(drop=True, inplace=True)
    if request_body.df.empty:
        raise ValueError("No rows have a value in every column to predict.")

    request_body.X = request_body.df.drop(columns=targets + request_body.ignore_columns, errors="ignore")
    if "classification" in request_body.task.lower():
        encodings, codes = {}, []
        for target in targets:
            values = request_body.df[target].tolist()
            if len(set(values)) < 2:
                raise ValueError(f"'{target}' has the same value in every row, so there is nothing to predict.")
            encodings[target], encoded = encode_categorical_list(values)
            codes.append(encoded)
        request_body.target_encodings_by_column = encodings
        request_body.y = np.column_stack(codes)
    else:
        for target in targets:
            if not pd.api.types.is_numeric_dtype(request_body.df[target]) or \
                    pd.api.types.is_bool_dtype(request_body.df[target]):
                raise ValueError(f"'{target}' does not hold numbers. Predict numbers and categories in separate "
                                 "runs (categories with a classification run).")
        request_body.y = request_body.df[targets].to_numpy(dtype=float)
    write_log(f"loaded {len(request_body.df)} rows with {len(targets)} columns to predict", "info",
              os.path.basename(__file__))


def encode_categorical_list(categorical_list):
    """
    Encodes a list of categorical values to numeric labels.

    Args:
        categorical_list (list): List of string/category labels.

    Returns:
        tuple:
            dict: Mapping of original labels to encoded values.
            list: List of encoded values.
    """

    unique_categories = sorted(set(categorical_list))
    encoding_dict = {category: index for index, category in enumerate(unique_categories)}
    encoded_list = [encoding_dict[category] for category in categorical_list]
    return encoding_dict, encoded_list
