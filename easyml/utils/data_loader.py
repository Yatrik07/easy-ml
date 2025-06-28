import pandas as pd
import os, json
from easyml.utils.logger import write_log
import traceback
from easyml.config.GLOBAL_VARS import error_code_mappings
from easyml.utils.utils import *
from easyml.config.GLOBAL_VARS import *
import shutil

def get_data_info(data):
    """
    Generates metadata and sample data from a given DataFrame.

    Args:
        data (pd.DataFrame): The dataset to analyze.

    Returns:
        dict: Contains sample data, shape, column names, and file size (in KB).
    """

    rows, cols = data.shape

    if rows < 5:
        data_sample = data.to_dict(orient="records")   # Convert DataFrame to a dictionary
        data_head = data.to_dict(orient="records")
        data_tail = data.to_dict(orient="records") 
    else:
        data_sample = data.sample(5).to_dict(orient="records") 
        data_head = data.head(5).to_dict(orient="records") 
        data_tail = data.tail(5).to_dict(orient="records")
    
    columns = data.columns.tolist()

    # Get dataset file size in bytes
    dataset_size = os.path.getsize(save_data_path)  # Size in bytes
    dataset_size_kb = round(dataset_size / 1024, 2)  # Convert to KB

    data_info = {"data_sample": data_sample, "data_head": data_head,
                 "data_tail": data_tail, "rows": rows, "cols": cols, 
                 "column_names": columns,"dataset_size_kb": dataset_size_kb}

    return data_info

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

    write_log("entered to load_data_from_path", "info", os.path.basename(__file__))

    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".xlsx"):
        return pd.read_excel(path)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")

def move_to_runs(request_body):
    """
    Moves the saved dataset to the `artifacts/runs/` directory.

    Args:
        request_body (ML_RequestBody): The request context containing save paths.
    """

    # moving dataset into artifacts/runs/
    request_body.dataset_path = os.path.join(request_body.path_to_save, "data.csv")
    shutil.move(save_data_path, request_body.dataset_path)


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

    write_log("entered data_loader()", "info", os.path.basename(__file__))

    move_to_runs(request_body)

    try:
        request_body.df = load_data_from_path(request_body.dataset_path)
        
        if request_body.target_column not in request_body.df.columns:
            raise
        
        if request_body.ignore_columns is None:
            request_body.ignore_columns = []

        # Drop rows where y is null
        request_body.df = request_body.df.dropna(subset=[request_body.target_column])
        request_body.df.reset_index(drop=True, inplace=True)
            
        drop_columns = [request_body.target_column] + request_body.ignore_columns
        request_body.X = request_body.df.drop(columns=drop_columns)
        request_body.y = request_body.df[request_body.target_column]

        write_log("read dataframe in python", "info", os.path.basename(__file__))

        # Encode target if it's a classification task
        if "classification" in request_body.task.lower():
            request_body.target_encodings, request_body.y = encode_categorical_list(request_body.y.tolist())
            write_log(f"classification task - encoded target variable : {json.dumps(request_body.target_encodings)}", "info",
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
