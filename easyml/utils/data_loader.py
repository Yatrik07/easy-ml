import pandas as pd
import os, json
from easyml.utils.logger import write_log
import traceback
from easyml.config.GLOBAL_VARS import error_code_mappings
from utils.utils import *
from config.GLOBAL_VARS import *


def get_data_info(data):
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
    """Load dataset from the given path based on the file type."""
    write_log("entered to load_data_from_path", "info", os.path.basename(__file__))

    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".xlsx"):
        return pd.read_excel(path)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")


def data_loader(request_body):

    write_log("entered data_loader()", "info", os.path.basename(__file__))

    try:
        request_body.df = load_data_from_path(request_body.dataset_path)
        
        if request_body.target_column not in request_body.df.columns:
            raise ValueError(f"Target column '{request_body.target_column}' not found in the dataset.")
        
        if request_body.ignore_columns is None:
            request_body.ignore_columns = []
            
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
    unique_categories = sorted(set(categorical_list))
    encoding_dict = {category: index for index, category in enumerate(unique_categories)}
    encoded_list = [encoding_dict[category] for category in categorical_list]
    return encoding_dict, encoded_list
