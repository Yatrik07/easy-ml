import pandas as pd
import numpy as np
import os


def data_loader(request_body):
    print("request_body task: ", request_body.task)
    def load_data_from_path(path):
        """Load dataset from the given path based on the file type."""
        if path.endswith(".csv"):
            return pd.read_csv(path)
        elif path.endswith(".xlsx"):
            return pd.read_excel(path)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")

    if request_body.dataset_path:
        try:
            request_body.df = load_data_from_path(request_body.dataset_path)
            if request_body.target_column not in request_body.df.columns:
                raise ValueError(f"Target column '{request_body.target_column}' not found in the dataset.")
            request_body.X = request_body.df.drop(columns=[request_body.target_column])
            request_body.y = request_body.df[request_body.target_column]
        except Exception as e:
            raise ValueError(f"Error loading dataset: {e}")

    elif request_body.actual_dataset is not None:
        request_body.df = request_body.actual_dataset
        if request_body.target_column not in request_body.df.columns:
            raise ValueError(f"Target column '{request_body.target_column}' not found in the dataset.")
        request_body.X = request_body.df.drop(columns=[request_body.target_column])
        request_body.y = request_body.df[request_body.target_column]
        print("request_body.df: ", request_body.df.head())
        print("request_body.X: ", request_body.X.head())
    else:
        raise ValueError("No dataset provided. Please provide a dataset path or an actual dataset.")

    # Encode target if it's a classification task
    if "classification" in request_body.task.lower():
        request_body.target_encodings, request_body.y = encode_categorical_list(request_body.y.tolist())
        print("Target Encodings: ", request_body.target_encodings)
        print("Encoded Target Values: ", request_body.y)


def encode_categorical_list(categorical_list):
    unique_categories = sorted(set(categorical_list))  # or use list(set(...)) to avoid sorting
    encoding_dict = {category: index + 1 for index, category in enumerate(unique_categories)}
    encoded_list = [encoding_dict[category] for category in categorical_list]
    return encoding_dict, encoded_list



