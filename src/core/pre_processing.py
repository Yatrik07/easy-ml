from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils.logger import write_log
import os
import traceback
import joblib
from src.config.GLOBAL_VARS import ML_RequestBody, error_code_mappings

def preprocessing_controller(request_body: ML_RequestBody):
    try:
        write_log("entered preprocessing controller", "info", os.path.basename(__file__))
        determine_column_types(request_body)

        # Process categorical variables (ordinal and nominal) before split
        encode_categorical(request_body)

        # Split data into train/test
        write_log("splitting the data into train-test-split", "info", os.path.basename(__file__))
        request_body.X_train, request_body.X_test, request_body.y_train, request_body.y_test = \
            train_test_split(request_body.X, request_body.y, test_size=0.2, random_state=42,
                             stratify=request_body.y if "classification" in request_body.task.lower() else None)

        # Process numerical variables after split
        process_numerical(request_body)

        write_log("appending all the transformers to request_body.column_transformer.transformers ", "info", os.path.basename(__file__))
        request_body.column_transformer.transformers = request_body.transformers

        # Create final pipeline with ColumnTransformer and PCA
        final_pipeline = Pipeline(steps=[
            ('column_transformer', request_body.column_transformer),
        ] + ([('pca', PCA(n_components=getattr(request_body, 'variance_retained', 0.95)))] if request_body.reduce_features else []))

        write_log("built final pipeline with column transformer and pca", "info", os.path.basename(__file__))

        print("printing final pipeline: ")
        print_pipeline(final_pipeline)

        # Transform train and test data using the final pipeline
        write_log("transforming the data using the final pipeline", "info", os.path.basename(__file__))
        request_body.X_train = final_pipeline.fit_transform(request_body.X_train)
        request_body.X_test = final_pipeline.transform(request_body.X_test)
        write_log("transformed the data using the final pipeline", "info", os.path.basename(__file__))

        joblib.dump(final_pipeline, os.path.join(request_body.path_to_save, "preprocessing_pipeline.pkl"), protocol=5)

        write_log(f"X_train transformed: shape->{request_body.X_train.shape} type->{type(request_body.X_train)}", "info", os.path.basename(
            __file__))
        write_log(f"X_test transformed: shape->{request_body.X_test.shape} type->{type(request_body.X_test)}", "info", os.path.basename(
            __file__))

    except Exception as e:
        if request_body.status_code == 200:
           request_body.error_message = "PreProcessingError"
           request_body.status_code = error_code_mappings[request_body.error_message]
           request_body.error_traceback = traceback.format_exc()

           write_log("error occurred- {} - {}".format(
               request_body.error_message, request_body.error_traceback),
               "error", os.path.basename(__file__))

        raise e

def determine_column_types(request_body: ML_RequestBody):

    try:
        write_log("entered - determine_column_types", "info", os.path.basename(__file__))

        if request_body.X is None:
            raise ValueError("X is None — Make sure dataset is loaded before determining column types.")

        # Ensure ordinal_columns and nominal_columns are lists
        request_body.ordinal_columns = request_body.ordinal_columns or []
        request_body.nominal_columns = request_body.nominal_columns or []

        # If nominal_columns are not provided, detect categorical columns
        if not request_body.nominal_columns:
            # Start with object and category columns, excluding ordinal columns
            nominal_columns = request_body.X.select_dtypes(
                include=["object", "category"]).columns.difference(request_body.ordinal_columns).tolist()

            # Extend with columns having fewer than 10 unique values
            nominal_columns.extend([col for col in request_body.X.columns if request_body.X[col].nunique() < 10])

            # Include boolean columns
            bool_columns = [col for col in request_body.X.columns if request_body.X[col].dtype == bool]
            nominal_columns = list(set(nominal_columns).union(bool_columns))

            # Update nominal_columns
            request_body.nominal_columns = nominal_columns

        # Combine ordinal and nominal columns into categorical columns
        request_body.categorical_columns = list(set(request_body.ordinal_columns).union(request_body.nominal_columns))

        # Numeric columns are those not in categorical columns
        request_body.numeric_columns = [
            col for col in request_body.X.columns
            if col not in request_body.categorical_columns
        ]

        write_log(f"determined column types - \n\tnumerical columns - {request_body.numeric_columns} \n\tcategorical columns - "
                  f"{request_body.categorical_columns}", "info", os.path.basename(__file__))

        print("Numeric columns:", request_body.numeric_columns)
        print("Categorical columns:", request_body.categorical_columns)

    except Exception as e:
        if request_body.status_code == 200:
           request_body.error_message = "PreProcessingError"
           request_body.status_code = error_code_mappings[request_body.error_message]
           request_body.error_traceback = traceback.format_exc()

           write_log("error occurred- {} - {}".format(
               request_body.error_message, request_body.error_traceback),
               "error", os.path.basename(__file__))

        raise e


def encode_categorical(request_body : ML_RequestBody):

    try:
        write_log("encoding categorical variables", "info", os.path.basename(__file__))

        if request_body.ordinal_columns:
            ordinal_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder()),
            ] + ([('scaler', StandardScaler())] if request_body.scaling else []))
            request_body.transformers.append(('ordinal_pipeline', ordinal_pipeline, request_body.ordinal_columns))

            write_log("appending pipeline to thr request_body.transformers for ordinal columns", "info", os.path.basename(
                __file__))

        # Nominal pipeline: Imputation -> OneHot Encoding -> Scaling
        if request_body.nominal_columns:
            nominal_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ] + ([('scaler', StandardScaler())] if request_body.scaling else []))
            request_body.transformers.append(('nominal_pipeline', nominal_pipeline, request_body.nominal_columns))

            write_log("appending pipeline to thr request_body.transformers for nominal columns", "info", os.path.basename(
                __file__))

    except Exception as e:
        if request_body.status_code == 200:
            request_body.error_message = "PreProcessingError"
            request_body.status_code = error_code_mappings[request_body.error_message]
            request_body.error_traceback = traceback.format_exc()

            write_log("error occurred- {} - {}".format(
                request_body.error_message, request_body.error_traceback),
                "error", os.path.basename(__file__))

        raise e

def process_numerical(request_body : ML_RequestBody):

    try:
        write_log("entered process_numerical()", "info", os.path.basename(__file__))

        if request_body.numeric_columns:
            numerical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
            ] + ([('scaler', StandardScaler())] if request_body.scaling else []))
            request_body.transformers.append(('numerical_pipeline', numerical_pipeline, request_body.numeric_columns))

            write_log("appending pipeline to the request_body.transformers for numeric columns", "info", os.path.basename(
                __file__))

    except Exception as e:
        if request_body.status_code == 200:
            request_body.error_message = "PreProcessingError"
            request_body.status_code = error_code_mappings[request_body.error_message]
            request_body.error_traceback = traceback.format_exc()

            write_log("error occurred- {} - {}".format(
                request_body.error_message, request_body.error_traceback),
                "error", os.path.basename(__file__))

        raise e


def print_pipeline(pipeline, indent=0):
    prefix = "    " * indent
    if isinstance(pipeline, Pipeline):
        print(f"{prefix}Pipeline(steps=[")
        for name, step in pipeline.steps:
            print(f"{prefix}    ('{name}',")
            print_pipeline(step, indent + 2)
            print(f"{prefix}    ),")
        print(f"{prefix}])")
    elif isinstance(pipeline, ColumnTransformer):
        print(f"{prefix}ColumnTransformer(transformers=[")
        for name, transformer, columns in pipeline.transformers:
            print(f"{prefix}    ('{name}',")
            print_pipeline(transformer, indent + 2)
            print(f"{prefix}    , columns={columns}),")
        print(f"{prefix}])")
    else:
        print(f"{prefix}{pipeline}")
