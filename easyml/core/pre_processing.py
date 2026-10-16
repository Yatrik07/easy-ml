from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from easyml.utils.logger import write_log
import os
import traceback
import joblib
import pandas as pd
from easyml.config.GLOBAL_VARS import ML_RequestBody, error_code_mappings

# Nominal columns with more distinct values than this keep their most frequent categories and
# share one "infrequent" column for the rest, so an ID-like text column cannot explode into
# thousands of dense one-hot columns.
MAX_ONEHOT_CATEGORIES = 30


def one_hot_encoder():
    """One-hot encoder with a cardinality cap; unseen or rare values go to the infrequent column."""
    return OneHotEncoder(handle_unknown='infrequent_if_exist', max_categories=MAX_ONEHOT_CATEGORIES,
                         sparse_output=False)


def reduction_steps(request_body):
    """
    PCA steps for optional feature reduction.

    PCA picks the directions of largest variance, so unscaled features (e.g. income in the tens of
    thousands next to 0/1 one-hot columns) would dominate it: features are standardised first
    whenever the column pipelines did not already scale them.
    """
    if not request_body.reduce_features:
        return []
    steps = [] if request_body.scaling else [('pca_scaler', StandardScaler())]
    return steps + [('pca', PCA(n_components=getattr(request_body, 'variance_retained', 0.95)))]

def preprocessing_controller(request_body: ML_RequestBody):
    """
    Main function to preprocess the data:
    - Determines column types
    - Encodes categorical variables
    - Splits data into train/test
    - Processes numerical variables
    - Builds and applies a transformation pipeline (with optional PCA)

    Args:
        request_body (ML_RequestBody): User request data structure containing dataset and config.

    Raises:
        Exception: Logs and re-raises any exception that occurs during preprocessing.
    """

    from easyml.utils.socket_emitter import emit_progress
    try:
        write_log("entered preprocessing controller", "debug", os.path.basename(__file__))
        
        emit_progress("🔍 Determining feature types...", pct=18)
        determine_column_types(request_body)

        emit_progress("🔤 Setting up categorical encodings...", pct=22)
        # Process categorical variables (ordinal and nominal) before split
        encode_categorical(request_body)

        # Split data — use stratify for classification but guard against rare classes
        write_log("splitting the data into train-test-split", "info", os.path.basename(__file__))
        is_classification = "classification" in request_body.task.lower()
        stratify = None
        if is_classification:
            counts = pd.Series(request_body.y).value_counts()
            stratify = request_body.y if counts.min() >= 2 else None
        request_body.X_train, request_body.X_test, request_body.y_train, request_body.y_test = \
            train_test_split(request_body.X, request_body.y, test_size=0.2, random_state=request_body.random_seed,
                             stratify=stratify)

        emit_progress("🔢 Setting up numerical transformations...", pct=26)
        # Process numerical variables after split
        process_numerical(request_body)

        write_log("appending all the transformers to request_body.column_transformer.transformers ", "info", os.path.basename(__file__))
        request_body.column_transformer.transformers = request_body.transformers

        # Create final pipeline with ColumnTransformer and PCA
        final_pipeline = Pipeline(steps=[('column_transformer', request_body.column_transformer)] + reduction_steps(request_body))

        emit_progress("🏗️ Building and transforming data pipeline...", pct=30)
        write_log("built final pipeline with column transformer and pca", "info", os.path.basename(__file__))

        # Transform train and test data using the final pipeline
        write_log("transforming the data using the final pipeline", "info", os.path.basename(__file__))
        request_body.X_test_raw = request_body.X_test  # for permutation importance (transform never modifies it)
        request_body.X_train_raw = request_body.X_train  # for cross-validation, which refits preprocessing per fold
        request_body.X_train = final_pipeline.fit_transform(request_body.X_train)
        request_body.X_test = final_pipeline.transform(request_body.X_test)
        request_body.pipeline = final_pipeline
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
    """
    Detects and assigns nominal, ordinal, categorical, and numeric columns
    in the dataset using heuristics and user input.

    Args:
        request_body (ML_RequestBody): The request body with the raw dataset.

    Raises:
        Exception: Logs and re-raises if dataset is missing or invalid.
    """

    try:
        write_log("entered - determine_column_types", "debug", os.path.basename(__file__))

        if request_body.X is None:
            raise ValueError("X is None — Make sure dataset is loaded before determining column types.")

        # Ensure ordinal_columns and nominal_columns are lists
        request_body.ordinal_columns = request_body.ordinal_columns or []
        request_body.nominal_columns = request_body.nominal_columns or []

        X = request_body.X
        # Text / category / bool columns can only ever be categorical.
        always_categorical = set(X.select_dtypes(include=["object", "category", "bool", "string"]).columns)

        user_selected = [col for col in (request_body.categorical_columns or []) if col in X.columns]
        if request_body.nominal_columns:
            nominal_columns = set(request_body.nominal_columns)
        elif user_selected:
            # Respect the user's choice, plus any column that cannot be treated as numeric.
            nominal_columns = set(user_selected) | always_categorical
        else:
            # Auto-detect: non-numeric columns plus low-cardinality columns.
            nominal_columns = always_categorical | {col for col in X.columns if X[col].nunique() < 10}

        request_body.nominal_columns = [
            col for col in X.columns if col in nominal_columns and col not in request_body.ordinal_columns
        ]

        # Combine ordinal and nominal columns into categorical columns
        categorical = set(request_body.ordinal_columns) | set(request_body.nominal_columns)
        request_body.categorical_columns = [col for col in X.columns if col in categorical]

        # Numeric columns are those not in categorical columns
        request_body.numeric_columns = [
            col for col in request_body.X.columns
            if col not in request_body.categorical_columns
        ]

        write_log(f"determined column types - \n\tnumerical columns - {request_body.numeric_columns} \n\tcategorical columns - "
                  f"{request_body.categorical_columns}", "info", os.path.basename(__file__))

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
    """
    Encodes categorical features using appropriate pipelines for:
    - Ordinal columns: Imputation + Ordinal Encoding (+ optional Scaling)
    - Nominal columns: Imputation + OneHotEncoding (+ optional Scaling)

    Args:
        request_body (ML_RequestBody): The request body with encoded transformers.

    Raises:
        Exception: Logs and re-raises if encoding fails.
    """

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
                ('encoder', one_hot_encoder()),
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
    """
    Processes numerical columns using a pipeline of:
    - Median imputation
    - Optional scaling

    Args:
        request_body (ML_RequestBody): The request body with dataset and config.

    Raises:
        Exception: Logs and re-raises if transformation fails.
    """

    try:
        write_log("entered process_numerical()", "debug", os.path.basename(__file__))

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
