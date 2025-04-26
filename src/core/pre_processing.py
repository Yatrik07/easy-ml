import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import List, Tuple, Union


def preprocessing_controller(request_body):
    # Initialize transformers list
    # request_body.transformers = []

    # Determine column types first
    determine_column_types(request_body)

    # Process categorical variables (ordinal and nominal) before split
    encode_categorical(request_body)

    # Split data into train/test
    request_body.X_train, request_body.X_test, request_body.y_train, request_body.y_test = \
        train_test_split(request_body.X, request_body.y, test_size=0.2, random_state=42,
                         stratify=request_body.y if "classification" in request_body.task.lower() else None)

    # Process numerical variables after split
    process_numerical(request_body)

    # Create ColumnTransformer with all transformers
    # column_transformer = ColumnTransformer(transformers=request_body.transformers)
    request_body.column_transformer.transformers = request_body.transformers

    # Create final pipeline with ColumnTransformer and PCA
    final_pipeline = Pipeline([
        ('column_transformer', request_body.column_transformer),
        ('pca', PCA(n_components=getattr(request_body, 'variance_retained', 0.95)))
    ])

    # Debug: Print transformers
    # for idx, transformer in enumerate(request_body.transformers):
    #     print(f"Transformer {idx}: {transformer[0]}, Columns: {transformer[2]}")

    print_pipeline(final_pipeline)

    # Transform train and test data using the final pipeline
    request_body.X_train = final_pipeline.fit_transform(request_body.X_train)
    request_body.X_test = final_pipeline.transform(request_body.X_test)

    print("X_train transformed:", request_body.X_train.shape, type(request_body.X_train))
    print("X_test transformed:", request_body.X_test.shape, type(request_body.X_test))


def get_column_transformer(operation: str, columns: Union[str, List[str]], **kwargs) -> Tuple[str, object, List[str]]:
    if operation == "scale":
        scaler = StandardScaler()
        return ("standard_scaler", scaler, columns)

    elif operation == "impute":
        strategy = kwargs.get("strategy", "most_frequent")
        imputer = SimpleImputer(strategy=strategy)
        return (f"{strategy}_imputer", imputer, columns)

    elif operation == "encode":
        encoding_type = kwargs.get("encoding_type", "onehot")
        if encoding_type == "onehot":
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        elif encoding_type == "ordinal":
            encoder = OrdinalEncoder()
        else:
            raise ValueError(f"Unsupported encoding type: {encoding_type}")
        return (f"{encoding_type}_encoder", encoder, columns)

    else:
        raise ValueError(f"Unsupported operation: {operation}")

def determine_column_types(request_body):
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

    print("Numeric columns:", request_body.numeric_columns)
    print("Categorical columns:", request_body.categorical_columns)


def encode_categorical(request_body):
    # Ordinal pipeline: Imputation -> Ordinal Encoding -> Scaling
    if request_body.ordinal_columns:
        ordinal_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder()),
            ('scaler', StandardScaler())
        ])
        request_body.transformers.append(('ordinal_pipeline', ordinal_pipeline, request_body.ordinal_columns))

    # Nominal pipeline: Imputation -> OneHot Encoding -> Scaling
    if request_body.nominal_columns:
        nominal_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ('scaler', StandardScaler())
        ])
        request_body.transformers.append(('nominal_pipeline', nominal_pipeline, request_body.nominal_columns))


def process_numerical(request_body):
    # Numerical pipeline: Imputation -> Scaling
    if request_body.numeric_columns:
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('scaler', StandardScaler())
        ])
        request_body.transformers.append(('numerical_pipeline', numerical_pipeline, request_body.numeric_columns))


def print_pipeline(pipeline, indent=0):
    print("print_pipeline")
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