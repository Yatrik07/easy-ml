import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from typing import List, Tuple, Union
from sklearn.compose import ColumnTransformer

def preprocessing_controller(request_body):

    encode_categorical(request_body)

    determine_column_types(request_body)

    if request_body.task.startswith("ML"):

        request_body.X_train, request_body.X_test, request_body.y_train, request_body.y_test = \
            train_test_split(request_body.X, request_body.y, test_size=0.2)

        if request_body.imputation:
            impute_missing_values(request_body)

        if request_body.scaling:
            scale_features(request_body)

        if request_body.reduce_features:
            apply_pca(request_body)

        request_body.column_transformer.transformers = request_body.transformers

        for idx, i in enumerate(request_body.transformers):
            print(f"{idx}: " , i)

        request_body.X_train = request_body.column_transformer.fit_transform(request_body.X_train)
        request_body.X_test = request_body.column_transformer.transform(request_body.X_test)

        print("X_transformed: ", request_body.X_train.shape, type(request_body.X_train) )
        print("request_body.X_test: ",request_body.X_test.shape, type(request_body.X_test) )


def get_column_transformer(operation: str, columns: Union[str,List[str]], **kwargs) -> Tuple[str, object, List[str]]:
    if operation == "scale":
        scaler_type = kwargs.get("scaling_type", "standard")
        scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        return (f"{scaler_type}_scaler", scaler, columns)

    elif operation == "impute":
        strategy = kwargs.get("strategy", "median")
        imputer = SimpleImputer(strategy=strategy)
        return (f"{strategy}_imputer", imputer, columns)

    elif operation == "pca":
        pca = PCA(n_components=kwargs.get("variance_retained", 0.95))
        columns = columns if type(columns) == list else None
        return ("pca", pca, columns)

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

# TODO : This function can be applied if we want to determine column types before encoding the categorical columns
# def determine_column_types(request_body):
#     if not request_body.nominal_columns:
#         request_body.nominal_columns = request_body.X.select_dtypes(include=["object", "category"]).columns.difference(
#             request_body.ordinal_columns).tolist()
#
#     request_body.categorical_columns = list(set(request_body.ordinal_columns).union(request_body.nominal_columns))
#     request_body.numeric_columns = [col for col in request_body.X.columns if col not in request_body.categorical_columns]
# def determine_column_types(request_body):
#     if request_body.X is None:
#         raise ValueError("X is None — Make sure dataset is loaded before determining column types.")
#
#     # Ensure ordinal_columns is at least an empty list
#     request_body.ordinal_columns = request_body.ordinal_columns or []
#
#     # If nominal columns not provided, detect automatically
#     if not request_body.nominal_columns:
#         request_body.nominal_columns = request_body.X.select_dtypes(include=["object", "category"])\
#                                           .columns.difference(request_body.ordinal_columns).tolist()
#
#     request_body.categorical_columns = list(set(request_body.ordinal_columns).union(request_body.nominal_columns))
#     request_body.numeric_columns = [col for col in request_body.X.columns if col not in request_body.categorical_columns]


def determine_column_types(request_body):
    if request_body.X is None:
        raise ValueError("X is None — Make sure dataset is loaded before determining column types.")

    # Ensure ordinal_columns is at least an empty list
    # request_body.ordinal_columns = request_body.ordinal_columns or []

    # If nominal columns are not provided, detect them automatically by checking for categorical-like values
    # if not request_body.nominal_columns:
    #     # Nominal columns are typically those that are not ordinal and have unique, low cardinality
    #     request_body.nominal_columns = [
    #         col for col in request_body.X.columns
    #         if request_body.X[col].dtype in ['int', 'float'] and request_body.X[col].nunique() < 10
    #     ]

    # Combine the provided ordinal columns and detected nominal columns to get all categorical columns
    request_body.categorical_columns = list(set(request_body.ordinal_columns).union(request_body.nominal_columns))

    # All remaining columns will be considered as numeric
    request_body.numeric_columns = [
        col for col in request_body.X.columns
        if col not in request_body.categorical_columns
    ]



# def scale_features(request_body, scaling_type="standard"):
#     # Select the appropriate scaler based on the scaling_type argument
#     scaler = StandardScaler() if scaling_type == "standard" else MinMaxScaler()
#
#     # Fit the scaler on X_train and transform both X_train and X_test
#     request_body.X_train = pd.DataFrame(scaler.fit_transform(request_body.X_train), columns=request_body.X_train.columns)
#     request_body.X_test = pd.DataFrame(scaler.transform(request_body.X_test), columns=request_body.X_test.columns)


def scale_features(request_body, scaling_type="standard"):
    if request_body.numeric_columns:
        name, transformer, columns = get_column_transformer("scale", request_body.numeric_columns, scaling_type=scaling_type)
        request_body.transformers.append((name, transformer, columns))



# def impute_missing_values(request_body):
#     # Impute categorical columns
#     if request_body.categorical_columns:
#         cat_imputer = SimpleImputer(strategy="most_frequent")
#         # Fit on training data
#         request_body.X_train[request_body.categorical_columns] = cat_imputer.fit_transform(request_body.X_train[request_body.categorical_columns])
#         # Transform test data
#         request_body.X_test[request_body.categorical_columns] = cat_imputer.transform(request_body.X_test[request_body.categorical_columns])
#
#     # Impute numeric columns
#     if request_body.numeric_columns:
#         num_imputer = SimpleImputer(strategy="median")
#         # Fit on training data
#         request_body.X_train[request_body.numeric_columns] = num_imputer.fit_transform(request_body.X_train[request_body.numeric_columns])
#         # Transform test data
#         request_body.X_test[request_body.numeric_columns] = num_imputer.transform(request_body.X_test[request_body.numeric_columns])


def impute_missing_values(request_body):
    if request_body.categorical_columns:
        name, transformer, columns = get_column_transformer("impute", request_body.categorical_columns, strategy="most_frequent")
        request_body.transformers.append((name, transformer, columns))

    if request_body.numeric_columns:
        name, transformer, columns = get_column_transformer("impute", request_body.numeric_columns, strategy="median")
        request_body.transformers.append((name, transformer, columns))

# def apply_pca(request_body, variance_retained=0.95):
#     # Initialize PCA
#     pca = PCA(n_components=variance_retained)
#
#     # Fit PCA on training data
#     request_body.X_train = pca.fit_transform(request_body.X_train)
#
#     # Transform test data using the same PCA model
#     request_body.X_test = pca.transform(request_body.X_test)
#
#     # Optionally store explained variance ratio for later reference


def apply_pca(request_body, variance_retained=0.95):
    if request_body.numeric_columns:
        name, transformer, columns = get_column_transformer("pca", "all", variance_retained=variance_retained)
        columns = columns if columns else list(range(request_body.X_train.shape[1]))
        request_body.transformers.append((name, transformer, columns))

#TODO - have to look at encode_categorical as there is also determine_column_types - also if want this to place before or
# after train test split
def encode_categorical(request_body):
    # Ensure ordinal_columns is a list, defaulting to an empty list if None
    ordinal_columns = request_body.ordinal_columns or []

    # Ensure nominal_columns is a list, defaulting to categorical columns excluding ordinal ones
    nominal_columns = request_body.nominal_columns or request_body.X.select_dtypes(
        include=["object", "category"]).columns.difference(ordinal_columns).tolist()

    # Extend nominal_columns with columns having fewer than 10 unique values
    nominal_columns.extend([column for column in request_body.X.columns if request_body.X[column].nunique() < 10])

    bool_columns = [col for col in request_body.X.columns if request_body.X[col].dtype == bool]
    nominal_columns = list(set(nominal_columns).union(bool_columns))
    request_body.nominal_columns = nominal_columns

    # Apply one-hot encoding using pd.get_dummies for nominal columns
    if nominal_columns:
        name, transformer, columns = get_column_transformer("encode", nominal_columns)
        request_body.transformers.append((name, transformer, columns))

    # Apply label encoding for ordinal columns
    if ordinal_columns:
        name, transformer, columns = get_column_transformer("encode", nominal_columns, encoding_type="ordinal")
        request_body.transformers.append((name, transformer, columns))

    # request_body.categorical_columns.extend(request_body.nominal_columns + request_body.ordinal_columns)

#
# def handle_imbalance(request_body):
#     from imblearn.over_sampling import SMOTE
#     from imblearn.under_sampling import TomekLinks
#     from imblearn.combine import SMOTETomek
#
#     technique = request_body.handle_imbalanced_data.upper()
#     if technique == "SMOTE":
#         resampler = SMOTE(random_state=42)
#     elif technique == "TOMEK":
#         resampler = TomekLinks()
#     elif technique == "SMOTETOMEK":
#         resampler = SMOTETomek(random_state=42)
#     else:
#         raise ValueError("Unsupported imbalance technique.")
#
#     # X_resampled, y_resampled = resampler.fit_resample(request_body.X, request_body.y)
#     # request_body.X = pd.DataFrame(X_resampled, columns=request_body.X.columns)
#     # request_body.y = pd.Series(y_resampled)
#     request_body.pipeline_steps.append(("sampling", resampler))
