from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from typing import Optional, List
import pandas as pd
import numpy as np


DEFAULT_ML_MODELS_LIST_FOR_REGRESSION = [
    LinearRegression(),
    KNeighborsRegressor(),
    DecisionTreeRegressor(),
    SVR()
]

PARAM_GRID_LEVEL1_FOR_REGRESSION = {
    "LinearRegression": {
        "fit_intercept": [True, False]
    },
    "KNeighborsRegressor": {
        "n_neighbors": [3, 5],
        "weights": ["uniform", "distance"]
    },
    "DecisionTreeRegressor": {
        "max_depth": [None, 10],
        "min_samples_split": [2, 5]
    },
    "SVR": {
        "kernel": ["linear", "rbf"],
        "C": [1, 10]
    }
}

PARAM_GRID_LEVEL2_FOR_REGRESSION = {
    "LinearRegression": {
        "fit_intercept": [True, False],
    },
    "KNeighborsRegressor": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]
    },
    "DecisionTreeRegressor": {
        "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "SVR": {
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "C": [0.1, 1, 10],
        "epsilon": [0.01, 0.1, 0.5]
    }
}

DEFAULT_ML_MODELS_LIST_FOR_CLASSIFICATION = [
    LogisticRegression(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    SVC()
]

PARAM_GRID_LEVEL1_FOR_CLASSIFICATION = {
    "LogisticRegression": {
        "penalty": ["l2"],  # Regularization type
        "C": [0.1, 1]       # Regularization strength
    },
    "KNeighborsClassifier": {
        "n_neighbors": [3, 5],
        "weights": ["uniform", "distance"]
    },
    "DecisionTreeClassifier": {
        "max_depth": [None, 10],
        "min_samples_split": [2, 5]
    },
    "SVC": {
        "kernel": ["linear", "rbf"],
        "C": [1, 10]
    }
}

PARAM_GRID_LEVEL2_FOR_CLASSIFICATION = {
    "LogisticRegression": {
        "penalty": ["l2"],           # Regularization type
        "C": [0.1, 1, 10]            # Regularization strength
    },
    "KNeighborsClassifier": {
        "n_neighbors": [3, 5, 7],    # Number of neighbors
        "weights": ["uniform", "distance"],  # Weight function for prediction
        "metric": ["euclidean", "manhattan"] # Distance metric
    },
    "DecisionTreeClassifier": {
        "criterion": ["gini", "entropy"],  # Split criterion
        "max_depth": [None, 10],           # Maximum depth of tree
        "min_samples_split": [2, 5],       # Min samples required to split a node
        "min_samples_leaf": [1, 2]         # Min samples required to be at a leaf node
    },
    "SVC": {
        "kernel": ["linear", "rbf"],      # Kernel type
        "C": [0.1, 1, 10],                # Regularization parameter
        "gamma": ["scale", "auto"]        # Kernel coefficient (for 'rbf')
    }
}


# TODO: re-set all the required datatypes of the available variables in the class - ML_RequestBody
class ML_RequestBody(BaseModel):
    task: str = ""

    dataset_path: Optional[str] = None
    X: Optional[pd.DataFrame] = None
    y: Optional[pd.Series] = None
    df: Optional[pd.DataFrame] = None
    target_column: str = ""
    target_encodings: Optional[dict] = None

    X_train: Optional[pd.DataFrame] = None
    X_test: Optional[pd.DataFrame] = None
    y_train: Optional[pd.Series] = None
    y_test: Optional[pd.Series] = None

    preprocessing: bool = False
    scaling: bool = False
    categorical_columns: Optional[List[str]] = None
    ordinal_columns: Optional[List[str]] = None
    nominal_columns: Optional[List[str]] = None
    numeric_columns: Optional[List[str]] = None
    imputation: bool = False
    reduce_features: bool = False
    handle_imbalanced_data: str = ""
    ignore_columns: Optional[List[str]] = None

    hyper_parameter_tuning: int = 0
    models: Optional[List[str]] = []  # List of model names
    performance: Optional[dict] = {}
    path_to_save: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True  # Allows non-standard types like DataFrame

    def jsonify(self):
        # TODO: convert the entire body into the returnable json (dict) for frontend - ML_RequestBody
        pass


