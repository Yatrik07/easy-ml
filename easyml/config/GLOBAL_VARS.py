from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor

from typing import Optional, List, Any
import pandas as pd
from sklearn.compose import ColumnTransformer
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print("Project root:", PROJECT_ROOT)

save_data_path = "data.csv"
# ---------------------------------------- ML - Regression Models ----------------------------------------------
AVAILABLE_ML_MODELS_LIST_FOR_REGRESSION = [
    LinearRegression(),
    RandomForestRegressor(),
    XGBRegressor(),
    LGBMRegressor(),
    CatBoostRegressor(),
    KNeighborsRegressor(),
    AdaBoostRegressor()
]

DEFAULT_ML_MODELS_LIST_FOR_REGRESSION = [
    LinearRegression(),
    RandomForestRegressor(),
    LGBMRegressor(),
    KNeighborsRegressor(),
]

PARAM_GRID_LEVEL1_FOR_REGRESSION = {
    "LinearRegression": {
        "fit_intercept": [True, False]
    },
    "RandomForestRegressor": {
        "n_estimators": [10, 50],
        "max_depth": [None, 10]
    },
    "XGBRegressor": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1]
    },
    "LGBMRegressor": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1]
    },
    "CatBoostRegressor": {
        "iterations": [100, 200],
        "learning_rate": [0.01, 0.1]
    },
    "KNeighborsRegressor": {
        "n_neighbors": [3, 5],
        "weights": ["uniform", "distance"]
    },
    "AdaBoostRegressor": {
        "n_estimators": [50, 100],
        "learning_rate": [0.01, 0.1]
    }
}

PARAM_GRID_LEVEL2_FOR_REGRESSION = {
    "LinearRegression": {
        "fit_intercept": [True, False],
        "normalize": [True, False]
    },
    "RandomForestRegressor": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False]
    },
    "XGBRegressor": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 6, 10],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "gamma": [0, 0.1, 0.5]
    },
    "LGBMRegressor": {
        "n_estimators": [100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [-1, 10, 20],
        "num_leaves": [31, 50, 100],
        "boosting_type": ["gbdt", "dart"]
    },
    "CatBoostRegressor": {
        "iterations": [100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "depth": [6, 10, 12],
        "l2_leaf_reg": [1, 3, 5],
        "bagging_temperature": [0, 0.5, 1]
    },
    "KNeighborsRegressor": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]
    },
    "AdaBoostRegressor": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "loss": ["linear", "square", "exponential"]
    }
}

# ---------------------------------------- ML - Classification Models ----------------------------------------------

AVAILABLE_ML_MODELS_LIST_FOR_CLASSIFICATION = [
    LogisticRegression(),
    RandomForestClassifier(),
    XGBClassifier(),
    LGBMClassifier(),
    CatBoostClassifier(),
    KNeighborsClassifier(),
    AdaBoostClassifier()
]

DEFAULT_ML_MODELS_LIST_FOR_CLASSIFICATION = [
    LogisticRegression(),
    RandomForestClassifier(),
    LGBMClassifier(),
    KNeighborsClassifier(),
]


PARAM_GRID_LEVEL1_FOR_CLASSIFICATION = {
    "LogisticRegression": {
        "penalty": ["l2"],  # Regularization type
        "C": [0.1, 1]       # Regularization strength
    },
    "RandomForestClassifier": {
        "n_estimators": [10, 50],
        "max_depth": [None, 10]
    },
    "XGBClassifier": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1]
    },
    "LGBMClassifier": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1]
    },
    "CatBoostClassifier": {
        "iterations": [100, 200],
        "learning_rate": [0.01, 0.1]
    },
    "KNeighborsClassifier": {
        "n_neighbors": [3, 5],
        "weights": ["uniform", "distance"]
    },
    "AdaBoostClassifier": {
        "n_estimators": [50, 100],
        "learning_rate": [0.01, 0.1]
    }
}


PARAM_GRID_LEVEL2_FOR_CLASSIFICATION = {
    "LogisticRegression": {
        "penalty": ["l2"],  # Regularization type
        "C": [0.1, 1, 10]   # Regularization strength
    },
    "RandomForestClassifier": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False]
    },
    "XGBClassifier": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 6, 10],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "gamma": [0, 0.1, 0.5]
    },
    "LGBMClassifier": {
        "n_estimators": [100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [-1, 10, 20],
        "num_leaves": [31, 50, 100],
        "boosting_type": ["gbdt", "dart"]
    },
    "CatBoostClassifier": {
        "iterations": [100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "depth": [6, 10, 12],
        "l2_leaf_reg": [1, 3, 5],
        "bagging_temperature": [0, 0.5, 1]
    },
    "KNeighborsClassifier": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]
    },
    "AdaBoostClassifier": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "algorithm": ["SAMME", "SAMME.R"]
    }
}

# ---------------------------------------- ui2back_mappings ----------------------------------------------
# Classification Adaboost
ui2back_mappings_regression_models = {
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(),
    "xg_boost": XGBRegressor(),
    "light_gbm": LGBMRegressor(verbose=-1),
    "cat_boost": CatBoostRegressor(
        logging_level="Silent",  # Disables logs
    ),
    "knn": KNeighborsRegressor(),
    "ada_boost": AdaBoostRegressor()
}

ui2back_mappings_classification_models = {
    "logistic_regression": LogisticRegression(solver='newton-cg'),
    "random_forest": RandomForestClassifier(),
    "xg_boost": XGBClassifier(),
    "light_gbm": LGBMClassifier(verbose=-1),
    "cat_boost": CatBoostClassifier(
        task_type="CPU",
        logging_level="Silent",  # Disables logs
        save_snapshot=False,  # Prevents creating temporary files
    ),
    "knn": KNeighborsClassifier(),
    "ada_boost": AdaBoostClassifier()
}

error_code_mappings = {
    "DataIngestionError": 5001,
    "PreProcessingError": 5003,
    "ModelTrainingError": 5004,
    "DatabaseError": 5005,
    "InferenceError": 5006,
    "OtherError": 5007
}

# ----------------------------------------Database --------------------------------------------------
db_runs_columns_config = [
    "id INTEGER PRIMARY KEY AUTOINCREMENT",
    "run_name TEXT",
    "folder_path TEXT",
    "timestamp TEXT",
    "task TEXT"
]

# ---------------------------------------- Request Body ----------------------------------------------

# TODO: re-set all the required datatypes of the available variables in the class - ML_RequestBody
class ML_RequestBody(BaseModel):
    task: str = ""
    name: str = ""

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
    categorical_columns: Optional[List[str]] = []
    ordinal_columns: Optional[List[str]] = []
    nominal_columns: Optional[List[str]] = []
    numeric_columns: Optional[List[str]] = []
    imputation: bool = False
    reduce_features: bool = False
    ignore_columns: Optional[List[str]] = None

    hyper_parameter_tuning: int = 0
    models: Optional[List[str]] = []  # List of model names
    performance: Optional[dict] = {}
    path_to_save: Optional[str] = None

    column_transformer: ColumnTransformer = ColumnTransformer(transformers=[], remainder='passthrough')
    transformers: Optional[List] = []

    status_code: int = 200
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None

    timestamp: str = ""

    class Config:
        arbitrary_types_allowed = True  # Allows non-standard types like DataFrame

    def jsonify(self):
        to_dump = dict()
        to_dump['task'] = self.task
        to_dump['dataset_path'] = self.dataset_path
        to_dump['target_column'] = self.target_column
        to_dump["preprocessing"] = self.preprocessing
        to_dump["scaling"] = self.scaling
        to_dump["categorical_columns"] = self.categorical_columns
        to_dump["ordinal_columns"] = self.ordinal_columns
        to_dump["nominal_columns"] = self.nominal_columns
        to_dump["numeric_columns"] = self.numeric_columns
        to_dump["imputation"] = self.imputation
        to_dump["reduce_features"] = self.reduce_features
        to_dump["hyper_parameter_tuning"] = self.hyper_parameter_tuning
        to_dump["ignore_columns"] = self.ignore_columns
        to_dump["models"] = self.models
        to_dump["performance"] = self.performance
        to_dump["status_code"] = self.status_code
        to_dump["target_encodings"] = self.target_encodings
        return to_dump

class ML_InferenceBody(BaseModel):
    task: str = ""
    run_id: int = ""
    data: Any = None

    target_encodings: Optional[dict] = None

    ignore_columns: Optional[List[str]] = None

    model: Any = None
    model_name: str = None
    pipeline: Any = None

    status_code: int = 200
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True  # Allows non-standard types like DataFrame

    def jsonify(self):
        pass
