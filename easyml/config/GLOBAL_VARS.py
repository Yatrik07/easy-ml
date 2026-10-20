from pydantic import BaseModel, Field, ConfigDict
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              AdaBoostClassifier, AdaBoostRegressor)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (confusion_matrix, classification_report, mean_absolute_percentage_error,
                            mean_absolute_error, r2_score, accuracy_score)
from functools import partial
from typing import Optional, List, Any
import pandas as pd
from sklearn.compose import ColumnTransformer
from easyml.config.paths import PACKAGE_ROOT
from easyml.config.constants import (  # noqa: F401 (re-exported for older imports)
    CATBOOST_MODEL_NAMES,
    CONFIG_VERSION,
    db_runs_columns_config,
    error_code_mappings,
)

# Gradient-boosting libraries ship in the optional [boost] extra.
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except (ImportError, OSError):
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError):
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except (ImportError, OSError):
    CATBOOST_AVAILABLE = False

# Keep PROJECT_ROOT as alias for backward compatibility (points to package dir)
PROJECT_ROOT = PACKAGE_ROOT

REGRESSION_METRICS = ['r2_score', 'mape', 'mae']
CLASSIFICATION_METRICS = ['accuracy', 'confusion_matrix', 'classification_report']
METRICS_MAPPING = {
        'accuracy' : accuracy_score,
        'classification_report' : partial(classification_report, output_dict=True),
        'confusion_matrix' : confusion_matrix,
        'r2_score' : r2_score,
        'mae' : mean_absolute_error,
        'mape' : mean_absolute_percentage_error
    }

# ---------------------------------------- ML - Regression Models ----------------------------------------------
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
        "copy_X": [True, False]
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
        "learning_rate": [0.01, 0.05, 0.1]
    }
}

# ---------------------------------------- ui2back_mappings ----------------------------------------------
# Only models whose libraries are installed are registered, so /get_models reflects the installation.
ui2back_mappings_regression_models = {
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(),
}
ui2back_mappings_classification_models = {
    "logistic_regression": LogisticRegression(solver='newton-cg', max_iter=1000),
    "random_forest": RandomForestClassifier(),
}
if XGBOOST_AVAILABLE:
    ui2back_mappings_regression_models["xg_boost"] = XGBRegressor()
    ui2back_mappings_classification_models["xg_boost"] = XGBClassifier()
if LIGHTGBM_AVAILABLE:
    ui2back_mappings_regression_models["light_gbm"] = LGBMRegressor(verbose=-1)
    ui2back_mappings_classification_models["light_gbm"] = LGBMClassifier(verbose=-1)
if CATBOOST_AVAILABLE:
    ui2back_mappings_regression_models["cat_boost"] = CatBoostRegressor(logging_level="Silent")
    ui2back_mappings_classification_models["cat_boost"] = CatBoostClassifier(
        task_type="CPU",
        logging_level="Silent",
        save_snapshot=False,  # Prevents creating temporary files
    )
ui2back_mappings_regression_models["knn"] = KNeighborsRegressor()
ui2back_mappings_regression_models["ada_boost"] = AdaBoostRegressor()
ui2back_mappings_classification_models["knn"] = KNeighborsClassifier()
ui2back_mappings_classification_models["ada_boost"] = AdaBoostClassifier()

DEFAULT_ML_MODELS_FOR_REGRESSION = ["linear_regression", "random_forest", "light_gbm", "knn"]
DEFAULT_ML_MODELS_FOR_CLASSIFICATION = ["logistic_regression", "random_forest", "light_gbm", "knn"]



def get_model_mappings(task: str):
    return ui2back_mappings_regression_models if task == "regression" else ui2back_mappings_classification_models


def get_ml_model_instance(task: str, model_name: str):
    model = get_model_mappings(task).get(model_name)
    if model is None:
        raise ValueError(f"Unknown or unavailable model '{model_name}' for task '{task}'")
    return model.__class__(**model.get_params())


def get_default_ml_models(task: str):
    defaults = DEFAULT_ML_MODELS_FOR_REGRESSION if task == "regression" else DEFAULT_ML_MODELS_FOR_CLASSIFICATION
    return [get_ml_model_instance(task, name) for name in defaults if name in get_model_mappings(task)]


def get_param_grid(task: str, level: int) -> dict:
    grids = {
        ("regression", 1): PARAM_GRID_LEVEL1_FOR_REGRESSION,
        ("regression", 2): PARAM_GRID_LEVEL2_FOR_REGRESSION,
        ("classification", 1): PARAM_GRID_LEVEL1_FOR_CLASSIFICATION,
        ("classification", 2): PARAM_GRID_LEVEL2_FOR_CLASSIFICATION,
    }
    return grids.get((task, int(level or 0)), {})

# error_code_mappings, db_runs_columns_config and CONFIG_VERSION live in config/constants.py.

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
    # 0.13, several target columns (core/multi_target.py): all of them, their weights and per-column classes.
    # Empty for a run with one target, which keeps using target_column / target_encodings only.
    target_columns: List[str] = Field(default_factory=list)
    target_weights: dict = Field(default_factory=dict)
    target_encodings_by_column: dict = Field(default_factory=dict)

    X_train: Optional[pd.DataFrame] = None
    X_test: Optional[pd.DataFrame] = None
    y_train: Optional[pd.Series] = None
    y_test: Optional[pd.Series] = None

    preprocessing: bool = False
    scaling: bool = False
    categorical_columns: Optional[List[str]] = Field(default_factory=list)
    ordinal_columns: Optional[List[str]] = Field(default_factory=list)
    nominal_columns: Optional[List[str]] = Field(default_factory=list)
    numeric_columns: Optional[List[str]] = Field(default_factory=list)
    imputation: bool = False
    reduce_features: bool = False
    ignore_columns: Optional[List[str]] = None

    hyper_parameter_tuning: int = 0
    imbalance_handling: str = "none"  # "none", "smote", "tomek", "smotetomek"
    models: Optional[List[str]] = Field(default_factory=list)  # List of model names
    performance: Optional[dict] = Field(default_factory=dict)
    path_to_save: Optional[str] = None

    column_transformer: ColumnTransformer = Field(
        default_factory=lambda: ColumnTransformer(transformers=[], remainder='passthrough')
    )
    transformers: Optional[List] = Field(default_factory=list)

    status_code: int = 200
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None

    timestamp: str = ""
    run_id: Optional[int] = None
    original_dataset_name: Optional[str] = None
    target_dtype: Optional[str] = None
    parent_run_id: Optional[int] = None
    recipe: Optional[dict] = None  # the data-prep recipe that produced the training data, if any
    source_rows: Any = None  # data.csv row of each training-frame row when rows were dropped (holdout.json)
    random_seed: int = 42  # split, resampling, CV folds and every model with a seed (core/seeding.py)
    tags: Optional[List[str]] = None  # the new run's tags (a retrain inherits its parent's)
    cv_folds: int = 0      # k-fold cross-validation of each model on the training data (0 = off)
    X_train_raw: Optional[pd.DataFrame] = None  # the training rows before preprocessing, for cross-validation
    # 0.12, the final model: trained on every row of this evaluated run, with its best settings (no test set).
    final_of_run: Optional[int] = None
    fixed_params: dict = Field(default_factory=dict)  # model class -> the evaluated run's best parameters
    estimates: dict = Field(default_factory=dict)     # model class -> the evaluated run's test scores
    requested_categorical_columns: Optional[List[str]] = Field(default_factory=list)
    # Kept in memory for the result charts (evaluation.json); never serialized.
    pipeline: Any = None
    X_test_raw: Optional[pd.DataFrame] = None
    charts: dict = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def jsonify(self):
        from easyml import __version__
        from easyml.utils.labels import labels_in_code_order

        to_dump = dict()
        to_dump["config_version"] = CONFIG_VERSION
        to_dump["easyml_version"] = __version__
        to_dump['task'] = self.task
        to_dump["name"] = self.name
        to_dump["original_dataset_name"] = self.original_dataset_name
        to_dump["parent_run_id"] = self.parent_run_id
        to_dump["requested_categorical_columns"] = self.requested_categorical_columns
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
        to_dump["imbalance_handling"] = self.imbalance_handling
        to_dump["ignore_columns"] = self.ignore_columns
        to_dump["models"] = self.models
        to_dump["performance"] = self.performance
        to_dump["status_code"] = self.status_code
        to_dump["target_encodings"] = self.target_encodings
        to_dump["target_labels"] = labels_in_code_order(self.target_encodings)
        to_dump["target_dtype"] = self.target_dtype
        to_dump["run_id"] = getattr(self, "run_id", None)
        to_dump["random_seed"] = self.random_seed
        to_dump["cv_folds"] = self.cv_folds
        if len(self.target_columns) > 1:
            to_dump["target_columns"] = self.target_columns
            to_dump["target_weights"] = self.target_weights
            if self.target_encodings_by_column:
                to_dump["target_classes"] = {target: labels_in_code_order(encodings)
                                             for target, encodings in self.target_encodings_by_column.items()}
        if self.recipe:
            to_dump["recipe"] = self.recipe
        if self.final_of_run:
            to_dump["final_of_run"] = self.final_of_run
        return to_dump

class ML_InferenceBody(BaseModel):
    task: str = ""
    run_id: Optional[int] = None
    data: Any = None

    target_encodings: Optional[dict] = None

    ignore_columns: Optional[List[str]] = None

    model: Any = None
    model_name: Optional[str] = None
    pipeline: Any = None

    status_code: int = 200
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def jsonify(self):
        pass
