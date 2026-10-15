"""
Plain constants shared across Easy-ML, with no third-party imports.

Modules that only need one of these (the CLI, the run database, artifact helpers) import them from
here, so they do not pull in scikit-learn and the boosting libraries that ``GLOBAL_VARS`` loads.
``GLOBAL_VARS`` re-exports them for older imports.
"""

error_code_mappings = {
    "DataIngestionError": 5001,
    "PreProcessingError": 5003,
    "ModelTrainingError": 5004,
    "DatabaseError": 5005,
    "InferenceError": 5006,
    "OtherError": 5007,
    "NLPDataIngestionError": 6000,
    "NLPPreProcessingError": 6001,
    "NLPFeatureEngineeringError": 6002,
    "NLPModelTrainingError": 6003,
    "NLPInferenceError": 6004,
}

# ----------------------------------------Database --------------------------------------------------
db_runs_columns_config = [
    "id INTEGER PRIMARY KEY AUTOINCREMENT",
    "run_name TEXT",
    "folder_path TEXT",
    "timestamp TEXT",
    "task TEXT",
    "dataset_name TEXT",
    "metrics_json TEXT",
    # Added in 0.6.0; older databases get them through the migration in create_database_table.
    "parent_run_id INTEGER",  # the run this one was retrained from
    "origin TEXT",            # NULL (trained here), "retrain" or "import"
]

# Data-prep recipes (0.8.0)
db_recipes_columns_config = [
    "id INTEGER PRIMARY KEY AUTOINCREMENT",
    "name TEXT",
    "steps_json TEXT",
    "source_columns_json TEXT",
    "created_at TEXT",
    "updated_at TEXT",
]

# Bumped when config_json.json gains keys that later code relies on (retrain, export).
CONFIG_VERSION = 2

# Model classes saved with CatBoost's native format (.cbm) instead of joblib (.pkl)
CATBOOST_MODEL_NAMES = {"CatBoostClassifier", "CatBoostRegressor"}
