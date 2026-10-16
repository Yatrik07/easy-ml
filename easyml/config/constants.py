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
    # Added in 0.10.0: what people write about a run (utils/run_meta.py)
    "notes TEXT",
    "tags TEXT",              # a JSON list of strings
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

# Scheduled retraining (0.9.0): one row per schedule, and one per attempt that trained a model or failed.
db_schedules_columns_config = [
    "id INTEGER PRIMARY KEY AUTOINCREMENT",
    "name TEXT",
    "base_run_id INTEGER",       # the run the schedule was created from
    "champion_run_id INTEGER",   # the run currently in use (starts as the base run)
    "task TEXT",
    "trigger TEXT",              # "interval" | "daily" | "weekly" | "new_data"
    "interval_minutes INTEGER",
    "at_time TEXT",              # "HH:MM", local time (daily / weekly)
    "weekday INTEGER",           # 0 = Monday (weekly)
    "source_path TEXT",          # a file, or a folder (newest matching file; image runs: the image folder)
    "file_pattern TEXT",
    "recipe TEXT",               # "" = the run's own recipe, "none", or a recipe id
    "min_improvement REAL",
    "enabled INTEGER",
    "next_run_at TEXT",
    "last_run_at TEXT",
    "last_status TEXT",
    "last_message TEXT",
    "last_signature TEXT",       # the data the last attempt used, so unchanged data is not retrained
    "running_since TEXT",        # claim: set while a scheduler works on it
    "runner TEXT",
    "job_id TEXT",
    "created_at TEXT",
    "updated_at TEXT",
]

db_schedule_events_columns_config = [
    "id INTEGER PRIMARY KEY AUTOINCREMENT",
    "schedule_id INTEGER",
    "trigger TEXT",              # "time" | "new data" | "manual"
    "data_file TEXT",
    "started_at TEXT",
    "finished_at TEXT",
    "challenger_run_id INTEGER",
    "champion_before INTEGER",
    "champion_after INTEGER",
    "metric TEXT",
    "champion_score REAL",
    "challenger_score REAL",
    "compared_on TEXT",          # "same new data" | "own test sets"
    "decision TEXT",             # "promoted" | "kept champion" | "failed"
    "message TEXT",
]

# Bumped when config_json.json gains keys that later code relies on (retrain, export).
CONFIG_VERSION = 2

# Model classes saved with CatBoost's native format (.cbm) instead of joblib (.pkl)
CATBOOST_MODEL_NAMES = {"CatBoostClassifier", "CatBoostRegressor"}
