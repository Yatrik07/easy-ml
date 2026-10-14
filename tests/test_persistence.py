"""Run storage added in 0.6.0: database migration, fuller config files, lineage fields, readable settings."""

import json
import os
import sqlite3
from contextlib import closing

import pytest

from easyml.trainer import MLTrainer
from easyml.utils import database_operations
from easyml.utils.run_summary import settings_for_display, summarize_run


def test_old_database_gains_the_new_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(database_operations, "db_dir", str(tmp_path))
    with closing(sqlite3.connect(tmp_path / "runs.db")) as conn:
        conn.execute("CREATE TABLE runs(id INTEGER PRIMARY KEY AUTOINCREMENT, run_name TEXT, folder_path TEXT, "
                     "timestamp TEXT, task TEXT, dataset_name TEXT, metrics_json TEXT)")
        conn.execute("INSERT INTO runs (run_name, task) VALUES ('old run', 'MLRegression')")
        conn.commit()

    database_operations.create_database_table("runs")
    database_operations.create_database_table("runs")  # running it twice is harmless

    with closing(sqlite3.connect(tmp_path / "runs.db")) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(runs)")]
    assert {"parent_run_id", "origin"} <= set(columns)
    rows = database_operations.get_data_from_database()["data"]
    assert rows[0]["run_name"] == "old run" and rows[0]["parent_run_id"] is None


@pytest.fixture(scope="module")
def tabular_run(classification_csv):
    return MLTrainer(task="MLClassification", name="persist", dataset_path=classification_csv, target_column="label",
                     models=["logistic_regression"], categorical_columns=["city"],
                     original_dataset_name="customers.csv").run()


def test_config_records_what_retraining_needs(tabular_run):
    with open(os.path.join(tabular_run.path_to_save, "config_json.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    assert config["config_version"] == 2 and config["easyml_version"]
    assert config["name"] == "persist" and config["original_dataset_name"] == "customers.csv"
    assert config["requested_categorical_columns"] == ["city"]
    assert "city" in config["categorical_columns"]
    assert config["target_labels"] == ["high", "low", "mid"] and config["parent_run_id"] is None


def test_runs_list_has_lineage_fields(client, tabular_run):
    runs = client.get("/runs").get_json()["data"]
    mine = next(r for r in runs if r["id"] == tabular_run.run_id)
    assert mine["parent_run_id"] is None and mine["origin"] is None
    assert summarize_run({"id": 3, "task": "MLRegression", "parent_run_id": 7.0, "origin": "retrain"})["parent_run_id"] == 7


def test_settings_are_readable():
    rows = settings_for_display("MLClassification", {
        "target_column": "label", "imputation": False, "scaling": True, "hyper_parameter_tuning": 1,
        "models": ["random_forest"], "imbalance_handling": "smote", "ignore_columns": [],
    })
    values = {r["key"]: r["value"] for r in rows}
    assert values == {"target_column": "label", "imputation": "Drop rows", "scaling": "Yes",
                      "hyper_parameter_tuning": "Quick", "models": "Random Forest",
                      "imbalance_handling": "SMOTE oversampling", "ignore_columns": "None"}
    regression = {r["key"] for r in settings_for_display("MLRegression", {"imbalance_handling": "none"})}
    assert "imbalance_handling" not in regression
