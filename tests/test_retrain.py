"""Retraining a saved run on new data: Python API, HTTP endpoints and the `easy-ml retrain` command."""

import json
import os
import time

import numpy as np
import pandas as pd
import pytest

import easyml
from easyml.app import main
from easyml.retraining import DatasetMismatchError, plan_retrain
from easyml.trainer import MLTrainer
from easyml.utils.cancellation import get_job
from easyml.utils.database_operations import RunNotFoundError, get_record_by_id


def _config(run_id):
    folder = get_record_by_id(run_id)["folder_path"]
    with open(os.path.join(folder, "config_json.json"), encoding="utf-8") as handle:
        return json.load(handle)


def _column_data(run_id):
    folder = get_record_by_id(run_id)["folder_path"]
    with open(os.path.join(folder, "column_data.json"), encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def parent(classification_csv):
    return MLTrainer(task="MLClassification", name="churn", dataset_path=classification_csv, target_column="label",
                     models=["logistic_regression", "random_forest"], scaling=True, imputation=True).run()


@pytest.fixture(scope="module")
def new_data(classification_csv, data_dir):
    """Same columns in another order, plus a column the parent never saw and more distinct 'rooms' values."""
    frame = pd.read_csv(classification_csv).sample(frac=1.0, random_state=3).reset_index(drop=True)
    frame["noise"] = np.arange(len(frame))
    frame["rooms"] = (np.arange(len(frame)) % 15) + 1
    frame = frame[["noise", "label"] + [c for c in frame.columns if c not in ("noise", "label")]]
    path = os.path.join(data_dir, "uploads", "churn_new.csv")
    frame.to_csv(path, index=False)
    return path


def test_retrain_keeps_the_input_schema(parent, new_data):
    result = easyml.retrain(parent.run_id, new_data, name="churn-oct")
    assert result.parent_run_id == parent.run_id and result.run_name == "churn-oct"
    assert set(result.performance) == set(parent.performance)
    assert any("noise" in w for w in result.warnings)

    record = get_record_by_id(result.run_id)
    assert record["parent_run_id"] == parent.run_id and record["origin"] == "retrain"
    assert _column_data(result.run_id)["column_dtypes"].keys() == _column_data(parent.run_id)["column_dtypes"].keys()
    new_config, old_config = _config(result.run_id), _config(parent.run_id)
    assert new_config["categorical_columns"] == old_config["categorical_columns"]
    assert new_config["scaling"] is True and new_config["parent_run_id"] == parent.run_id

    row = {"age": 40, "income": 50000, "city": "NY", "owns_car": "yes", "rooms": 3}
    assert result.load_model().predict(row)[0] in {"high", "mid", "low"}
    assert "RunResult(" in repr(result)


def test_missing_columns_are_reported(parent, classification_csv, data_dir):
    path = os.path.join(data_dir, "uploads", "churn_missing.csv")
    pd.read_csv(classification_csv).drop(columns=["income", "city"]).to_csv(path, index=False)
    with pytest.raises(DatasetMismatchError) as caught:
        plan_retrain(parent.run_id, path)
    assert caught.value.missing_columns == ["income", "city"]
    assert "label" in caught.value.required_columns


def test_bad_targets_and_models_are_rejected(parent, classification_csv, data_dir):
    path = os.path.join(data_dir, "uploads", "churn_one_class.csv")
    frame = pd.read_csv(classification_csv)
    frame["label"] = "high"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="two different classes"):
        plan_retrain(parent.run_id, path)
    with pytest.raises(ValueError, match="not available"):
        plan_retrain(parent.run_id, classification_csv, models=["no_such_model"])
    with pytest.raises(RunNotFoundError):
        plan_retrain(999999, classification_csv)


def test_regression_target_must_be_numeric(regression_csv, data_dir):
    run = MLTrainer(task="MLRegression", name="reg-parent", dataset_path=regression_csv, target_column="y",
                    models=["linear_regression"]).run()
    path = os.path.join(data_dir, "uploads", "reg_text_target.csv")
    frame = pd.read_csv(regression_csv)
    frame["y"] = "high"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="must contain numbers"):
        plan_retrain(run.run_id, path)


def test_models_come_from_saved_scores_when_none_were_chosen(regression_csv):
    run = MLTrainer(task="MLRegression", name="reg-defaults", dataset_path=regression_csv, target_column="y").run()
    plan = plan_retrain(run.run_id, regression_csv)
    assert plan.kwargs["models"] and "linear_regression" in plan.kwargs["models"]


def test_text_runs_retrain(text_csv):
    run = MLTrainer(task="NLPTextClassification", name="topics", dataset_path=text_csv, target_column="label",
                    text_column="text", models=["logistic_regression"], min_df=1).run()
    result = easyml.retrain(run.run_id, text_csv)
    assert result.run_name == "topics-retrain"
    assert result.load_model().predict(["the striker scored a goal"]) == ["sports"]
    assert _config(result.run_id)["min_df"] == 1


def _wait(job_id, timeout=60):
    job = get_job(job_id)
    deadline = time.time() + timeout
    while not job.finished:
        assert time.time() < deadline, "retrain job did not finish"
        time.sleep(0.05)
    return job


def test_http_retrain(client, parent, new_data, data_dir):
    response = client.post(f"/runs/{parent.run_id}/retrain", json={
        "dataset_path": new_data, "original_filename": "C:\\data\\churn_october.csv", "name": "from-ui"})
    assert response.status_code == 202, response.data[:300]
    body = response.get_json()
    assert body["parent_run_id"] == parent.run_id and body["warnings"]

    job = _wait(body["job_id"])
    result = job.history[-1]["result"]
    assert result["status_code"] == 200
    record = get_record_by_id(result["run_id"])
    assert record["origin"] == "retrain" and record["dataset_name"] == "churn_october.csv"


def test_http_retrain_errors(client, parent, classification_csv, data_dir, tmp_path):
    outside = tmp_path / "outside.csv"
    pd.read_csv(classification_csv).to_csv(outside, index=False)
    assert client.post(f"/runs/{parent.run_id}/retrain", json={"dataset_path": str(outside)}).status_code == 403
    assert client.post("/runs/999999/retrain", json={"dataset_path": classification_csv}).status_code == 404

    path = os.path.join(data_dir, "uploads", "churn_missing_http.csv")
    pd.read_csv(classification_csv).drop(columns=["age"]).to_csv(path, index=False)
    response = client.post(f"/runs/{parent.run_id}/retrain", json={"dataset_path": path})
    assert response.status_code == 400
    assert response.get_json()["missing_columns"] == ["age"]


def test_config_endpoint_describes_the_run_without_paths(client, parent, data_dir):
    response = client.get(f"/runs/{parent.run_id}/config")
    assert response.status_code == 200
    spec = response.get_json()
    assert spec["retrain"]["supported"] and spec["retrain"]["input"] == "table"
    assert set(spec["retrain"]["required_columns"]) == {"age", "income", "city", "owns_car", "rooms", "label"}
    assert {s["key"] for s in spec["settings"]} >= {"target_column", "scaling", "models"}
    text = json.dumps(spec)
    assert data_dir not in text and data_dir.replace("\\", "\\\\") not in text
    assert client.get("/runs/999999/config").status_code == 404


def test_cli_retrain(capsys, parent, new_data, classification_csv, data_dir):
    assert main(["retrain", str(parent.run_id), new_data, "--name", "cli-retrain"]) == 0
    captured = capsys.readouterr()
    assert "retrained from run #" in captured.out and "MODEL" in captured.out
    assert "%]" in captured.err  # progress lines

    path = os.path.join(data_dir, "uploads", "churn_missing_cli.csv")
    pd.read_csv(classification_csv).drop(columns=["city"]).to_csv(path, index=False)
    assert main(["retrain", str(parent.run_id), path]) == 1
    assert "city" in capsys.readouterr().err


def test_image_runs_retrain(tmp_path):
    pytest.importorskip("torch")
    from PIL import Image
    from easyml.cv.cv_controllers import CVController

    root = tmp_path / "shapes"
    for label, color in (("red", (220, 20, 20)), ("blue", (20, 20, 220))):
        (root / label).mkdir(parents=True)
        for i in range(8):
            Image.new("RGB", (32, 32), color).save(root / label / f"{i}.png")
    parent_run = CVController(kwargs={"name": "cv-parent", "dataset_path": str(root), "models": ["SimpleCNN"],
                                      "epochs": 1, "batch_size": 4, "image_size": [32, 32]}).execute_pipeline()
    config = _config(parent_run["run_id"])
    assert config["epochs"] == 1 and config["batch_size"] == 4

    result = easyml.retrain(parent_run["run_id"], str(root))
    assert get_record_by_id(result.run_id)["parent_run_id"] == parent_run["run_id"]

    # Runs from before 0.6.0 did not save their training settings: defaults are used and reported.
    path = os.path.join(get_record_by_id(parent_run["run_id"])["folder_path"], "config_json.json")
    for key in ("epochs", "batch_size"):
        config.pop(key)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle)
    plan = plan_retrain(parent_run["run_id"], str(root))
    assert set(plan.assumed_defaults) >= {"epochs", "batch_size"}
