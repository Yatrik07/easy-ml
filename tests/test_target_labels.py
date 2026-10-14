"""Classification predictions keep the type of the original labels (0/1 stays int, not "0"/"1")."""

import io
import json
import os
import shutil
import zipfile

import numpy as np
import pandas as pd
import pytest

import easyml
from easyml.app import main
from easyml.trainer import MLTrainer
from easyml.utils.labels import decoder_from_config, labels_in_code_order


@pytest.fixture(scope="module")
def binary_csv(data_dir):
    rng = np.random.default_rng(1)
    n = 120
    x1 = rng.normal(size=n)
    x2 = rng.choice(["a", "b"], n)
    frame = pd.DataFrame({"x1": x1, "x2": x2, "outcome": (x1 > 0).astype(int)})
    path = os.path.join(data_dir, "uploads", "binary_numeric.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def binary_run(binary_csv):
    return MLTrainer(task="MLClassification", name="binary-int", dataset_path=binary_csv, target_column="outcome",
                     models=["logistic_regression"]).run()


def test_python_api_returns_ints(binary_run, binary_csv):
    model = easyml.load_model(binary_run.run_id)
    predictions = model.predict(binary_csv)
    assert set(predictions) <= {0, 1}
    assert all(type(p) is int for p in predictions)


def test_http_inference_returns_ints(client, binary_run):
    response = client.post("/inference", json={"run_id": binary_run.run_id, "model_name": "LogisticRegression",
                                                "data": {"x1": "2.5", "x2": "a"}})
    assert response.status_code == 200, response.data[:300]
    assert response.get_json()["prediction"] == 1
    assert isinstance(response.get_json()["prediction"], int)


def test_cli_writes_ints(capsys, binary_run, binary_csv):
    assert main(["predict", str(binary_run.run_id), binary_csv]) == 0
    frame = pd.read_csv(io.StringIO(capsys.readouterr().out))
    assert frame["Prediction"].dtype.kind == "i"


def test_old_runs_without_target_labels_still_return_ints(binary_run, binary_csv):
    record = easyml.load_model(binary_run.run_id)
    config_path = os.path.join(record._folder, "config_json.json")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    backup = config_path + ".bak"
    shutil.copy(config_path, backup)
    try:
        config.pop("target_labels"), config.pop("target_dtype")
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(config, handle)
        assert set(easyml.load_model(binary_run.run_id).predict(binary_csv)) <= {0, 1}
    finally:
        shutil.move(backup, config_path)


def test_export_ships_typed_labels(client, binary_run):
    response = client.post("/export", json={"run_id": binary_run.run_id, "model_name": "LogisticRegression"})
    assert response.status_code == 200
    archive = zipfile.ZipFile(io.BytesIO(response.data))
    assert json.loads(archive.read("labels.json")) == [0, 1]
    compile(archive.read("main.py").decode(), "main.py", "exec")


def test_text_runs_with_numeric_labels(data_dir):
    rows = [("great product, works well", 5), ("terrible, broke after a day", 1)] * 20
    path = os.path.join(data_dir, "uploads", "stars.csv")
    pd.DataFrame(rows, columns=["review", "stars"]).to_csv(path, index=False)
    run = MLTrainer(task="NLPTextClassification", name="stars", dataset_path=path, target_column="stars",
                    text_column="review", models=["logistic_regression"], min_df=1).run()
    assert easyml.load_model(run.run_id).predict(["works great"]) == [5]


@pytest.mark.parametrize("encodings, expected", [
    ({"0": 0, "1": 1}, {0: 0, 1: 1}),
    ({"0.5": 0, "1.5": 1}, {0: 0.5, 1: 1.5}),
    ({"false": 0, "true": 1}, {0: False, 1: True}),
    ({"001": 0, "1": 1}, {0: "001", 1: "1"}),
    ({"high": 0, "low": 1}, {0: "high", 1: "low"}),
])
def test_legacy_decoder_only_casts_exact_matches(encodings, expected):
    assert decoder_from_config({"target_encodings": encodings}) == expected


def test_target_labels_win_over_encodings():
    config = {"target_labels": [10, 20], "target_encodings": {"10": 0, "20": 1}}
    assert decoder_from_config(config) == {0: 10, 1: 20}
    assert labels_in_code_order({np.int64(7): 1, np.int64(3): 0}) == [3, 7]
    assert decoder_from_config({}) == {}
