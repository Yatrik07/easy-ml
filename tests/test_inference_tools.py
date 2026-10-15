"""Prediction page helpers: a real row to try the model with, and SHAP explanations of binary models."""

import os

import numpy as np
import pandas as pd
import pytest

from easyml.trainer import MLTrainer
from easyml.utils.database_operations import get_record_by_id
from easyml.utils.holdout import read_holdout


@pytest.fixture(scope="module")
def people_run(data_dir):
    rng = np.random.default_rng(3)
    n = 240
    frame = pd.DataFrame({"alone": rng.integers(0, 11, n).astype(float), "events": rng.integers(0, 10, n).astype(float),
                          "drained": rng.choice(["Yes", "No"], n), "outside": rng.choice([0.0, 1.0, 2.0], n)})
    frame.loc[::17, "alone"] = np.nan
    frame["kind"] = np.where(frame["alone"].fillna(5) + (frame["drained"] == "Yes") * 3 > 6, "Introvert", "Extrovert")
    path = os.path.join(data_dir, "uploads", "people.csv")
    frame.to_csv(path, index=False)
    return MLTrainer(task="MLClassification", name="people", dataset_path=path, target_column="kind",
                     models=["logistic_regression", "random_forest"]).run()


def test_sample_is_a_test_row_the_model_can_predict(client, people_run):
    folder = get_record_by_id(people_run.run_id)["folder_path"]
    data = pd.read_csv(os.path.join(folder, "data.csv"))
    seen = set()
    for _ in range(12):
        sample = client.get(f"/runs/{people_run.run_id}/sample").get_json()
        assert sample["source"] == "test" and sample["row"] in read_holdout(folder)
        assert set(sample["values"]) == {"alone", "events", "drained", "outside"}
        row = data.iloc[sample["row"]]
        assert sample["actual"] == row["kind"]
        assert sample["values"]["alone"] == (None if pd.isna(row["alone"]) else row["alone"])
        seen.add(sample["row"])
        predicted = client.post("/inference", json={"run_id": people_run.run_id, "model_name": "RandomForestClassifier",
                                                    "data": sample["values"]})
        assert predicted.status_code == 200
    assert len(seen) > 1  # random, not always the same row
    assert client.get("/runs/999999/sample").status_code == 404


def test_sample_of_a_text_run_and_an_older_run(client, text_csv, people_run):
    text_run = MLTrainer(task="NLPTextClassification", name="topics-sample", dataset_path=text_csv, target_column="label",
                         text_column="text", models=["logistic_regression"]).run()
    sample = client.get(f"/runs/{text_run.run_id}/sample").get_json()
    assert sample["text"] and sample["actual"] in ("sports", "tech") and sample["source"] == "test"
    # A run from before 0.9.0 has no holdout.json: any row of its data is used.
    folder = get_record_by_id(people_run.run_id)["folder_path"]
    holdout = os.path.join(folder, "holdout.json")
    saved = open(holdout, encoding="utf-8").read()
    os.remove(holdout)
    try:
        assert client.get(f"/runs/{people_run.run_id}/sample").get_json()["source"] == "training"
    finally:
        open(holdout, "w", encoding="utf-8").write(saved)


@pytest.mark.parametrize("model_name", ["LogisticRegression", "RandomForestClassifier"])
def test_binary_explanations_describe_the_predicted_class(client, people_run, model_name):
    pytest.importorskip("shap")
    rows = [{"alone": 9, "events": 1, "drained": "Yes", "outside": 0.0},   # clearly Introvert
            {"alone": 1, "events": 9, "drained": "No", "outside": 2.0}]    # clearly Extrovert
    labels = set()
    for row in rows:
        predicted = client.post("/inference", json={"run_id": people_run.run_id, "model_name": model_name,
                                                    "data": row}).get_json()
        labels.add(predicted["prediction"])
        explanation = client.post("/explain", json={"run_id": people_run.run_id, "model_name": model_name,
                                                    "data": row}).get_json()["explanation"]
        if explanation["scale"] == "log-odds":
            # The score of the predicted class: positive for a binary prediction, whichever class it is.
            assert explanation["output_value"] > 0
            assert abs(1 / (1 + np.exp(-explanation["output_value"])) - predicted["probability"]) < 1e-3
        else:
            assert abs(explanation["output_value"] - predicted["probability"]) < 1e-4
        assert explanation["features"][0]["name"] in ("alone", "drained")  # the columns that decide the label
    assert labels == {"Introvert", "Extrovert"}
