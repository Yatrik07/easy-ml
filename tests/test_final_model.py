"""0.12: the final model, trained on all of an evaluated run's rows with its best settings."""

import json
import os
import time

import numpy as np
import pandas as pd
import pytest

import easyml
from easyml.retraining import plan_final
from easyml.trainer import MLTrainer
from easyml.utils.cancellation import get_job
from easyml.utils.database_operations import get_record_by_id


def _plain(value):
    from easyml.utils.utils import NumpyEncoder

    return json.loads(json.dumps(value, cls=NumpyEncoder))


def _config(run_id):
    with open(os.path.join(get_record_by_id(run_id)["folder_path"], "config_json.json"), encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def evaluated(classification_csv):
    """A tuned, cross-validated run with three models, like one a user would want to finalize."""
    return MLTrainer(task="MLClassification", name="churn-eval", dataset_path=classification_csv, target_column="label",
                     models=["logistic_regression", "random_forest", "knn"], hyper_parameter_tuning=1, cv_folds=3,
                     imputation=True).run()


def test_final_model_trains_on_every_row_with_the_best_settings(evaluated, classification_csv):
    parent = evaluated.performance
    best = easyml.load_model(evaluated.run_id).model_name
    final = easyml.train_final(evaluated.run_id)
    record = get_record_by_id(final.run_id)
    assert record["origin"] == "final" and record["parent_run_id"] == evaluated.run_id
    assert final.run_name == "churn-eval-final" and list(final.performance) == [best]

    config = _config(final.run_id)
    assert config["final_of_run"] == evaluated.run_id and "final_of_run" not in _config(evaluated.run_id)
    entry = final.performance[best]
    assert entry["best_params"] == parent[best]["best_params"]  # no new search
    # No test set: the test and CV scores are the evaluated run's, marked as estimates.
    assert entry["estimated_from_run"] == evaluated.run_id
    assert entry["accuracy_test"] == parent[best]["accuracy_test"]
    assert _plain(entry["test_metrics"]) == _plain(parent[best]["test_metrics"])  # numpy arrays in memory
    assert _plain(entry["cv"]) == _plain(parent[best]["cv"]) and entry["train_metrics"]
    folder = record["folder_path"]
    assert not os.path.exists(os.path.join(folder, "holdout.json"))
    with open(os.path.join(folder, "evaluation.json"), encoding="utf-8") as handle:
        evaluation = json.load(handle)
    assert evaluation["estimated_from_run"] == evaluated.run_id and set(evaluation["models"]) == {best}


def test_all_rows_are_used_and_results_reproduce(evaluated, classification_csv):
    rows = len(pd.read_csv(classification_csv).dropna(subset=["label"]))
    final = easyml.train_final(evaluated.run_id, models=["knn"])  # a model key works as well as a class name
    knn = final.load_model()._ready_controller().inference_body.model
    assert knn.n_samples_fit_ == rows  # the evaluated run's KNN saw 80 % of them
    parent_knn = easyml.load_model(evaluated.run_id, "KNeighborsClassifier")._ready_controller().inference_body.model
    assert parent_knn.n_samples_fit_ < rows

    data = pd.read_csv(classification_csv).drop(columns=["label"])
    again = easyml.train_final(evaluated.run_id, models=["KNeighborsClassifier"], name="knn twice")
    assert again.run_name == "knn twice"
    assert final.load_model().predict(data) == again.load_model().predict(data)


def test_plan_errors(evaluated):
    with pytest.raises(ValueError, match="not a model of run"):
        plan_final(evaluated.run_id, models=["svm"])
    final = easyml.train_final(evaluated.run_id, models=["logistic_regression"])
    with pytest.raises(ValueError, match="already is a final model"):
        plan_final(final.run_id)
    # Retraining a final model trains an evaluated run again, and says so.
    child = easyml.retrain(final.run_id, os.path.join(get_record_by_id(final.run_id)["folder_path"], "data.csv"))
    assert any("is a final model" in w for w in child.warnings)
    assert "final_of_run" not in _config(child.run_id) and os.path.exists(
        os.path.join(get_record_by_id(child.run_id)["folder_path"], "holdout.json"))


def test_final_model_keeps_the_frozen_recipe_and_text_runs(data_dir, text_csv):
    from easyml.prep import Recipe

    raw = pd.read_csv(os.path.join(os.path.dirname(text_csv), "classification.csv"))
    raw["junk"] = "x"
    source = os.path.join(data_dir, "uploads", "final_raw.csv")
    raw.to_csv(source, index=False)
    run = MLTrainer(task="MLClassification", name="cleaned", dataset_path=source, target_column="label",
                    models=["logistic_regression"], recipe=Recipe(name="tidy", steps=[
                        {"op": "drop_columns", "params": {"columns": ["junk"]}},
                        {"op": "fill_missing", "params": {"columns": ["income"], "strategy": "median"}}])).run()
    final = easyml.train_final(run.run_id)
    folder = get_record_by_id(final.run_id)["folder_path"]
    for name in ("prep.json", "raw_data.csv", "data_profile.json"):
        assert os.path.exists(os.path.join(folder, name)), name
    model = final.load_model()
    assert model.cleans_input and model.predict(raw.drop(columns=["label"]).head(5))
    assert _config(final.run_id)["recipe"]["name"] == "tidy"

    text = MLTrainer(task="NLPTextClassification", name="topics", dataset_path=text_csv, target_column="label",
                     text_column="text", models=["logistic_regression"], min_df=1).run()
    text_final = easyml.train_final(text.run_id)
    entry = text_final.performance["LogisticRegression"]
    assert entry["estimated_from_run"] == text.run_id
    assert entry["accuracy_test"] == text.performance["LogisticRegression"]["accuracy_test"]
    assert text_final.load_model().predict(["the striker scored a goal", "a new laptop chip"]) == ["sports", "tech"]


def _wait(job_id, timeout=60):
    job = get_job(job_id)
    deadline = time.time() + timeout
    while not job.finished:
        assert time.time() < deadline, "the final-model job did not finish"
        time.sleep(0.05)
    return job


def test_http_cli_and_report(client, evaluated, capsys):
    from easyml.app import main

    response = client.post(f"/runs/{evaluated.run_id}/final", json={"models": ["RandomForestClassifier"]})
    assert response.status_code == 202, response.data[:300]
    result = _wait(response.get_json()["job_id"]).history[-1]["result"]
    assert result["status_code"] == 200 and result["pipeline_summary"]["final_of_run"] == evaluated.run_id
    assert list(result["performance"]) == ["RandomForestClassifier"]
    assert client.post(f"/runs/{evaluated.run_id}/final", json={"models": "rf"}).status_code == 400
    assert client.post("/runs/999999/final", json={}).status_code == 404

    report = client.get(f"/runs/{result['run_id']}/report").get_data(as_text=True)
    assert f"final model of run #{evaluated.run_id}" in report and "Expected" in report

    assert main(["final", str(evaluated.run_id), "--model", "logistic_regression"]) == 0
    assert f"the final model of run #{evaluated.run_id}" in capsys.readouterr().out
    assert main(["runs", "--limit", "3"]) == 0 and "(est.)" in capsys.readouterr().out
    assert main(["final", "999999"]) == 1


def test_schedule_keeps_a_final_champion_final(data_dir, tmp_path):
    from easyml.scheduling import engine, service

    rng = np.random.default_rng(5)

    def frame(signal, n=400):
        data = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
        data["bought"] = np.where(data[signal] > 0, "yes", "no")
        return data

    base = os.path.join(data_dir, "uploads", "final_sched_base.csv")
    frame("x1").to_csv(base, index=False)
    evaluated = MLTrainer(task="MLClassification", name="sched-eval", dataset_path=base, target_column="bought",
                          models=["logistic_regression"]).run()
    champion = easyml.train_final(evaluated.run_id).run_id
    feed = tmp_path / "feed"
    feed.mkdir()
    frame("x2").to_csv(feed / "new.csv", index=False)  # the behaviour changed: a new model wins
    row = service.create({"run_id": champion, "name": "final feed", "source_path": str(feed), "trigger": "new_data"})
    event = engine.run_schedule(row["id"], trigger="manual")
    assert event["decision"] == "promoted"
    in_use = get_record_by_id(event["champion_after"])
    assert in_use["origin"] == "final" and in_use["parent_run_id"] == event["challenger_run_id"]
    assert "final model, trained on all the new data" in event["message"]
    service.delete(row["id"])
