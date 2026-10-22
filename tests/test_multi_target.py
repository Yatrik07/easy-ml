"""0.13: several target columns in one tabular run, one model per target, target weights."""

import importlib.util
import io
import json
import os
import sys
import time
import types
import zipfile

import numpy as np
import pandas as pd
import pytest

import easyml
from easyml.core.multi_target import BEST_FOR_EACH_TARGET, PerTargetModel
from easyml.trainer import MLTrainer
from easyml.utils.cancellation import get_job
from easyml.utils.database_operations import get_record_by_id

REG_MODELS = ["linear_regression", "random_forest", "knn"]
CLF_MODELS = ["logistic_regression", "random_forest", "knn"]


@pytest.fixture(scope="module")
def houses_csv(data_dir):
    rng = np.random.default_rng(7)
    n = 240
    frame = pd.DataFrame({"area": rng.normal(100, 25, n).round(1), "rooms": rng.integers(1, 6, n),
                          "city": rng.choice(["NY", "LA", "SF"], n), "age": rng.integers(0, 60, n)})
    frame.loc[::23, "area"] = np.nan
    area = frame["area"].fillna(100)
    frame["price"] = (area * 3 + frame["rooms"] * 15 - frame["age"] + rng.normal(0, 8, n)).round(2)
    frame["rent"] = (area * 0.2 + (frame["city"] == "NY") * 25 + rng.normal(0, 3, n)).round(2)
    frame["size"] = np.where(area > 110, "big", np.where(area > 90, "mid", "small"))
    frame["garden"] = (frame["rooms"] > 3).astype(int)  # 0/1 label columns: a multi-label task
    frame["pool"] = ((frame["city"] == "LA") & (area > 95)).astype(int)
    path = os.path.join(data_dir, "uploads", "houses.csv")
    frame.to_csv(path, index=False)
    return path


LABELS = ["size", "garden", "pool"]
NUMBERS = ["price", "rent"]


@pytest.fixture(scope="module")
def reg_run(houses_csv):
    return MLTrainer(task="MLRegression", name="houses-reg", dataset_path=houses_csv, target_column=NUMBERS,
                     target_weights={"price": 3}, models=REG_MODELS, ignore_columns=LABELS, cv_folds=3).run()


@pytest.fixture(scope="module")
def clf_run(houses_csv):
    return MLTrainer(task="MLClassification", name="houses-clf", dataset_path=houses_csv,
                     target_columns=LABELS, models=CLF_MODELS, ignore_columns=NUMBERS).run()


def _config(run):
    with open(os.path.join(run.path_to_save, "config_json.json"), encoding="utf-8") as handle:
        return json.load(handle)


def test_each_target_gets_its_own_model_so_nothing_is_traded(reg_run, houses_csv):
    """A target's model and score are exactly those of a run with that target alone."""
    for index, target in enumerate(NUMBERS):
        other = [t for t in NUMBERS if t != target]
        alone = MLTrainer(task="MLRegression", name=f"alone-{target}", dataset_path=houses_csv, target_column=target,
                          models=REG_MODELS, ignore_columns=LABELS + other, cv_folds=3).run()
        for model, entry in reg_run.performance.items():
            assert entry["targets"][target]["accuracy_test"] == alone.performance[model]["accuracy_test"], (model, target)
            assert entry["targets"][target]["cv"]["mean"] == alone.performance[model]["cv"]["mean"]
        multi = easyml.load_model(reg_run.run_id, "RandomForestRegressor").predict(houses_csv)
        single = easyml.load_model(alone.run_id, "RandomForestRegressor").predict(houses_csv)
        assert np.allclose([row[target] for row in multi], single)


def test_the_run_score_is_the_weighted_average(reg_run):
    config = _config(reg_run)
    assert config["target_column"] == "price" and config["target_columns"] == NUMBERS
    assert config["target_weights"] == {"price": 3.0, "rent": 1.0} and "target_classes" not in config
    for entry in reg_run.performance.values():
        scores = entry["targets"]
        expected = (3 * scores["price"]["accuracy_test"] + scores["rent"]["accuracy_test"]) / 4
        assert entry["accuracy_test"] == pytest.approx(expected, abs=1e-5)
        assert set(entry["test_metrics"]["targets"]) == set(NUMBERS) and "r2_score" in entry["test_metrics"]["targets"]["rent"]
        assert entry["cv"]["folds"] == 3 and entry["cv"]["mean"] is not None
        assert set(entry["best_params"]) == set(NUMBERS)
    with open(os.path.join(reg_run.path_to_save, "evaluation.json"), encoding="utf-8") as handle:
        evaluation = json.load(handle)
    assert evaluation["targets"] == NUMBERS
    charts = evaluation["models"]["LinearRegression"]["targets"]
    assert set(charts) == set(NUMBERS) and "scatter" in charts["rent"] and "importance" in charts["rent"]


def test_classification_targets_and_label_columns(clf_run):
    config = _config(clf_run)
    assert config["target_classes"] == {"size": ["big", "mid", "small"], "garden": [0, 1], "pool": [0, 1]}
    assert config["target_labels"] is None and config["target_encodings"] is None
    entry = clf_run.performance["RandomForestClassifier"]
    assert 0 <= entry["test_metrics"]["exact_match"] <= min(s["accuracy_test"] for s in entry["targets"].values())
    assert entry["test_metrics"]["targets"]["size"]["confusion_matrix_order"] == ["big", "mid", "small"]

    model = easyml.load_model(clf_run.run_id)
    assert model.model_name == BEST_FOR_EACH_TARGET and model.targets == LABELS
    rows = [{"area": 130, "rooms": 5, "city": "LA", "age": 3}, {"area": 70, "rooms": 1, "city": "NY", "age": 50}]
    predictions, confidences = model.predict_with_confidence(rows)
    assert predictions[0]["size"] == "big" and predictions[1]["size"] == "small"
    assert predictions[0]["garden"] == 1 and isinstance(predictions[0]["garden"], int)  # labels keep their type
    assert set(confidences[0]) == set(LABELS) and all(0 < c <= 1 for c in confidences[0].values())


def test_best_for_each_target_uses_each_targets_winner(clf_run, houses_csv):
    from easyml.core.multi_target import combined_summary, winners

    chosen = combined_summary(clf_run.path_to_save)
    assert chosen == winners(clf_run.performance, LABELS)
    combined = easyml.load_model(clf_run.run_id).predict(houses_csv)
    for target, model in chosen.items():
        alone = easyml.load_model(clf_run.run_id, model).predict(houses_csv)
        assert [row[target] for row in combined] == [row[target] for row in alone]
        best = max(clf_run.performance[m]["targets"][target]["accuracy_test"] for m in clf_run.performance)
        assert clf_run.performance[model]["targets"][target]["accuracy_test"] == best


def test_weights_change_the_ranking_never_the_predictions(houses_csv, reg_run):
    plain = MLTrainer(task="MLRegression", name="houses-plain", dataset_path=houses_csv, target_column=NUMBERS,
                      models=REG_MODELS, ignore_columns=LABELS, cv_folds=3).run()
    for model, entry in plain.performance.items():
        assert entry["targets"] == reg_run.performance[model]["targets"]
        mean = (entry["targets"]["price"]["accuracy_test"] + entry["targets"]["rent"]["accuracy_test"]) / 2
        assert entry["accuracy_test"] == pytest.approx(mean, abs=1e-5)


def test_studio_endpoints_cli_and_serve(client, clf_run, reg_run, houses_csv, tmp_path, capsys):
    from easyml.app import main
    from easyml.predictor import Predictor
    from easyml.serving.server import create_serve_app

    data = client.get(f"/get_run_data?run_id={clf_run.run_id}").get_json()
    assert data["models"][0] == BEST_FOR_EACH_TARGET and data["best_model"] == BEST_FOR_EACH_TARGET
    assert data["targets"]["columns"] == LABELS and set(data["targets"]["combined"]) == set(LABELS)
    assert data["targets"]["scores"]["KNeighborsClassifier"]["size"] is not None
    assert "price" not in data["columns"] and "size" not in data["columns"]

    row = {"area": "130", "rooms": "5", "city": "LA", "age": ""}
    answer = client.post("/inference", json={"run_id": clf_run.run_id, "data": row,
                                             "model_name": BEST_FOR_EACH_TARGET}).get_json()
    assert set(answer["prediction"]) == set(LABELS) and set(answer["probability"]) == set(LABELS)
    answer = client.post("/inference", json={"run_id": reg_run.run_id, "data": row,
                                             "model_name": "LinearRegression"}).get_json()
    assert set(answer["prediction"]) == set(NUMBERS) and answer["probability"] is None

    frame = pd.read_csv(houses_csv).drop(columns=LABELS + NUMBERS).head(5)
    response = client.post("/inference_batch", data={"run_id": str(clf_run.run_id), "model_name": BEST_FOR_EACH_TARGET,
                                                     "file": (io.BytesIO(frame.to_csv(index=False).encode()), "new.csv")})
    out = pd.read_csv(io.BytesIO(response.data))
    assert [c for c in out.columns if c.startswith(("Prediction", "Confidence"))] == [
        "Prediction: size", "Confidence: size", "Prediction: garden", "Confidence: garden",
        "Prediction: pool", "Confidence: pool"]

    path = tmp_path / "new.csv"
    frame.to_csv(path, index=False)
    assert main(["predict", str(reg_run.run_id), str(path), "-o", str(tmp_path / "out.csv"), "--no-check"]) == 0
    capsys.readouterr()
    written = pd.read_csv(tmp_path / "out.csv")
    assert list(written.columns[-2:]) == ["Prediction: price", "Prediction: rent"]

    serve = create_serve_app(Predictor(clf_run.run_id)).test_client()
    one = serve.post("/predict", json={"features": row}).get_json()
    assert set(one["prediction"]) == set(LABELS) and set(one["confidence"]) == set(LABELS)
    many = serve.post("/predict", json={"rows": [row, row]}).get_json()
    assert len(many["predictions"]) == 2 and many["predictions"][0] == one["prediction"]
    info = serve.get("/info").get_json()
    assert info["targets"] == LABELS and info["classes"]["size"] == ["big", "mid", "small"]
    assert "Each answer predicts every column" in serve.get("/").get_data(as_text=True)


def test_run_details_report_and_sample(client, reg_run):
    detail = client.get(f"/runs/{reg_run.run_id}").get_json()
    assert detail["targets"] == NUMBERS and detail["target_weights"] == {"price": 3.0, "rent": 1.0}
    summary = next(r for r in client.get("/runs").get_json()["data"] if r["id"] == reg_run.run_id)
    assert summary["targets"] == NUMBERS
    html = client.get(f"/runs/{reg_run.run_id}/report").get_data(as_text=True)
    assert "By column to predict" in html and "price (ranking weight 3)" in html and "Test R² (weighted)" in html
    sample = client.get(f"/runs/{reg_run.run_id}/sample").get_json()
    assert set(sample["actual"]) == set(NUMBERS) and "price" not in sample["values"]
    spec = client.get(f"/runs/{reg_run.run_id}/config").get_json()
    shown = {s["label"]: s["value"] for s in spec["settings"]}
    assert shown["Columns to predict"] == "price (ranking weight 3), rent"
    assert spec["retrain"]["target_columns"] == NUMBERS and set(NUMBERS) <= set(spec["retrain"]["required_columns"])


def test_explain_one_target(client, clf_run):
    from easyml.capabilities import get_capabilities

    if not get_capabilities()["features"]["explain"]:
        pytest.skip("needs SHAP")
    row = {"area": 130, "rooms": 5, "city": "LA", "age": 3}
    for model in (BEST_FOR_EACH_TARGET, "RandomForestClassifier"):
        answer = client.post("/explain", json={"run_id": clf_run.run_id, "data": row, "model_name": model,
                                               "target": "pool"}).get_json()
        assert answer["target"] == "pool" and answer["explanation"]["features"]
    default = client.post("/explain", json={"run_id": clf_run.run_id, "data": row,
                                            "model_name": "LogisticRegression"}).get_json()
    assert default["target"] == "size"
    assert client.post("/explain", json={"run_id": clf_run.run_id, "data": row, "model_name": "LogisticRegression",
                                         "target": "price"}).status_code == 400


def _run_exported(zip_bytes, folder, features, monkeypatch):
    """Run the exported main.py in-process with a stand-in for FastAPI (not installed in the test env)."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        archive.extractall(folder)
    fake = types.ModuleType("fastapi")

    class FastAPI:
        def __init__(self, **kwargs):
            self.routes = {}

        def _route(self, path):
            def register(func):
                self.routes[path] = func
                return func
            return register

        post = get = _route

    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            super().__init__(detail)

    fake.FastAPI, fake.HTTPException = FastAPI, HTTPException
    monkeypatch.setitem(sys.modules, "fastapi", fake)
    monkeypatch.chdir(folder)
    spec = importlib.util.spec_from_file_location(f"exported_{time.time_ns()}", os.path.join(folder, "main.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.predict(module.InferenceInput(features=features)), open("requirements.txt").read()


def test_export_as_api(client, clf_run, reg_run, tmp_path, monkeypatch):
    row = {"area": 130, "rooms": 5, "city": "LA", "age": 3}
    expected = easyml.load_model(clf_run.run_id).predict_with_confidence(row)
    response = client.post("/export", json={"run_id": clf_run.run_id, "model_name": BEST_FOR_EACH_TARGET})
    assert response.status_code == 200
    answer, requirements = _run_exported(response.data, tmp_path / "clf", row, monkeypatch)
    assert answer["prediction"] == expected[0][0] and answer["confidence"] == pytest.approx(expected[1][0])
    assert "easy-ml-studio==" in requirements

    expected = easyml.load_model(reg_run.run_id, "KNeighborsRegressor").predict(row)[0]
    response = client.post("/export", json={"run_id": reg_run.run_id, "model_name": "KNeighborsRegressor"})
    answer, _ = _run_exported(response.data, tmp_path / "reg", row, monkeypatch)
    assert answer["prediction"] == pytest.approx(expected) and "confidence" not in answer


def _wait(job_id, timeout=120):
    job = get_job(job_id)
    deadline = time.time() + timeout
    while not job.finished:
        assert time.time() < deadline, "the training job did not finish"
        time.sleep(0.05)
    return job


def test_training_endpoint_and_bad_requests(client, houses_csv):
    base = {"task": "MLRegression", "name": "houses-http", "dataset_path": houses_csv, "models": ["linear_regression"],
            "ignore_columns": LABELS}
    response = client.post("/ml_predict", json={**base, "target_columns": NUMBERS, "target_weights": {"rent": 2}})
    assert response.status_code == 202, response.data[:300]
    result = _wait(response.get_json()["job_id"]).history[-1]["result"]
    assert result["pipeline_summary"]["targets"] == NUMBERS
    assert result["pipeline_summary"]["target_weights"] == {"price": 1.0, "rent": 2.0}

    def refused(**changes):
        response = client.post("/ml_predict", json={**base, "target_columns": NUMBERS, **changes})
        return response.status_code == 400 and response.get_json()["error_message"]

    assert "only be chosen once" in refused(target_columns=["price", "price"])
    assert "between 0.1 and 10" in refused(target_weights={"price": 50})
    assert "only for the columns to predict" in refused(target_weights={"area": 2})
    assert "cannot also be left out" in refused(ignore_columns=["rent"])
    assert "not available with several" in refused(task="MLClassification", target_columns=LABELS,
                                                   ignore_columns=NUMBERS, models=["random_forest"],
                                                   imbalance_handling="smote")
    assert client.post("/ml_predict", json={**base, "target_columns": "price"}).status_code == 400
    with pytest.raises(ValueError, match="several columns to predict"):
        MLTrainer(task="MLRegression", name="x", dataset_path=houses_csv, target_column="price", target_weights={"price": 2})


def test_amounts_are_refused_as_classes(houses_csv):
    # A price is not a category: one class per value used to crash the tuning ("n_splits=4 cannot be greater").
    for target in (["size", "price"], "price"):
        with pytest.raises(ValueError, match="looks like an amount to predict"):
            MLTrainer(task="MLClassification", name="amount", dataset_path=houses_csv, target_column=target,
                      models=["random_forest"], hyper_parameter_tuning=1).run()


def test_kinds_are_checked_when_the_data_is_read(houses_csv):
    with pytest.raises(ValueError, match="does not hold numbers"):
        MLTrainer(task="MLRegression", name="mixed", dataset_path=houses_csv, target_column=["price", "size"],
                  models=["linear_regression"]).run()
    with pytest.raises(ValueError, match="not found"):
        MLTrainer(task="MLRegression", name="missing", dataset_path=houses_csv, target_column=["price", "nope"],
                  models=["linear_regression"]).run()


def test_retrain_final_model_and_schedule_compare(reg_run, houses_csv):
    from easyml.retraining import plan_retrain
    from easyml.scheduling.compare import SAME_DATA, compare

    plan = plan_retrain(reg_run.run_id, houses_csv)
    assert plan.kwargs["target_columns"] == NUMBERS and plan.kwargs["target_weights"] == {"price": 3.0, "rent": 1.0}
    retrained = easyml.retrain(reg_run.run_id, houses_csv)
    assert retrained.performance["LinearRegression"]["targets"] == reg_run.performance["LinearRegression"]["targets"]

    result = compare(reg_run.run_id, retrained.run_id)
    assert result["compared_on"] == SAME_DATA and result["challenger_score"] == pytest.approx(result["champion_score"])

    final = easyml.train_final(reg_run.run_id, models=["RandomForestRegressor"])
    entry = final.performance["RandomForestRegressor"]
    parent = reg_run.performance["RandomForestRegressor"]
    assert entry["estimated_from_run"] == reg_run.run_id and entry["best_params"] == parent["best_params"]
    assert entry["targets"]["price"]["accuracy_test"] == parent["targets"]["price"]["accuracy_test"]
    assert entry["targets"]["price"]["accuracy_train"] is not None
    assert len(final.load_model().predict(houses_csv)) == len(pd.read_csv(houses_csv))
    assert get_record_by_id(final.run_id)["origin"] == "final"


def test_recipe_archive_and_catboost(houses_csv, tmp_path, data_dir):
    from easyml.prep import Recipe

    recipe = Recipe(name="tidy", steps=[{"op": "fill_missing", "params": {"columns": ["area"], "strategy": "median"}}])
    run = MLTrainer(task="MLRegression", name="houses-clean", dataset_path=houses_csv, target_column=NUMBERS,
                    models=["linear_regression"], ignore_columns=LABELS, recipe=recipe).run()
    model = easyml.load_model(run.run_id)
    assert model.cleans_input and not set(NUMBERS) & set(model.input_columns)
    assert set(model.predict({"area": None, "rooms": 3, "city": "NY", "age": 10})[0]) == set(NUMBERS)

    archive = easyml.export_run(run.run_id, tmp_path / "run.zip")
    imported = easyml.import_run(archive, trusted=True)
    row = {"area": 101, "rooms": 3, "city": "NY", "age": 10}
    assert easyml.load_model(imported).predict(row) == model.predict(row)

    try:
        import catboost  # noqa: F401
    except ImportError:
        return
    cat = MLTrainer(task="MLRegression", name="houses-cat", dataset_path=houses_csv, target_column=NUMBERS,
                    models=["cat_boost"], ignore_columns=LABELS).run()
    assert os.path.exists(os.path.join(cat.path_to_save, "CatBoostRegressor.pkl"))
    loaded = easyml.load_model(cat.run_id, "CatBoostRegressor")
    assert set(loaded.predict(row)[0]) == set(NUMBERS)
    assert isinstance(loaded._ready_controller().inference_body.model, PerTargetModel)


def test_single_target_runs_are_unchanged(reg_run, regression_csv):
    run = MLTrainer(task="MLRegression", name="one-target", dataset_path=regression_csv, target_column="y",
                    models=["linear_regression"]).run()
    config = _config(run)
    assert not {"target_columns", "target_weights", "target_classes"} & set(config)
    assert "targets" not in run.performance["LinearRegression"]
    model = easyml.load_model(run.run_id)
    assert model.targets == [] and model.model_name == "LinearRegression"
    assert isinstance(model.predict({"x1": 0.5, "x2": "b"})[0], float)
