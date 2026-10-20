"""0.14: input column weights - how much the models rely on each input column."""

import importlib.util
import io
import json
import os
import sys
import types
import zipfile

import numpy as np
import pandas as pd
import pytest

import easyml
from easyml.core.column_weights import follows_weights, weight_effect
from easyml.trainer import MLTrainer


def _installed(module):
    return importlib.util.find_spec(module) is not None


BOOSTERS = [key for key, module in (("light_gbm", "lightgbm"), ("cat_boost", "catboost"), ("xg_boost", "xgboost"))
            if _installed(module)]


@pytest.fixture(scope="module")
def twins_csv(data_dir):
    """Two equally useful numeric columns (x1, x2), a category with a rare value, and noise."""
    rng = np.random.default_rng(3)
    n = 400
    frame = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "noise": rng.normal(size=n),
                          "city": rng.choice(["NY", "LA", "SF"], n, p=[0.48, 0.48, 0.04])})
    frame["y"] = frame["x1"] + frame["x2"] + (frame["city"] == "NY") * 0.5 + rng.normal(0, 0.3, n)
    frame["y2"] = frame["x1"] - frame["x2"] + rng.normal(0, 0.3, n)
    frame["label"] = np.where(frame["x1"] + frame["x2"] + rng.normal(0, 0.8, n) > 0, "yes", "no")
    path = os.path.join(data_dir, "uploads", "twins.csv")
    frame.to_csv(path, index=False)
    return path


def _run(path, task="MLRegression", target="y", models=("knn",), **kwargs):
    others = [c for c in ("y", "y2", "label") if c != target and c not in (target if isinstance(target, list) else [])]
    return MLTrainer(task=task, name="twins", dataset_path=path, target_column=target, models=list(models),
                     ignore_columns=others, scaling=True, **kwargs).run()


def _importance(run, model):
    with open(os.path.join(run.path_to_save, "evaluation.json"), encoding="utf-8") as handle:
        chart = json.load(handle)["models"][model]["importance"]
    values = {f["name"]: f["value"] for f in chart["features"]}
    return values["x1"] / (values["x1"] + values["x2"])  # x1's share of the two twins


def _model(run, name):
    return easyml.load_model(run.run_id, name)._ready_controller().inference_body.model


def _weights(params):
    """The weights a booster was given: a list or array, CatBoost's "index:weight" text, or its saved dict."""
    vector = params.get("feature_contri") or params.get("feature_weights")
    if isinstance(vector, str):
        return [float(part.split(":")[1]) for part in vector.split(",")]
    return [float(v) for v in (vector.values() if isinstance(vector, dict) else vector)]


@pytest.mark.parametrize("models, task, target, name", [
    (["knn"], "MLRegression", "y", "KNeighborsRegressor"),
] + [([key], "MLRegression", "y", name) for key, name in (("light_gbm", "LGBMRegressor"),) if key in BOOSTERS])
def test_up_weighting_a_column_makes_the_model_rely_on_it_more(twins_csv, models, task, target, name):
    plain = _run(twins_csv, task, target, models)
    weighted = _run(twins_csv, task, target, models, feature_weights={"x1": 4})
    assert _importance(weighted, name) > _importance(plain, name) + 0.05, name
    assert follows_weights(name)


def test_logistic_regression_follows_only_slightly(twins_csv):
    # Scaling a column only changes how much regularisation holds it back: a small effect, said so in the UI.
    plain = _run(twins_csv, "MLClassification", "label", ["logistic_regression"])
    weighted = _run(twins_csv, "MLClassification", "label", ["logistic_regression"], feature_weights={"x1": 4})
    assert 0 <= _importance(weighted, "LogisticRegression") - _importance(plain, "LogisticRegression") < 0.05
    assert weight_effect("LogisticRegression") == "slight" and weight_effect("RandomForestClassifier") == "none"
    # Measured by permutation, CatBoost and XGBoost barely follow weights either (see column_weights).
    assert weight_effect("XGBRegressor") == weight_effect("CatBoostClassifier") == "slight"
    assert weight_effect("LGBMRegressor") == weight_effect("KNeighborsClassifier") == "follows"


def test_models_without_a_setting_are_not_steered(twins_csv):
    plain = _run(twins_csv, models=["random_forest", "linear_regression"])
    weighted = _run(twins_csv, models=["random_forest", "linear_regression"], feature_weights={"x1": 4, "city": 0.5})
    rows = pd.read_csv(twins_csv).head(50)
    assert np.allclose(easyml.load_model(plain.run_id, "LinearRegression").predict(rows),
                       easyml.load_model(weighted.run_id, "LinearRegression").predict(rows))
    # Trees only shift through float rounding at split points (scaling a column by 3 is not exact).
    forest = (plain.performance["RandomForestRegressor"]["accuracy_test"],
              weighted.performance["RandomForestRegressor"]["accuracy_test"])
    assert abs(forest[0] - forest[1]) < 0.01 and abs(_importance(plain, "RandomForestRegressor")
                                                     - _importance(weighted, "RandomForestRegressor")) < 0.02
    assert not follows_weights("RandomForestRegressor") and not follows_weights("LinearRegression")


@pytest.mark.skipif(not BOOSTERS, reason="needs a gradient-boosting library")
def test_settings_reach_the_boosters_and_cross_validation_works(twins_csv):
    # Tuning and cross-validation both clone the models with their weights.
    run = _run(twins_csv, models=BOOSTERS, feature_weights={"x1": 2, "city": 3}, cv_folds=3, hyper_parameter_tuning=1)
    with open(os.path.join(run.path_to_save, "config_json.json"), encoding="utf-8") as handle:
        assert json.load(handle)["feature_weights"] == {"x1": 2.0, "city": 3.0}
    for name, entry in run.performance.items():
        assert entry["cv"]["mean"] is not None, (name, entry["cv"])  # each fold matched its own columns
        params = _model(run, name).get_params()
        values = _weights(params)
        assert 2.0 in values and 3.0 in values, name
        if name.startswith("XGB"):
            assert params["colsample_bynode"] == 0.8


def test_weights_with_several_targets_and_every_prediction_path(twins_csv, client, tmp_path, monkeypatch):
    models = ["knn"] + BOOSTERS[:1]
    run = _run(twins_csv, target=["y", "y2"], models=models, feature_weights={"x2": 3})
    row = {"x1": 0.5, "x2": -1.0, "noise": 0.1, "city": "LA"}
    model = easyml.load_model(run.run_id, "KNeighborsRegressor")
    expected = model.predict(row)[0]
    answer = client.post("/inference", json={"run_id": run.run_id, "data": row,
                                             "model_name": "KNeighborsRegressor"}).get_json()
    assert answer["prediction"] == pytest.approx(expected)
    for name in [m for m in run.performance if m != "KNeighborsRegressor"]:
        for estimator in _model(run, name).estimators_:
            assert 3.0 in _weights(estimator.get_params())

    # The exported app needs no Easy-ML code for the weights (they are plain scikit-learn steps).
    response = client.post("/export", json={"run_id": run.run_id, "model_name": "KNeighborsRegressor"})
    folder = tmp_path / "app"
    zipfile.ZipFile(io.BytesIO(response.data)).extractall(folder)
    fake = types.ModuleType("fastapi")

    class FastAPI:
        def __init__(self, **kwargs):
            pass

        def _route(self, path):
            return lambda func: func

        post = get = _route

    fake.FastAPI, fake.HTTPException = FastAPI, Exception
    monkeypatch.setitem(sys.modules, "fastapi", fake)
    monkeypatch.chdir(folder)
    spec = importlib.util.spec_from_file_location("weighted_app", folder / "main.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    exported = app.predict(app.InferenceInput(features=row))["prediction"]
    assert exported["y"] == pytest.approx(expected["y"]) and exported["y2"] == pytest.approx(expected["y2"])


def test_explain_retrain_and_settings(twins_csv, client):
    from easyml.capabilities import get_capabilities
    from easyml.retraining import plan_retrain

    run = _run(twins_csv, models=["knn"], feature_weights={"x1": 2})
    plan = plan_retrain(run.run_id, twins_csv)
    assert plan.kwargs["feature_weights"] == {"x1": 2.0}
    detail = client.get(f"/runs/{run.run_id}").get_json()
    assert detail["feature_weights"] == {"x1": 2.0}
    settings = {s["label"]: s["value"] for s in client.get(f"/runs/{run.run_id}/config").get_json()["settings"]}
    assert settings["Column weights"] == "x1 ×2"
    if get_capabilities()["features"]["explain"]:
        answer = client.post("/explain", json={"run_id": run.run_id, "model_name": "KNeighborsRegressor",
                                               "data": {"x1": 1, "x2": 0, "noise": 0, "city": "NY"}}).get_json()
        names = {f["name"] for f in answer["explanation"]["features"]}
        assert {"x1", "x2", "city"} <= names  # contributions still add up per original column


def test_bad_weights_are_refused(twins_csv, client):
    with pytest.raises(ValueError, match="between 0.1 and 10"):
        _run(twins_csv, feature_weights={"x1": 0})
    with pytest.raises(ValueError, match="not an input column"):
        _run(twins_csv, feature_weights={"y": 2})
    with pytest.raises(ValueError, match="not an input column"):
        MLTrainer(task="MLRegression", name="x", dataset_path=twins_csv, target_column="y", models=["knn"],
                  ignore_columns=["noise"], feature_weights={"noise": 2})
    with pytest.raises(ValueError, match="PCA"):
        _run(twins_csv, feature_weights={"x1": 2}, reduce_features=True)
    with pytest.raises(ValueError, match="not inputs"):
        _run(twins_csv, feature_weights={"nope": 2})
    response = client.post("/ml_predict", json={"task": "MLRegression", "name": "w", "dataset_path": twins_csv,
                                                "target_column": "y", "models": ["knn"], "feature_weights": {"x1": 99}})
    assert response.status_code == 400 and "between 0.1 and 10" in response.get_json()["error_message"]


def test_runs_without_weights_are_unchanged(twins_csv):
    run = _run(twins_csv, models=["knn"], feature_weights={"x1": 1})  # weight 1 = no weight
    with open(os.path.join(run.path_to_save, "config_json.json"), encoding="utf-8") as handle:
        assert "feature_weights" not in json.load(handle)
    steps = easyml.load_model(run.run_id)._ready_controller().inference_body.pipeline
    assert [name for name, _, _ in steps.named_steps["column_transformer"].transformers] == [
        "nominal_pipeline", "numerical_pipeline"]
