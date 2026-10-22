"""
0.14.2: what several columns to predict do, on data with known answers.

* Each column is learned from the inputs alone, independently of the others and of the ranking weights.
* A column the inputs do not explain is predicted close to its average (and the run says so).
* A column that depends on another predicted column loses it as an input (and the run says so).
* Predictions land in the right column on every path.
"""

import io
import json
import os

import numpy as np
import pandas as pd
import pytest

import easyml
from easyml.trainer import MLTrainer
from easyml.utils.holdout import read_holdout

MODELS = ["linear_regression", "random_forest", "knn"]
WEIGHTS = [(1, 1), (1.5, 1), (2, 1), (1, 2), (3.1, 1)]


def _write(data_dir, name, frame):
    path = os.path.join(data_dir, "uploads", f"{name}.csv")
    frame.to_csv(path, index=False)
    return path


def _depends(x, rng, n):
    y1 = x.x1 + 2 * rng.normal(size=n)  # the inputs explain a fifth of it
    return x.assign(y1=y1.round(4), y2=(y1 + 3 * x.x2 + rng.normal(0, 0.3, n)).round(4))


@pytest.fixture(scope="module")
def cases(data_dir):
    rng = np.random.default_rng(11)
    n = 500
    x = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "x3": rng.normal(size=n),
                      "city": rng.choice(["NY", "LA", "SF"], n)})
    y1 = 3 * x.x1 + rng.normal(0, 0.3, n)
    return {
        # Two learnable columns on very different scales.
        "A": _write(data_dir, "A", x.assign(y1=y1.round(4), y2=(1000 * x.x2 + rng.normal(0, 100, n)).round(2))),
        # One learnable column, one pure noise.
        "B": _write(data_dir, "B", x.assign(y1=y1.round(4), y2=rng.normal(50, 10, n).round(3))),
        # Like taxi price and trip duration: y1 is partly hidden from the inputs, and y2 needs y1.
        "C": _write(data_dir, "C", _depends(x, rng, n)),
        # y2 = y1 + 1000: a swapped column would be off by about 1000.
        "D": _write(data_dir, "D", x.assign(y1=y1.round(4), y2=(y1 + 1000).round(4))),
    }


def _run(path, target, weights=None, ignore=(), models=MODELS):
    return MLTrainer(task="MLRegression", name="behaviour", dataset_path=path, target_column=target, models=models,
                     target_weights=weights, ignore_columns=list(ignore)).run()


def _test_rows(run, targets):
    data = pd.read_csv(os.path.join(run.path_to_save, "data.csv"))
    test = data.iloc[read_holdout(run.path_to_save)]
    return test.drop(columns=targets), test[targets]


def test_weights_never_change_what_a_column_learns(cases):
    for name in ("A", "B", "C"):
        first, first_predictions = None, None
        for weights in WEIGHTS:
            run = _run(cases[name], ["y1", "y2"], dict(zip(["y1", "y2"], weights)))
            inputs, _ = _test_rows(run, ["y1", "y2"])
            predictions = {m: easyml.load_model(run.run_id, m).predict(inputs) for m in run.performance}
            if first is None:
                first, first_predictions = run, predictions
                continue
            assert predictions == first_predictions, (name, weights)  # identical answers for every column
            for model, entry in run.performance.items():
                assert entry["targets"] == first.performance[model]["targets"]
                t = entry["targets"]
                expected = (weights[0] * t["y1"]["accuracy_test"] + weights[1] * t["y2"]["accuracy_test"]) / sum(weights)
                assert entry["accuracy_test"] == pytest.approx(expected, abs=1e-5)  # only the ranking score moves


def test_learnable_columns_are_learned_on_any_scale(cases):
    run = _run(cases["A"], ["y1", "y2"], {"y1": 1, "y2": 3})
    for target in ("y1", "y2"):
        best = max(entry["targets"][target]["accuracy_test"] for entry in run.performance.values())
        assert best > 0.9, target
        spread = run.performance["LinearRegression"]["test_metrics"]["targets"][target]["spread"]
        assert 0.85 < spread < 1.15
    assert "target_notes" not in json.load(open(os.path.join(run.path_to_save, "config_json.json")))


def test_a_column_the_inputs_do_not_explain_is_predicted_near_its_average(cases):
    run = _run(cases["B"], ["y1", "y2"], {"y1": 1, "y2": 2})  # the noise column even weighs more
    entry = run.performance["LinearRegression"]
    assert entry["targets"]["y1"]["accuracy_test"] > 0.9
    assert entry["targets"]["y2"]["accuracy_test"] < 0.05
    assert entry["test_metrics"]["targets"]["y2"]["spread"] < 0.4  # about constant
    assert abs(entry["test_metrics"]["targets"]["y2"]["baseline"]) < 0.05
    notes = json.load(open(os.path.join(run.path_to_save, "config_json.json")))["target_notes"]
    assert [n["kind"] for n in notes["y2"]] == ["not_predictable"] or "not_predictable" in [n["kind"] for n in notes["y2"]]
    assert "y1" not in notes or "not_predictable" not in [n["kind"] for n in notes["y1"]]
    assert "cannot be predicted from these inputs" in notes["y2"][0]["text"]


def test_a_column_that_needs_another_target_loses_it(cases):
    together = _run(cases["C"], ["y1", "y2"])
    alone_with_y1 = _run(cases["C"], "y2")  # y1 is an input here
    with_target = together.performance["LinearRegression"]["targets"]["y2"]["accuracy_test"]
    with_input = alone_with_y1.performance["LinearRegression"]["accuracy_test"]
    assert with_input > with_target + 0.1, (with_target, with_input)
    notes = json.load(open(os.path.join(together.path_to_save, "config_json.json")))["target_notes"]
    related = [n for n in notes["y2"] if n["kind"] == "depends_on_target"]
    assert related and related[0]["other"] == "y1" and "train y2 in its own run" in related[0]["text"]
    assert abs(related[0]["corr"]) > 0.5  # y1 explains most of what the models miss for y2


def test_rows_missing_a_target_are_counted(cases, data_dir):
    frame = pd.read_csv(cases["A"])
    frame.loc[:9, "y2"] = np.nan
    run = _run(_write(data_dir, "A_missing", frame), ["y1", "y2"])
    assert run.rows_without_targets == 10 and len(run.df) == len(frame) - 10


def test_predictions_land_in_the_right_column_everywhere(cases, client):
    from easyml.core.evaluation_charts import read_evaluation
    from easyml.core.multi_target import BEST_FOR_EACH_TARGET
    from easyml.predictor import Predictor
    from easyml.serving.server import create_serve_app

    run = _run(cases["D"], ["y1", "y2"])
    inputs, truth = _test_rows(run, ["y1", "y2"])
    for model in [BEST_FOR_EACH_TARGET] + list(run.performance):
        rows = easyml.load_model(run.run_id, model).predict(inputs)
        y1, y2 = np.array([r["y1"] for r in rows]), np.array([r["y2"] for r in rows])
        assert abs(np.mean(y2 - y1) - 1000) < 5, model  # y2 stays y1 + 1000: nothing is swapped
        assert abs(np.mean(y1) - truth["y1"].mean()) < 1

    row = inputs.iloc[0].to_dict()
    expected = easyml.load_model(run.run_id).predict(row)[0]
    answer = client.post("/inference", json={"run_id": run.run_id, "data": row,
                                             "model_name": BEST_FOR_EACH_TARGET}).get_json()["prediction"]
    assert answer == pytest.approx(expected)
    batch = client.post("/inference_batch", data={"run_id": str(run.run_id), "model_name": BEST_FOR_EACH_TARGET,
                                                  "file": (io.BytesIO(inputs.head(3).to_csv(index=False).encode()), "x.csv")})
    out = pd.read_csv(io.BytesIO(batch.data))
    assert (out["Prediction: y2"] - out["Prediction: y1"]).between(995, 1005).all()
    served = create_serve_app(Predictor(run.run_id)).test_client().post("/predict", json={"features": row}).get_json()
    assert served["prediction"] == pytest.approx(expected)

    scatter = read_evaluation(run.path_to_save)["models"]["LinearRegression"]["targets"]["y2"]["scatter"]
    assert min(scatter["actual"]) > 900 and min(scatter["predicted"]) > 900  # y2's chart shows y2
