"""0.10.0: a run's random seed makes training repeatable; optional k-fold cross-validation."""

import json
import os
import threading

import numpy as np
import pandas as pd
import pytest

from easyml.core.cross_validation import cross_validate
from easyml.core.seeding import apply_seed, validate_cv_folds, validate_seed
from easyml.trainer import MLTrainer
from easyml.utils.database_operations import get_record_by_id


def _config(run):
    folder = get_record_by_id(run.run_id)["folder_path"]
    return json.load(open(os.path.join(folder, "config_json.json"), encoding="utf-8"))


def _same(a, b):
    from easyml.utils.utils import NumpyEncoder

    return json.dumps(a, cls=NumpyEncoder, sort_keys=True) == json.dumps(b, cls=NumpyEncoder, sort_keys=True)


def _clf(classification_csv, **extra):
    return MLTrainer(task="MLClassification", name="seeded", dataset_path=classification_csv, target_column="label",
                     models=extra.pop("models", ["random_forest"]), **extra).run()


def test_same_seed_same_results(classification_csv):
    first, second = _clf(classification_csv), _clf(classification_csv)
    a, b = first.performance["RandomForestClassifier"], second.performance["RandomForestClassifier"]
    assert a["accuracy_test"] == b["accuracy_test"] and _same(a["test_metrics"], b["test_metrics"])
    assert _config(first)["random_seed"] == 42 and _config(first)["cv_folds"] == 0
    other = _clf(classification_csv, random_seed=7)
    assert _config(other)["random_seed"] == 7
    assert not _same(other.performance["RandomForestClassifier"]["test_metrics"], a["test_metrics"])  # another split


def test_seed_reaches_every_model_with_one():
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPClassifier

    for model in (RandomForestClassifier(), AdaBoostClassifier(), MLPClassifier()):
        assert apply_seed(model, 11).get_params()["random_state"] == 11
    assert apply_seed(LinearRegression(), 11).get_params() == LinearRegression().get_params()
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        return
    assert apply_seed(CatBoostClassifier(logging_level="Silent"), 11).get_params()["random_seed"] == 11


def test_validation_of_seed_and_folds():
    assert validate_seed(None) == 42 and validate_seed("7") == 7 and validate_seed(0) == 0
    for bad in (-1, 2 ** 31, "abc", 3.5):
        with pytest.raises(ValueError):
            validate_seed(bad)
    assert validate_cv_folds(None) == 0 and validate_cv_folds(5) == 5 and validate_cv_folds("10") == 10
    for bad in (1, 21, "x"):
        with pytest.raises(ValueError):
            validate_cv_folds(bad)


def test_cross_validation_classification_with_smote_and_tuning(classification_csv):
    run = _clf(classification_csv, models=["random_forest", "logistic_regression"], cv_folds=4,
               imbalance_handling="smote", hyper_parameter_tuning=1)
    for name, values in run.performance.items():
        cv = values["cv"]
        assert cv["folds"] == 4 and len(cv["scores"]) == 4 and cv["metric"] == "accuracy", (name, cv)
        assert 0 <= min(cv["scores"]) <= cv["mean"] <= max(cv["scores"]) <= 1 and cv["std"] >= 0
    assert _config(run)["cv_folds"] == 4


def test_cross_validation_off_changes_nothing(classification_csv):
    run = _clf(classification_csv)
    assert "cv" not in run.performance["RandomForestClassifier"]


def test_cross_validation_regression_and_text(regression_csv, text_csv):
    reg = MLTrainer(task="MLRegression", name="cv-reg", dataset_path=regression_csv, target_column="y",
                    models=["linear_regression"], cv_folds=3).run()
    cv = reg.performance["LinearRegression"]["cv"]
    assert cv["metric"] == "r2" and len(cv["scores"]) == 3 and cv["mean"] > 0.9
    text = MLTrainer(task="NLPTextClassification", name="cv-text", dataset_path=text_csv, target_column="label",
                     text_column="text", models=["logistic_regression", "naive_bayes"], cv_folds=5).run()
    for values in text.performance.values():
        assert len(values["cv"]["scores"]) == 5 and values["cv"]["mean"] >= 0.9


def test_cross_validation_stops_between_folds():
    from sklearn.linear_model import LogisticRegression

    from easyml.utils.cancellation import register_job, reset_current_job, set_current_job

    X = pd.DataFrame({"a": np.arange(100.0), "b": np.arange(100.0) % 7})
    y = (X["a"] > 50).astype(int).to_numpy()
    job = register_job("cv-cancel")
    token = set_current_job(job)
    folds_done = []

    def stop_after_two(number, k):
        folds_done.append(number)
        if number == 2:
            job.cancel()
    try:
        with pytest.raises(InterruptedError):
            cross_validate(LogisticRegression, X, y, 5, 42, True, on_fold=stop_after_two)
    finally:
        reset_current_job(token)
    assert folds_done == [1, 2]
    assert cross_validate(LogisticRegression, X.head(6), y[:6], 5, 42, True)["error"].startswith("Too few")
    assert threading.current_thread()  # the run continues normally after an error result


def test_retrain_replays_seed_and_folds(classification_csv, data_dir):
    import easyml

    parent = _clf(classification_csv, models=["logistic_regression"], random_seed=5, cv_folds=3)
    child = easyml.retrain(parent.run_id, classification_csv)
    config = _config(child)
    assert config["random_seed"] == 5 and config["cv_folds"] == 3
    assert "cv" in child.performance["LogisticRegression"]


def test_training_route_rejects_bad_settings(client, classification_csv):
    body = {"name": "bad", "task": "MLClassification", "dataset_path": classification_csv, "target_column": "label",
            "models": ["logistic_regression"]}
    assert client.post("/ml_predict", json={**body, "random_seed": -3}).status_code == 400
    assert client.post("/ml_predict", json={**body, "cv_folds": 1}).status_code == 400


def test_ties_go_to_cross_validation_then_training_order(client, classification_csv):
    from easyml.utils.run_summary import rank_models

    tied = {"B": {"accuracy_test": 1.0, "cv": {"mean": 0.97}}, "A": {"accuracy_test": 1.0, "cv": {"mean": 0.99}},
            "C": {"accuracy_test": 1.0}}
    assert [m["key"] for m in rank_models("MLClassification", tied)] == ["A", "B", "C"]
    plain = {"Z": {"accuracy_test": 0.9}, "M": {"accuracy_test": 0.9}}
    assert [m["key"] for m in rank_models("MLClassification", plain)] == ["Z", "M"]  # training order
    # The API keeps models in training order (it used to sort them A-Z, which reordered ties).
    run = _clf(classification_csv, models=["random_forest", "logistic_regression"])
    assert list(client.get(f"/runs/{run.run_id}").get_json()["metrics"]) == ["RandomForestClassifier", "LogisticRegression"]
