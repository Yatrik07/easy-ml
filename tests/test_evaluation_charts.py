"""Result chart data (evaluation.json): curves, scatter/residuals, feature importance, top terms."""

import os

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from easyml.core import evaluation_charts
from easyml.core.evaluation_charts import (
    build_model_charts,
    classification_curves,
    read_evaluation,
    regression_points,
)
from easyml.trainer import MLTrainer
from easyml.utils.database_operations import get_record_by_id

FEATURES = {"age", "income", "city", "owns_car", "rooms"}


def test_binary_auc_matches_sklearn_and_curves_are_small():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.3 + rng.random(2000) * 0.7, 0, 1)
    roc, pr, reason = classification_curves(y, np.column_stack([1 - p, p]), ["no", "yes"], "probability")
    assert reason is None
    curve = roc["curves"][0]
    assert curve["label"] == "yes" and abs(curve["auc"] - roc_auc_score(y, p)) < 1e-3
    assert len(curve["fpr"]) <= 200 and curve["fpr"] == sorted(curve["fpr"]) and curve["tpr"] == sorted(curve["tpr"])
    assert pr["curves"][0]["ap"] > 0.5 and len(pr["curves"][0]["recall"]) <= 200


def test_many_classes_are_capped():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 12, 600)
    scores = rng.random((600, 12))
    roc, pr, _ = classification_curves(y, scores, [f"c{i}" for i in range(12)], "probability")
    assert len(roc["curves"]) == 10 and len(pr["curves"]) == 10
    assert "micro" in roc and roc["micro_auc"] is not None


def test_single_class_test_set_is_skipped():
    roc, pr, reason = classification_curves([1, 1, 1], np.ones((3, 2)), ["a", "b"], "probability")
    assert roc is None and "single class" in reason


def test_regression_points():
    y = np.arange(1000, dtype=float)
    points = regression_points(y, y + 1)
    assert len(points["scatter"]["actual"]) == 500 and points["scatter"]["sampled"]
    assert sum(points["residual_histogram"]["counts"]) == 1000
    assert points["residual_histogram"]["mean"] == -1


def test_a_failing_chart_never_fails_the_run():
    def boom():
        raise RuntimeError("broken")

    charts = build_model_charts({"roc": boom, "pr": lambda: ({"curves": []}, None)})
    assert charts["pr"] == {"curves": []} and "roc" in charts["skipped"]
    with pytest.raises(InterruptedError):
        build_model_charts({"roc": lambda: (_ for _ in ()).throw(InterruptedError())})


@pytest.fixture(scope="module")
def classification_run(classification_csv):
    return MLTrainer(task="MLClassification", name="charts-clf", dataset_path=classification_csv, target_column="label",
                     models=["logistic_regression", "random_forest", "knn"]).run()


def test_classification_run_writes_charts(classification_run):
    evaluation = read_evaluation(classification_run.path_to_save)
    assert evaluation["format_version"] == 1 and evaluation["classes"] == ["high", "low", "mid"]
    for name, charts in evaluation["models"].items():
        assert {"roc", "pr", "importance"} <= set(charts), (name, charts.get("skipped"))
        names = {f["name"] for f in charts["importance"]["features"]}
        assert names <= FEATURES and names
    assert evaluation["models"]["RandomForestClassifier"]["importance"]["method"] == "model"
    assert evaluation["models"]["KNeighborsClassifier"]["importance"]["method"] == "permutation"


def test_pca_runs_use_permutation_importance(classification_csv):
    run = MLTrainer(task="MLClassification", name="charts-pca", dataset_path=classification_csv, target_column="label",
                    models=["random_forest"], reduce_features=True).run()
    importance = read_evaluation(run.path_to_save)["models"]["RandomForestClassifier"]["importance"]
    assert importance["method"] == "permutation" and {f["name"] for f in importance["features"]} <= FEATURES


def test_regression_run_writes_charts(regression_csv):
    run = MLTrainer(task="MLRegression", name="charts-reg", dataset_path=regression_csv, target_column="y",
                    models=["linear_regression"]).run()
    charts = read_evaluation(run.path_to_save)["models"]["LinearRegression"]
    assert {"scatter", "residual_histogram", "importance"} <= set(charts)
    assert {f["name"] for f in charts["importance"]["features"]} <= {"x1", "x2"}


def test_text_run_charts(text_csv):
    run = MLTrainer(task="NLPTextClassification", name="charts-nlp", dataset_path=text_csv, target_column="label",
                    text_column="text", models=["logistic_regression", "svm"], min_df=1).run()
    models = read_evaluation(run.path_to_save)["models"]
    assert models["LinearSVC"]["roc"]["score_source"] == "decision"
    assert models["LogisticRegression"]["roc"]["score_source"] == "probability"
    labels = {group["label"] for group in models["LogisticRegression"]["top_terms"]["by_class"]}
    assert labels == {"sports", "tech"}


def test_chart_errors_are_reported_not_raised(classification_csv, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("no curves today")

    monkeypatch.setattr(evaluation_charts, "classification_curves", broken)
    run = MLTrainer(task="MLClassification", name="charts-broken", dataset_path=classification_csv,
                    target_column="label", models=["logistic_regression"]).run()
    charts = read_evaluation(run.path_to_save)["models"]["LogisticRegression"]
    assert "roc" in charts["skipped"] and "importance" in charts


def test_charts_endpoint(client, classification_run):
    body = client.get(f"/runs/{classification_run.run_id}/charts").get_json()
    assert body["available"] and "RandomForestClassifier" in body["models"]

    os.rename(os.path.join(classification_run.path_to_save, "evaluation.json"),
              os.path.join(classification_run.path_to_save, "evaluation.json.bak"))
    try:
        old = client.get(f"/runs/{classification_run.run_id}/charts")
        assert old.status_code == 200 and not old.get_json()["available"] and "0.6.0" in old.get_json()["reason"]
    finally:
        os.rename(os.path.join(classification_run.path_to_save, "evaluation.json.bak"),
                  os.path.join(classification_run.path_to_save, "evaluation.json"))
    assert client.get("/runs/999999/charts").status_code == 404


def test_image_run_charts(tmp_path):
    pytest.importorskip("torch")
    from PIL import Image
    from easyml.cv.cv_controllers import CVController

    root = tmp_path / "shapes"
    for label, color in (("red", (220, 20, 20)), ("blue", (20, 20, 220))):
        (root / label).mkdir(parents=True)
        for i in range(10):
            Image.new("RGB", (32, 32), color).save(root / label / f"{i}.png")
    result = CVController(kwargs={"name": "cv-charts", "dataset_path": str(root), "models": ["SimpleCNN"],
                                  "epochs": 1, "batch_size": 4, "image_size": [32, 32]}).execute_pipeline()
    evaluation = read_evaluation(get_record_by_id(result["run_id"])["folder_path"])
    charts = evaluation["models"]["SimpleCNN"]
    assert ("roc" in charts) or ("roc" in charts.get("skipped", {}))
    assert evaluation["classes"] == ["blue", "red"]
