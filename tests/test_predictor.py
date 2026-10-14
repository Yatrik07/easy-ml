"""easyml.load_model and the `easy-ml runs` / `easy-ml predict` commands."""

import json
import os
import subprocess
import sys

import pandas as pd
import pytest

import easyml
from easyml.app import main
from easyml.trainer import MLTrainer
from easyml.utils.database_operations import RunNotFoundError

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def clf_run(classification_csv):
    return MLTrainer(task="MLClassification", name="pred-clf", dataset_path=classification_csv, target_column="label",
                     models=["logistic_regression", "random_forest"]).run()


@pytest.fixture(scope="module")
def reg_run(regression_csv):
    return MLTrainer(task="MLRegression", name="pred-reg", dataset_path=regression_csv, target_column="y",
                     models=["linear_regression", "knn"]).run()


@pytest.fixture(scope="module")
def text_run(text_csv):
    return MLTrainer(task="NLPTextClassification", name="pred-nlp", dataset_path=text_csv, target_column="label",
                     text_column="text", models=["logistic_regression", "svm"], min_df=1).run()


ROW = {"age": "45", "income": "", "city": "NY", "owns_car": "yes", "rooms": "3"}


def test_default_model_is_the_best_one(clf_run):
    model = easyml.load_model(clf_run.run_id)
    best = max(clf_run.performance, key=lambda k: clf_run.performance[k]["accuracy_test"])
    assert model.model_name == best
    assert set(model.available_models) == set(clf_run.performance)
    assert "Tabular Classification" in repr(model)


def test_tabular_inputs_match_the_http_endpoint(client, clf_run, classification_csv):
    model = easyml.load_model(clf_run.run_id, "RandomForestClassifier")
    api = client.post("/inference", json={"run_id": clf_run.run_id, "model_name": "RandomForestClassifier", "data": ROW})
    assert model.predict(ROW) == [api.get_json()["prediction"]]

    frame = pd.read_csv(classification_csv).drop(columns=["label"]).head(5)
    from_frame = model.predict(frame)
    assert len(from_frame) == 5 and set(from_frame) <= {"high", "mid", "low"}
    assert model.predict(frame.to_dict(orient="records")) == from_frame
    assert model.predict(classification_csv)[:5] == from_frame

    predictions, confidences = model.predict_with_confidence(frame)
    assert len(confidences) == 5 and all(0 <= c <= 1 for c in confidences)


def test_regression_predictions_are_numbers(reg_run):
    predictions, confidences = easyml.load_model(reg_run.run_id).predict_with_confidence([{"x1": 1.0, "x2": "b"}])
    assert isinstance(predictions[0], float) and confidences is None
    assert abs(predictions[0] - 5.0) < 1.0


def test_text_inputs(text_run, text_csv):
    model = easyml.load_model(text_run.run_id, "LogisticRegression")
    assert model.predict("the striker scored a goal in the match") == ["sports"]
    assert model.predict(["a faster chip for gaming laptops", "the coach praised the team"]) == ["tech", "sports"]
    assert len(model.predict(text_csv)) == len(pd.read_csv(text_csv))
    labels, confidences = easyml.load_model(text_run.run_id, "LinearSVC").predict_with_confidence("new laptop")
    assert labels == ["tech"] and confidences is None


def test_errors(clf_run):
    with pytest.raises(RunNotFoundError):
        easyml.load_model(999999)
    with pytest.raises(ValueError):
        easyml.load_model(clf_run.run_id, "XGBClassifier")
    with pytest.raises(ValueError):
        easyml.load_model(clf_run.run_id, "../../evil")
    with pytest.raises(TypeError):
        easyml.load_model(clf_run.run_id).predict(42)


def test_cli_runs(capsys, clf_run, reg_run):
    assert main(["runs", "--json"]) == 0
    runs = json.loads(capsys.readouterr().out)
    ids = [r["id"] for r in runs]
    assert clf_run.run_id in ids and reg_run.run_id in ids
    assert ids == sorted(ids, reverse=True)

    assert main(["runs", "--task", "MLRegression"]) == 0
    table = capsys.readouterr().out
    assert "pred-reg" in table and "pred-clf" not in table and "BEST MODEL" in table


def test_cli_predict_csv(capsys, tmp_path, clf_run, classification_csv):
    output = tmp_path / "predictions.csv"
    assert main(["predict", str(clf_run.run_id), classification_csv, "--output", str(output)]) == 0
    result = pd.read_csv(output)
    assert {"Prediction", "Confidence"} <= set(result.columns)
    assert len(result) == len(pd.read_csv(classification_csv))

    assert main(["predict", str(clf_run.run_id), classification_csv]) == 0
    stdout = capsys.readouterr().out
    assert stdout.splitlines()[0].endswith("Prediction,Confidence")


def test_cli_predict_text(capsys, text_run):
    assert main(["predict", str(text_run.run_id), "the team won the football match", "--model", "LogisticRegression"]) == 0
    assert capsys.readouterr().out.startswith("sports")


def test_cli_predict_errors(capsys, clf_run):
    assert main(["predict", "999999", "x.csv"]) == 1
    assert "not found" in capsys.readouterr().err
    assert main(["predict", str(clf_run.run_id), "missing.csv"]) == 1
    assert main(["predict", str(clf_run.run_id), "x.csv", "--model", "Nope"]) == 1


def test_image_prediction(tmp_path):
    pytest.importorskip("torch")
    from PIL import Image
    from easyml.cv.cv_controllers import CVController

    root = tmp_path / "shapes"
    for label, color in (("red", (220, 20, 20)), ("blue", (20, 20, 220))):
        (root / label).mkdir(parents=True)
        for i in range(8):
            Image.new("RGB", (32, 32), color).save(root / label / f"{i}.png")
    result = CVController(kwargs={"name": "cv-pred", "dataset_path": str(root), "models": ["SimpleCNN"],
                                  "epochs": 2, "batch_size": 4, "image_size": [32, 32]}).execute_pipeline()

    model = easyml.load_model(result["run_id"])
    folder_predictions = model.predict(str(root))
    assert len(folder_predictions) == 16 and set(folder_predictions) <= {"red", "blue"}
    assert len(model.predict(str(root / "red" / "0.png"))) == 1


def test_import_easyml_does_not_load_heavy_libraries():
    code = ("import sys, easyml, easyml.retraining, easyml.run_archive, easyml.core.evaluation_charts; "
            "easyml.load_model, easyml.retrain, easyml.export_run, easyml.import_run; "
            "print(','.join(m for m in ('torch', 'shap', 'spacy', 'sentence_transformers') if m in sys.modules))")
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True,
                            env={**os.environ, "EASYML_NO_CONSOLE_LOGS": "1"})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""
