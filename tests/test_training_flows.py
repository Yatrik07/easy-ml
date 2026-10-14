"""End-to-end flows: train through MLTrainer, then use the HTTP API for inference, explain and export."""

import io
import os
import zipfile

import pytest

from easyml.config.GLOBAL_VARS import ui2back_mappings_classification_models
from easyml.route_helpers import sanitize_run_name
from easyml.trainer import MLTrainer


def _train(data_dir, **kwargs):
    name = kwargs.pop("name")
    path = os.path.join(data_dir, "runs", sanitize_run_name(name))
    os.makedirs(path, exist_ok=True)
    trainer = MLTrainer(name=name, path_to_save=path, timestamp="2026-01-01 00-00-00", **kwargs)
    return trainer.run()


CLASSIFIERS = [m for m in ["logistic_regression", "random_forest", "cat_boost", "xg_boost", "light_gbm", "knn", "ada_boost"]
               if m in ui2back_mappings_classification_models]


@pytest.fixture(scope="module")
def classification_run(data_dir, classification_csv):
    return _train(data_dir, name="clf", task="MLClassification", dataset_path=classification_csv,
                  target_column="label", models=CLASSIFIERS)


def test_classification_trains_every_model(classification_run):
    assert classification_run.run_id is not None and classification_run.run_id > 0
    assert len(classification_run.performance) == len(CLASSIFIERS)
    for metrics in classification_run.performance.values():
        assert 0 <= metrics["accuracy_test"] <= 1
        assert os.path.exists(metrics["model_path"])


@pytest.mark.parametrize("model_name", ["LogisticRegression", "RandomForestClassifier", "CatBoostClassifier",
                                        "XGBClassifier", "LGBMClassifier", "KNeighborsClassifier"])
def test_inference_accepts_form_strings_and_blanks(client, classification_run, model_name):
    if model_name not in classification_run.performance:
        pytest.skip(f"{model_name} not installed")
    payload = {
        "run_id": classification_run.run_id,
        "model_name": model_name,
        # Everything arrives as strings from the HTML form; income is left blank.
        "data": {"age": "45", "income": "", "city": "NY", "owns_car": "yes", "rooms": "3"},
    }
    response = client.post("/inference", json=payload)
    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["prediction"] in {"high", "mid", "low"}
    if body["probability"] is not None:
        assert 0 <= body["probability"] <= 1


def test_batch_inference_returns_csv(client, classification_run, classification_csv):
    with open(classification_csv, "rb") as handle:
        response = client.post("/inference_batch", data={
            "run_id": str(classification_run.run_id),
            "model_name": "CatBoostClassifier" if "CatBoostClassifier" in classification_run.performance else "RandomForestClassifier",
            "file": (handle, "batch.csv"),
        }, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"Prediction" in response.data.splitlines()[0]


def test_get_run_data_lists_models_and_column_details(client, classification_run):
    body = client.get(f"/get_run_data?run_id={classification_run.run_id}").get_json()
    assert set(body["models"]) == set(classification_run.performance)
    assert body["column_details"]["city"]["type"] == "categorical"
    assert "NY" in body["column_details"]["city"]["values"]
    assert body["column_details"]["income"]["type"] == "numeric"


@pytest.mark.parametrize("model_name", ["RandomForestClassifier", "LogisticRegression"])
def test_explain_returns_plot(client, classification_run, model_name):
    pytest.importorskip("shap")
    response = client.post("/explain", json={
        "run_id": classification_run.run_id, "model_name": model_name,
        "data": {"age": "45", "income": "75000", "city": "NY", "owns_car": "yes", "rooms": "3"},
    })
    assert response.status_code == 200, response.get_json()
    assert response.get_json()["plot_b64"]


@pytest.mark.parametrize("model_name", ["RandomForestClassifier", "CatBoostClassifier", "LGBMClassifier"])
def test_export_builds_complete_package(client, classification_run, model_name):
    if model_name not in classification_run.performance:
        pytest.skip(f"{model_name} not installed")
    response = client.post("/export", json={"run_id": classification_run.run_id, "model_name": model_name})
    assert response.status_code == 200, response.data[:300]
    names = set(zipfile.ZipFile(io.BytesIO(response.data)).namelist())
    assert {"main.py", "pipeline.pkl", "requirements.txt", "Dockerfile", "column_data.json"} <= names
    requirements = zipfile.ZipFile(io.BytesIO(response.data)).read("requirements.txt").decode()
    if model_name == "LGBMClassifier":
        assert "lightgbm==" in requirements
    if model_name == "CatBoostClassifier":
        assert "model.cbm" in names


def test_regression_flow(client, data_dir, regression_csv):
    run = _train(data_dir, name="reg", task="MLRegression", dataset_path=regression_csv, target_column="y",
                 models=["linear_regression", "random_forest"], hyper_parameter_tuning=1)
    assert run.performance["LinearRegression"]["accuracy_test"] > 0.9
    assert run.performance["RandomForestRegressor"]["best_params"]
    response = client.post("/inference", json={"run_id": run.run_id, "model_name": "LinearRegression",
                                               "data": {"x1": "1.0", "x2": "b"}})
    assert response.status_code == 200
    assert abs(response.get_json()["prediction"] - 5.0) < 1.0


def test_nlp_vectorizer_flow(client, data_dir, text_csv):
    run = _train(data_dir, name="nlp", task="NLPTextClassification", dataset_path=text_csv,
                 target_column="label", text_column="text", models=["logistic_regression", "complement_nb"],
                 lemmatize=True, min_df=1)
    assert set(run.performance) == {"LogisticRegression", "ComplementNB"}
    response = client.post("/nlp/inference", json={"run_id": run.run_id, "model_name": "LogisticRegression",
                                                   "data": "the striker scored a goal in the match"})
    assert response.status_code == 200, response.get_json()
    assert response.get_json()["prediction"] == "sports"


def test_cv_flow(client, data_dir):
    pytest.importorskip("torch")
    from PIL import Image

    root = os.path.join(data_dir, "images")
    for label, color in (("red", (220, 20, 20)), ("blue", (20, 20, 220))):
        os.makedirs(os.path.join(root, label), exist_ok=True)
        for i in range(6):
            Image.new("RGB", (32, 32), color).save(os.path.join(root, label, f"{i}.png"))
    with open(os.path.join(root, "red", "broken.jpg"), "wb") as handle:
        handle.write(b"not an image")

    from easyml.cv.cv_controllers import CVController

    controller = CVController(kwargs={"name": "cv", "dataset_path": root, "models": ["SimpleCNN"],
                                      "epochs": 2, "batch_size": 4, "image_size": [64, 64]})
    result = controller.execute_pipeline()
    assert result["run_id"] > 0

    # Held-out test metrics are stored per class, like tabular runs.
    perf = result["performance"]["SimpleCNN"]
    test_metrics = perf["test_metrics"]
    assert test_metrics["confusion_matrix_order"] == ["blue", "red"]
    assert {"blue", "red", "weighted avg"} <= set(test_metrics["classification_report"])
    assert 0 <= perf["accuracy_test"] <= 1 and 0 <= perf["accuracy_val"] <= 1 and perf["f1_test"] is not None
    sizes = [int(n) for n in result["pipeline_summary"]["split"].split()[0].split("/")]
    assert sum(sizes) == 12 and all(sizes)
    assert sum(map(sum, test_metrics["confusion_matrix"])) == sizes[2]
    assert sum(map(sum, perf["train_metrics"]["confusion_matrix"])) == sizes[0]
    stored_metrics = client.get(f"/runs/{result['run_id']}").get_json()["metrics"]
    assert stored_metrics["SimpleCNN"]["test_metrics"]["classification_report"]["red"]["support"] >= 1

    import base64
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (220, 20, 20)).save(buffer, format="PNG")
    response = client.post("/cv/inference", json={"run_id": result["run_id"], "model_name": "SimpleCNN",
                                                  "data": base64.b64encode(buffer.getvalue()).decode()})
    assert response.status_code == 200, response.get_json()
    assert response.get_json()["prediction"] in {"red", "blue"}
