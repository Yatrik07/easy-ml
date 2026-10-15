"""`easy-ml serve`: the prediction service for tabular, text and image runs."""

import base64
import io
import os
import subprocess
import sys
import threading

import pandas as pd
import pytest

import easyml
from easyml.app import _build_parser
from easyml.predictor import Predictor
from easyml.serving.server import create_serve_app
from easyml.trainer import MLTrainer

ROW = {"age": 45, "income": 52000, "city": "NY", "owns_car": "yes", "rooms": 3}


@pytest.fixture(scope="module")
def clf_run(classification_csv):
    return MLTrainer(task="MLClassification", name="serve-clf", dataset_path=classification_csv, target_column="label",
                     models=["logistic_regression", "random_forest"]).run()


@pytest.fixture(scope="module")
def reg_run(regression_csv):
    return MLTrainer(task="MLRegression", name="serve-reg", dataset_path=regression_csv, target_column="y",
                     models=["linear_regression"]).run()


@pytest.fixture(scope="module")
def text_run(text_csv):
    return MLTrainer(task="NLPTextClassification", name="serve-nlp", dataset_path=text_csv, target_column="label",
                     text_column="text", models=["logistic_regression"], min_df=1).run()


def _client(run_id, **kwargs):
    predictor = Predictor(run_id).warm_up()
    return create_serve_app(predictor, **kwargs).test_client(), predictor


def test_tabular_answers_match_the_library(clf_run, classification_csv):
    client, predictor = _client(clf_run.run_id)
    expected = predictor.predict_with_confidence([ROW])
    single = client.post("/predict", json={"features": ROW}).get_json()
    assert single == {"prediction": expected[0][0], "confidence": expected[1][0]}

    rows = pd.read_csv(classification_csv).drop(columns=["label"]).head(25)
    rows = rows.astype(object).where(rows.notna(), None).to_dict(orient="records")
    batch = client.post("/predict", json={"rows": rows}).get_json()
    preds, confs = predictor.predict_with_confidence(rows)
    assert batch["predictions"] == preds and batch["confidences"] == pytest.approx(confs)


def test_regression_has_no_confidence(reg_run):
    client, predictor = _client(reg_run.run_id)
    answer = client.post("/predict", json={"features": {"x1": 1.0, "x2": "b"}}).get_json()
    assert set(answer) == {"prediction"}
    assert answer["prediction"] == pytest.approx(predictor.predict({"x1": 1.0, "x2": "b"})[0])


def test_text_single_and_many(text_run):
    client, _ = _client(text_run.run_id)
    assert client.post("/predict", json={"text": "the striker scored a goal"}).get_json()["prediction"] == "sports"
    many = client.post("/predict", json={"texts": ["a faster laptop chip", "the coach praised the team"]}).get_json()
    assert many["predictions"] == ["tech", "sports"] and len(many["confidences"]) == 2


def test_info_health_and_docs_page(clf_run):
    client, _ = _client(clf_run.run_id)
    assert client.get("/health").get_json() == {"status": "ok"}
    info = client.get("/info").get_json()
    assert info["run_id"] == clf_run.run_id and info["inputs"]["kind"] == "features"
    assert set(info["example"]["features"]) == {"age", "income", "city", "owns_car", "rooms"}
    assert info["classes"] == ["high", "low", "mid"]
    page = client.get("/").get_data(as_text=True)
    assert "POST /predict" in page and "Invoke-RestMethod" in page and "owns_car" in page


def test_bad_requests_are_explained(clf_run, text_run):
    client, _ = _client(clf_run.run_id)
    assert client.post("/predict", json={"wrong": 1}).status_code == 400
    assert "features" in client.post("/predict", json={"wrong": 1}).get_json()["error"]
    assert client.post("/predict", data="not json").status_code == 400
    broken = client.post("/predict", data="{features: {age: 45}}", content_type="application/json")
    assert broken.status_code == 400 and "not valid JSON" in broken.get_json()["error"]
    assert client.get("/predict").status_code == 405
    assert client.get("/nope").status_code == 404
    text_client, _ = _client(text_run.run_id)
    assert "text" in text_client.post("/predict", json={"features": {}}).get_json()["error"]


def test_api_key_and_size_limit(clf_run):
    client, _ = _client(clf_run.run_id, api_key="s3cret", max_mb=0.001)
    assert client.get("/health").status_code == 200  # always open
    assert client.post("/predict", json={"features": ROW}).status_code == 401
    assert client.post("/predict", json={"features": ROW}, headers={"X-API-Key": "nope"}).status_code == 401
    ok = client.post("/predict", json={"features": ROW}, headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200
    big = client.post("/predict", json={"rows": [ROW] * 200}, headers={"X-API-Key": "s3cret"})
    assert big.status_code == 413


def test_cors_only_for_listed_origins(clf_run):
    client, _ = _client(clf_run.run_id, cors_origins=["http://127.0.0.1:3000"])
    allowed = client.post("/predict", json={"features": ROW}, headers={"Origin": "http://127.0.0.1:3000"})
    assert allowed.headers.get("Access-Control-Allow-Origin") == "http://127.0.0.1:3000"
    other = client.post("/predict", json={"features": ROW}, headers={"Origin": "http://evil.example"})
    assert "Access-Control-Allow-Origin" not in other.headers
    assert client.open("/predict", method="OPTIONS", headers={"Origin": "http://127.0.0.1:3000"}).status_code == 204


def test_concurrent_first_requests_agree(clf_run):
    # A cold Predictor shared by many threads: every thread must get the same, complete answer.
    app = create_serve_app(Predictor(clf_run.run_id))
    results, errors = [], []

    def call():
        try:
            results.append(app.test_client().post("/predict", json={"features": ROW}).get_json())
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=call) for _ in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(results) == 40
    assert all(r == results[0] for r in results) and "prediction" in results[0]


def test_cli_arguments():
    args = _build_parser().parse_args(["serve", "7", "--port", "9001", "--host", "0.0.0.0", "--cors-origin", "*"])
    assert (args.command, args.run_id, args.serve_port, args.serve_host, args.cors_origin) == ("serve", 7, 9001,
                                                                                             "0.0.0.0", ["*"])
    assert _build_parser().parse_args(["serve", "7"]).serve_port == 8000


def test_unknown_run_and_busy_port(capsys, clf_run):
    from easyml.app import main

    assert main(["serve", "999999"]) == 1
    assert "not found" in capsys.readouterr().err
    import socket

    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        port = busy.getsockname()[1]
        assert main(["serve", str(clf_run.run_id), "--port", str(port)]) == 1
    assert "already in use" in capsys.readouterr().err


def test_serving_package_is_light():
    code = ("import sys, easyml, easyml.serving; easyml.serve; "
            "print(','.join(m for m in ('sklearn', 'torch', 'pandas', 'waitress') if m in sys.modules))")
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True,
                            env={**os.environ, "EASYML_NO_CONSOLE_LOGS": "1"})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_images(tmp_path):
    pytest.importorskip("torch")
    from PIL import Image
    from easyml.cv.cv_controllers import CVController

    root = tmp_path / "shapes"
    for label, color in (("red", (220, 20, 20)), ("blue", (20, 20, 220))):
        (root / label).mkdir(parents=True)
        for i in range(8):
            Image.new("RGB", (32, 32), color).save(root / label / f"{i}.png")
    result = CVController(kwargs={"name": "serve-cv", "dataset_path": str(root), "models": ["SimpleCNN"],
                                  "epochs": 2, "batch_size": 4, "image_size": [32, 32]}).execute_pipeline()
    client, predictor = _client(result["run_id"])
    png = (root / "red" / "0.png").read_bytes()
    expected = predictor.predict(str(root / "red" / "0.png"))[0]

    one = client.post("/predict", data={"file": (io.BytesIO(png), "x.png")}, content_type="multipart/form-data")
    assert one.status_code == 200 and one.get_json()["prediction"] == expected
    two = client.post("/predict", data={"file": [(io.BytesIO(png), "a.png"), (io.BytesIO(png), "b.png")]},
                      content_type="multipart/form-data").get_json()
    assert two["predictions"] == [expected, expected]
    b64 = client.post("/predict", json={"image_base64": base64.b64encode(png).decode()}).get_json()
    assert b64["prediction"] == expected
    assert client.post("/predict", json={"image_base64": "%%%"}).status_code == 400
    assert easyml.load_model(result["run_id"]).predict([png])[0] == expected  # bytes work in the library too
