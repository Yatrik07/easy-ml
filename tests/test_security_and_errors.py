import os

from easyml.route_helpers import sanitize_run_name


def test_cross_site_writes_are_rejected(client):
    response = client.post("/exit", headers={"Origin": "https://evil.example"})
    assert response.status_code == 403
    response = client.delete("/settings/clear-data", headers={"Origin": "null"})
    assert response.status_code == 403


def test_same_origin_requests_are_allowed(client):
    response = client.post("/stop_training", headers={"Origin": "http://localhost"}, json={"job_id": "unknown"})
    assert response.status_code == 404  # reached the endpoint instead of being blocked


def test_no_wildcard_cors(client):
    response = client.get("/get_models", headers={"Origin": "https://evil.example"})
    assert "Access-Control-Allow-Origin" not in response.headers


def test_dataset_paths_outside_data_dir_are_rejected(client, tmp_path):
    outside = tmp_path / "secret.csv"
    outside.write_text("a,b\n1,2\n")
    response = client.post("/class_distribution", json={"dataset_path": str(outside), "target_column": "a"})
    assert response.status_code == 403
    response = client.post("/ml_predict", json={"task": "MLClassification", "name": "x", "dataset_path": str(outside),
                                                "target_column": "a", "models": ["random_forest"]})
    assert response.status_code == 403


def test_model_name_cannot_traverse(client):
    response = client.post("/inference", json={"run_id": 1, "model_name": "../../evil", "data": {}})
    assert response.status_code == 400


def test_unknown_run_is_404(client):
    assert client.get("/get_run_data?run_id=999999").status_code == 404
    response = client.post("/inference", json={"run_id": 999999, "model_name": "RandomForestClassifier", "data": {}})
    assert response.status_code == 404
    response = client.post("/nlp/inference", json={"run_id": 999999, "model_name": "LogisticRegression", "data": "x"})
    assert response.status_code == 404


def test_unavailable_models_are_rejected_up_front(client, classification_csv):
    response = client.post("/ml_predict", json={"task": "MLClassification", "name": "x", "dataset_path": classification_csv,
                                                "target_column": "label", "models": ["not_a_model"]})
    assert response.status_code == 400


def test_sanitize_run_name():
    assert sanitize_run_name("../../etc/passwd") == "etc_passwd"
    assert sanitize_run_name("C:\\Windows\\x") == "C_Windows_x"
    assert sanitize_run_name("my run: v2?") == "my run_ v2"
    assert sanitize_run_name("") == "run"
    assert os.sep not in sanitize_run_name("a/b\\c")


def test_capabilities_endpoint(client):
    body = client.get("/settings/capabilities").get_json()
    assert {"cv", "embeddings", "lemmatize", "explain"} <= set(body["features"])
    assert "MLClassification" in body["models"]
