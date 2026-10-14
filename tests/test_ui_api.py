"""Backend behaviour the redesigned UI relies on: original file names, run summaries, credits."""

import io
import json
import os

import pytest

from easyml.route_helpers import clean_display_filename
from easyml.trainer import MLTrainer
from easyml.utils.run_summary import display_dataset_name, normalize_timestamp, rank_models, summarize_run


def test_clean_display_filename():
    assert clean_display_filename("..\\..\\evil.csv") == "evil.csv"
    assert clean_display_filename("a/b/c.csv") == "c.csv"
    assert clean_display_filename("C:\\Users\\me\\My Data.csv") == "My Data.csv"
    assert clean_display_filename("bad\x00name\n.csv") == "badname.csv"
    assert clean_display_filename("") is None
    assert clean_display_filename(None) is None
    assert len(clean_display_filename("x" * 500 + ".csv")) == 120


def test_load_data_returns_original_filename(client, tmp_path):
    response = client.post("/load_data", data={"file": (io.BytesIO(b"a,b\n1,x\n2,y\n"), "My Data.csv")},
                           content_type="multipart/form-data")
    assert response.status_code == 200
    body = response.get_json()
    assert body["original_filename"] == "My Data.csv"
    assert os.path.basename(body["saved_path"]) != "My Data.csv"

    source = tmp_path / "Housing.csv"
    source.write_text("a,b\n1,2\n")
    assert client.post("/load_data", json={"path": str(source)}).get_json()["original_filename"] == "Housing.csv"

    response = client.post("/nlp/load_data", data={"file": (io.BytesIO(b"text,label\nhi,a\n"), "reviews.csv")},
                           content_type="multipart/form-data")
    assert response.get_json()["original_filename"] == "reviews.csv"


@pytest.fixture(scope="module")
def named_runs(data_dir, classification_csv, regression_csv, text_csv):
    runs = {}
    runs["clf"] = MLTrainer(task="MLClassification", name="ui-clf", dataset_path=classification_csv,
                            target_column="label", models=["logistic_regression", "random_forest"],
                            original_dataset_name="customers.csv").run()
    runs["reg"] = MLTrainer(task="MLRegression", name="ui-reg", dataset_path=regression_csv, target_column="y",
                            models=["linear_regression", "knn"]).run()
    runs["nlp"] = MLTrainer(task="NLPTextClassification", name="ui-nlp", dataset_path=text_csv, target_column="label",
                            text_column="text", models=["logistic_regression"], min_df=1,
                            original_dataset_name="reviews.csv").run()
    return runs


def test_dataset_name_is_the_users_file(client, named_runs):
    summaries = {r["id"]: r for r in client.get("/runs").get_json()["data"]}
    assert summaries[named_runs["clf"].run_id]["dataset_name"] == "customers.csv"
    assert summaries[named_runs["nlp"].run_id]["dataset_name"] == "reviews.csv"
    # Without an explicit name the dataset file's own name is used.
    assert summaries[named_runs["reg"].run_id]["dataset_name"] == "regression.csv"


def test_runs_are_summarised_newest_first_without_paths(client, named_runs):
    response = client.get("/runs")
    assert response.status_code == 200
    data = response.get_json()["data"]
    ids = [r["id"] for r in data]
    assert ids == sorted(ids, reverse=True)
    assert "folder_path" not in response.get_data(as_text=True)
    assert "model_path" not in response.get_data(as_text=True)

    by_id = {r["id"]: r for r in data}
    clf = by_id[named_runs["clf"].run_id]
    assert clf["metric"] == {"key": "accuracy", "name": "Accuracy", "kind": "percent"}
    assert clf["task_label"] == "Tabular Classification"
    assert clf["best_model"] in {"LogisticRegression", "RandomForestClassifier"}
    assert {m["label"] for m in clf["models"]} == {"Logistic Regression", "Random Forest"}
    reg = by_id[named_runs["reg"].run_id]
    assert reg["metric"]["kind"] == "decimal" and reg["metric"]["name"] == "R²"
    assert reg["best_model"] == "LinearRegression"
    assert reg["best_score"] == max(m["score"] for m in reg["models"])


def test_run_detail(client, named_runs):
    run_id = named_runs["clf"].run_id
    response = client.get(f"/runs/{run_id}")
    assert response.status_code == 200
    body = response.get_json()
    assert set(body["metrics"]) == {"LogisticRegression", "RandomForestClassifier"}
    assert "model_path" not in json.dumps(body)
    assert client.get("/runs/999999").status_code == 404


def test_get_run_data_lists_best_model_first(client, named_runs):
    body = client.get(f"/get_run_data?run_id={named_runs['reg'].run_id}").get_json()
    assert body["models"][0] == body["best_model"] == "LinearRegression"
    assert body["run_name"] == "ui-reg"
    assert body["metric"]["kind"] == "decimal"
    assert set(body["model_scores"]) == {"LinearRegression", "KNeighborsRegressor"}


def test_summary_helpers_handle_odd_inputs():
    assert rank_models("MLRegression", {"A": {"accuracy_test": -0.5}, "B": {"accuracy_test": None}})[0]["key"] == "A"
    empty = summarize_run({"id": 1, "task": "MLClassification", "metrics_json": "not json"})
    assert empty["best_model"] is None and empty["models"] == []
    assert display_dataset_name("data_0a1b2c3d.csv") == "Uploaded dataset"
    assert display_dataset_name("data_nlp_0a1b2c3d.csv") == "Uploaded dataset"
    assert display_dataset_name("Housing.csv") == "Housing.csv"
    assert normalize_timestamp("2026-10-13T15-35-01") == "2026-10-13T15:35:01"
    assert normalize_timestamp("2026-10-13 15-35-01") == "2026-10-13T15:35:01"
    cv = summarize_run({"id": 2, "task": "ImageClassification",
                        "metrics_json": json.dumps({"SimpleCNN": {"accuracy_test": 0.5, "accuracy_val": 0.6}})})
    assert cv["best_model"] == "SimpleCNN" and cv["best_model_label"] == "Simple CNN"


def test_credits_hide_repo_links_and_personal_emails(client, monkeypatch):
    body = client.get("/settings/credits").get_json()
    assert body["license"] == "Apache-2.0"
    assert [link["id"] for link in body["links"]] == ["pypi"]
    assert all("email" not in dev for dev in body["developers"])

    from easyml import project_info
    monkeypatch.setattr(project_info, "REPO_PUBLIC", True)
    links = [link["id"] for link in client.get("/settings/credits").get_json()["links"]]
    assert {"pypi", "github", "issues"} <= set(links)


def test_device_endpoint_has_current_cuda_hint(client):
    body = client.get("/settings/device").get_json()
    assert "cu118" not in body["cuda_install_hint"]
    assert body["help_url"].startswith("https://pytorch.org/")


def test_index_is_versioned_and_served_without_placeholders(client):
    html = client.get("/").get_data(as_text=True)
    assert "__EASYML_VERSION__" not in html and "__VUE_BUILD__" not in html


def test_static_javascript_has_a_javascript_mime_type(client):
    response = client.get("/static/vendor/vue.global.prod.js")
    assert response.status_code == 200
    assert response.mimetype in ("text/javascript", "application/javascript")
    assert client.get("/static/vendor/fonts.css").mimetype == "text/css"
