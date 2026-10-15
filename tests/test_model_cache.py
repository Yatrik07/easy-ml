"""The in-memory model cache: reuse while files are unchanged, reload when they change, bounded, opt-out."""

import os
import time

import pytest

from easyml.core import model_cache
from easyml.core.artifacts import load_pipeline, load_tabular_model
from easyml.trainer import MLTrainer
from easyml.utils.database_operations import get_record_by_id

ROW = {"age": "45", "income": "", "city": "NY", "owns_car": "yes", "rooms": "3"}


@pytest.fixture(scope="module")
def run(classification_csv):
    return MLTrainer(task="MLClassification", name="cache", dataset_path=classification_csv, target_column="label",
                     models=["logistic_regression", "random_forest"]).run()


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):
    monkeypatch.delenv("EASYML_MODEL_CACHE", raising=False)
    monkeypatch.delenv("EASYML_MODEL_CACHE_MB", raising=False)
    model_cache.clear()
    yield
    model_cache.clear()


def test_models_are_reused_while_unchanged(run):
    folder = run.path_to_save
    first = load_tabular_model(folder, "RandomForestClassifier", "MLClassification")
    assert load_tabular_model(folder, "RandomForestClassifier", "MLClassification") is first
    assert load_pipeline(folder) is load_pipeline(folder)
    assert model_cache.cached_count() == 2


def test_changed_files_are_loaded_again(run):
    folder = run.path_to_save
    path = os.path.join(folder, "LogisticRegression.pkl")
    first = load_tabular_model(folder, "LogisticRegression", "MLClassification")
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert load_tabular_model(folder, "LogisticRegression", "MLClassification") is not first


def test_cache_can_be_disabled_or_bounded(run, monkeypatch):
    folder = run.path_to_save
    monkeypatch.setenv("EASYML_MODEL_CACHE", "0")
    assert load_pipeline(folder) is not load_pipeline(folder)
    assert model_cache.cached_count() == 0

    monkeypatch.setenv("EASYML_MODEL_CACHE", "8")
    monkeypatch.setenv("EASYML_MODEL_CACHE_MB", "0.000001")  # smaller than any model file
    assert load_pipeline(folder) is not load_pipeline(folder)

    monkeypatch.setenv("EASYML_MODEL_CACHE_MB", "1024")
    monkeypatch.setenv("EASYML_MODEL_CACHE", "1")
    load_pipeline(folder)
    load_tabular_model(folder, "LogisticRegression", "MLClassification")
    assert model_cache.cached_count() == 1  # least recently used entry was dropped


def test_repeated_predictions_are_identical(client, run):
    payload = {"run_id": run.run_id, "model_name": "RandomForestClassifier", "data": ROW}
    answers = [client.post("/inference", json=payload).get_json() for _ in range(3)]
    assert answers[0] == answers[1] == answers[2]
    started = time.perf_counter()
    client.post("/inference", json=payload)
    assert time.perf_counter() - started < 5


def test_deleting_a_run_forgets_its_models(client, classification_csv):
    doomed = MLTrainer(task="MLClassification", name="cache-delete", dataset_path=classification_csv,
                       target_column="label", models=["logistic_regression"]).run()
    client.post("/inference", json={"run_id": doomed.run_id, "model_name": "LogisticRegression", "data": ROW})
    folder = get_record_by_id(doomed.run_id)["folder_path"]
    assert any(key[1].startswith(os.path.normcase(os.path.abspath(folder))) for key in model_cache._entries)
    assert client.delete("/runs", json={"run_id": doomed.run_id}).status_code == 200
    assert not any(key[1].startswith(os.path.normcase(os.path.abspath(folder))) for key in model_cache._entries)
