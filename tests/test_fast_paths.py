"""
Faster implementations give exactly the results of the straightforward ones they replaced.
Each test keeps the original logic inline as the reference.
"""

import io
import json
import os
import zipfile

import numpy as np
import pandas as pd
import pytest

from easyml.core.artifacts import _blank_strings
from easyml.routes_settings import _directory_size
from easyml.utils.data_loader import get_data_info


def _mixed_frame(n=60):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "f": rng.random(n), "i": np.arange(n), "s": rng.choice(["a", "b", " ", ""], n),
        "b": rng.choice([True, False], n), "o": [None if i % 4 == 0 else ("x" if i % 3 else 7) for i in range(n)],
    })
    frame.loc[::5, "f"] = np.nan
    return frame


@pytest.mark.parametrize("n", [3, 60])
def test_upload_preview_matches_full_conversion(n):
    frame = _mixed_frame(n)
    safe = frame.replace({np.nan: None})
    np.random.seed(7)
    info = get_data_info(frame)
    np.random.seed(7)
    sample = safe.to_dict(orient="records") if n < 5 else safe.sample(5).to_dict(orient="records")
    head = safe.to_dict(orient="records") if n < 5 else safe.head(5).to_dict(orient="records")
    tail = safe.to_dict(orient="records") if n < 5 else safe.tail(5).to_dict(orient="records")
    assert json.dumps(info["data_sample"]) == json.dumps(sample)
    assert json.dumps(info["data_head"]) == json.dumps(head)
    assert json.dumps(info["data_tail"]) == json.dumps(tail)


@pytest.mark.parametrize("values", [
    ["a", " ", "", "  b ", None, np.nan, 3, 4.5, True, "\t", "　"],
    [None, None],
    [1, 2, None],
    [" ", " "],
])
def test_blank_string_detection_matches_per_cell_check(values):
    series = pd.Series(values, dtype=object)
    expected = series.map(lambda v: isinstance(v, str) and not v.strip())
    assert _blank_strings(series).tolist() == expected.tolist()


def test_storage_size_matches_os_walk(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    for i, folder in enumerate([tmp_path, tmp_path / "a", tmp_path / "a" / "b"]):
        (folder / f"f{i}.bin").write_bytes(b"x" * (100 * (i + 1)))
    try:
        os.symlink(tmp_path / "a" / "f1.bin", tmp_path / "link.bin")
        os.symlink(tmp_path / "missing", tmp_path / "broken.bin")
        os.symlink(tmp_path / "a", tmp_path / "dirlink", target_is_directory=True)
    except (OSError, NotImplementedError):
        pass  # no symlink permission on this machine

    expected = 0
    for dirpath, _, filenames in os.walk(tmp_path):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            if os.path.exists(path):
                expected += os.path.getsize(path)
    assert _directory_size(str(tmp_path)) == expected


def test_class_distribution_reads_only_the_target(client, classification_csv):
    body = client.post("/class_distribution", json={"dataset_path": classification_csv, "target_column": "label"}).get_json()
    counts = pd.read_csv(classification_csv)["label"].dropna().value_counts()
    assert {d["class"]: d["count"] for d in body["distribution"]} == {str(k): int(v) for k, v in counts.items()}
    missing = client.post("/class_distribution", json={"dataset_path": classification_csv, "target_column": "nope"})
    assert missing.status_code == 400 and "Column 'nope' not found" in missing.get_data(as_text=True)


def test_export_archive_contents(client, classification_csv):
    from easyml.trainer import MLTrainer

    run = MLTrainer(task="MLClassification", name="export-fast", dataset_path=classification_csv,
                    target_column="label", models=["logistic_regression"]).run()
    response = client.post("/export", json={"run_id": run.run_id, "model_name": "LogisticRegression"})
    archive = zipfile.ZipFile(io.BytesIO(response.data))
    assert set(archive.namelist()) == {"model.pkl", "pipeline.pkl", "config.json", "column_data.json", "labels.json",
                                       "main.py", "requirements.txt", "Dockerfile", "README.md"}
    folder = run.path_to_save
    with open(os.path.join(folder, "LogisticRegression.pkl"), "rb") as handle:
        assert archive.read("model.pkl") == handle.read()
    with open(os.path.join(folder, "config_json.json"), "rb") as handle:
        assert archive.read("config.json") == handle.read()
    assert json.loads(archive.read("labels.json")) == ["high", "low", "mid"]


def test_batched_grid_search_matches_plain_grid_search():
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV

    from easyml.core.cancellable_search import CancellableGridSearchCV

    X, y = make_classification(n_samples=300, n_features=8, random_state=0)
    grid = {"C": [0.01, 0.1, 1, 10, 100], "fit_intercept": [True, False], "tol": [1e-4, 1e-3]}
    plain = GridSearchCV(LogisticRegression(max_iter=500), grid, cv=4, n_jobs=2).fit(X, y)
    batched = CancellableGridSearchCV(LogisticRegression(max_iter=500), grid, cv=4, n_jobs=2).fit(X, y)
    assert batched.best_params_ == plain.best_params_
    assert list(batched.cv_results_["params"]) == list(plain.cv_results_["params"])
    assert np.array_equal(batched.cv_results_["mean_test_score"], plain.cv_results_["mean_test_score"])
    assert np.array_equal(batched.best_estimator_.coef_, plain.best_estimator_.coef_)


def test_all_cores_restores_the_saved_parameters():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    from easyml.core.model_builder import _PARALLEL_FIT, _all_cores

    forest = RandomForestClassifier()
    with _all_cores(forest, _PARALLEL_FIT, 5000):
        assert forest.n_jobs == -1
    assert forest.n_jobs is None
    with _all_cores(forest, _PARALLEL_FIT, 100):  # small data: not worth the threads
        assert forest.n_jobs is None
    linear = LogisticRegression()
    with _all_cores(linear, _PARALLEL_FIT, 5000):
        assert linear.n_jobs is None


def test_image_cache_gives_the_same_tensors(tmp_path):
    torch = pytest.importorskip("torch")
    from PIL import Image

    from easyml.cv.cv_preprocessing import ImageSampleDataset, ResizedImageCache, build_transforms

    rng = np.random.default_rng(0)
    samples = []
    for i in range(4):
        path = tmp_path / f"{i}.jpg"
        Image.fromarray((rng.random((90, 130, 3)) * 255).astype("uint8")).save(path)
        samples.append((str(path), i % 2))
    full_train, full_eval = build_transforms([32, 48], augment=True)
    post_train, post_eval = build_transforms([32, 48], augment=True, include_resize=False)
    cache = ResizedImageCache([32, 48])
    for full, post in ((full_train, post_train), (full_eval, post_eval)):
        for epoch in range(2):  # the second pass reads from the cache
            torch.manual_seed(epoch)
            plain = [ImageSampleDataset(samples, full)[i][0] for i in range(4)]
            torch.manual_seed(epoch)
            cached = [ImageSampleDataset(samples, post, cache=cache)[i][0] for i in range(4)]
            assert all(torch.equal(a, b) for a, b in zip(plain, cached))
    assert len(cache.images) == 4
    assert ResizedImageCache([32, 48], max_bytes=0).load(samples[0][0]).size == (48, 32)


def test_cli_and_package_start_without_ml_libraries(data_dir):
    import subprocess
    import sys

    code = ("import sys, io, easyml, easyml.app; easyml.app._build_parser(); "
            "from easyml.cli import list_runs; list_runs(out=io.StringIO()); "
            "print(','.join(m for m in ('sklearn', 'xgboost', 'lightgbm', 'catboost', 'torch', 'pandas') "
            "if m in sys.modules))")
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True,
                            env={**os.environ, "EASYML_NO_CONSOLE_LOGS": "1", "EASYML_DATA_DIR": data_dir})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_exported_api_explains_itself(client, classification_csv):
    from easyml.trainer import MLTrainer

    run = MLTrainer(task="MLClassification", name="export-guide", dataset_path=classification_csv,
                    target_column="label", models=["logistic_regression"]).run()
    data = client.get(f"/get_run_data?run_id={run.run_id}").get_json()
    row = data["example_row"]
    assert set(row) == {"age", "income", "city", "owns_car", "rooms"}
    assert isinstance(row["age"], int)

    archive = zipfile.ZipFile(io.BytesIO(client.post("/export", json={"run_id": run.run_id,
                                                                     "model_name": "LogisticRegression"}).data))
    readme = archive.read("README.md").decode("utf-8")
    assert "uvicorn main:app" in readme and "pip install -r requirements.txt" in readme
    assert "Invoke-RestMethod" in readme and "curl -X POST" in readme and "/docs" in readme
    from easyml.utils.export_utils import example_features

    assert example_features(run.path_to_save) == row
    # the one-line example uses this run's real columns, in the dataset's order
    assert json.dumps({"features": example_features(run.path_to_save)}) in readme
    assert "`labels.json`" in readme
