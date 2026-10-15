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
