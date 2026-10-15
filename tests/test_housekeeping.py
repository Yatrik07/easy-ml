"""0.9.2 housekeeping: packaging lists in sync, log rotation, upload clean-up, honest optional-import errors."""

import logging
import os
import re
import sys
import time

import pytest

from easyml.capabilities import OptionalFeatureError, describe_import_failure, optional_import
from easyml.utils import logger as log_module
from easyml.utils.storage import clean_uploads, uploads_usage

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _requirements(text):
    return {line.split("#")[0].strip() for line in text.splitlines()
            if line.split("#")[0].strip() and not line.strip().startswith("-")}


def test_requirements_core_matches_pyproject():
    # The Docker images install requirements-core.txt and then the package with --no-deps.
    pyproject = open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8").read()
    block = re.search(r"^dependencies = \[(.*?)^\]", pyproject, re.S | re.M).group(1)
    declared = {item.strip() for item in re.findall(r'"([^"]+)"', block)}
    listed = _requirements(open(os.path.join(ROOT, "requirements-core.txt"), encoding="utf-8").read())
    assert declared == listed, f"only in pyproject: {declared - listed}; only in requirements-core.txt: {listed - declared}"


def test_log_files_rotate_and_survive_a_refused_rename(tmp_path):
    handler = log_module.SafeRotatingFileHandler(str(tmp_path / "logs.log"), maxBytes=400, backupCount=2,
                                                 encoding="utf-8")
    test_logger = logging.getLogger("easyml-rotation-test")
    test_logger.propagate = False
    test_logger.addHandler(handler)
    try:
        for i in range(60):
            test_logger.warning("line %d %s", i, "x" * 40)
        assert (tmp_path / "logs.log.1").exists() and (tmp_path / "logs.log.2").exists()
        assert not (tmp_path / "logs.log.3").exists()  # only backupCount old files are kept

        def refuse():  # Windows refuses to rename a log file another process has open
            raise PermissionError("in use")
        handler.rotate = lambda source, dest: refuse()
        for i in range(30):
            test_logger.warning("after %d %s", i, "y" * 40)  # never raises
        assert "after 29" in (tmp_path / "logs.log").read_text(encoding="utf-8")
    finally:
        test_logger.removeHandler(handler)
        handler.close()


def test_log_names_and_debug_messages(monkeypatch):
    monkeypatch.setenv("EASYML_LOG_NAME", "serve")
    assert log_module.log_file_name("logs.log") == "serve_logs.log"
    monkeypatch.setenv("EASYML_LOG_NAME", "../evil")
    assert log_module.log_file_name("logs.log") == "evil_logs.log"
    monkeypatch.delenv("EASYML_LOG_NAME")
    assert log_module.log_file_name("logs.log") == "logs.log"
    monkeypatch.setenv("EASYML_LOG_MAX_MB", "0.5")
    assert log_module.max_log_bytes() == 512 * 1024
    log_module.write_log("a routine debug message 7f3a", "debug", "test")
    path = os.path.join(log_module.get_log_directory(), "logs.log")
    if os.path.exists(path):
        assert "7f3a" not in open(path, encoding="utf-8").read()


def test_clean_uploads_only_removes_old_files_easyml_made(tmp_path):
    now = time.time()
    old, fresh = now - 3 * 24 * 3600, now - 600

    def make(name, mtime, size=10):
        path = tmp_path / name
        path.write_bytes(b"x" * size)
        os.utime(path, (mtime, mtime))
        return path

    gone = [make("data_0a1b2c3d.csv", old), make("data_nlp_0a1b2c3e.csv", old), make("clean_0a1b2c3f.csv", old),
            make("sched_0a1b2c40.xlsx", old)]
    kept = [make("data_0a1b2c41.csv", fresh), make("my_notes.csv", old), make("data_final.csv", old)]
    (tmp_path / "photos").mkdir()
    os.utime(tmp_path / "photos", (old, old))

    usage = uploads_usage(folder=str(tmp_path), now=now)
    assert usage["files"] == 5 and usage["old_files"] == 4 and usage["old_bytes"] == 40
    result = clean_uploads(older_than_hours=24, folder=str(tmp_path), now=now)
    assert result == {"deleted": 4, "freed_bytes": 40}
    assert not any(p.exists() for p in gone) and all(p.exists() for p in kept)
    assert (tmp_path / "photos").is_dir()


def test_clean_uploads_endpoint(client):
    info = client.get("/settings/info").get_json()
    assert {"files", "bytes", "old_files", "old_bytes"} <= set(info["uploads"])
    assert client.post("/settings/clean-uploads", json={"older_than_hours": 0}).status_code == 400
    assert client.post("/settings/clean-uploads", json={"older_than_hours": "soon"}).status_code == 400
    response = client.post("/settings/clean-uploads", json={})
    assert response.status_code == 200 and {"deleted", "freed_bytes"} <= set(response.get_json())


def test_missing_and_broken_optional_libraries_get_different_advice():
    missing = describe_import_failure("easyml_no_such_package", "Embedding mode", "nlp", ImportError("nope"))
    assert "Install it with" in missing and "easy-ml-studio[nlp]" in missing
    broken = describe_import_failure("json", "Explanations", "eda", ImportError("circular import in tokenizers"))
    assert "installed but failed to load" in broken and "circular import" in broken and "--force-reinstall json" in broken
    with pytest.raises(OptionalFeatureError, match="Install it with"):
        optional_import("easyml_no_such_package", "Embedding mode", "nlp")
    assert issubclass(OptionalFeatureError, ImportError)


def test_broken_torch_gives_a_json_503(client, monkeypatch):
    from easyml.capabilities import get_capabilities

    if not get_capabilities()["features"]["cv"]:
        pytest.skip("simulates a broken PyTorch install, so PyTorch must be installed")
    # sys.modules[name] = None makes importing that module fail, like a broken install.
    monkeypatch.setitem(sys.modules, "easyml.cv.cv_controllers", None)
    monkeypatch.setitem(sys.modules, "easyml.cv.cv_inference", None)
    for path, body in (("/cv/predict", {"name": "x", "dataset_path": "y", "models": ["SimpleCNN"]}),
                       ("/cv/inference", {"run_id": 1, "data": "", "model_name": "SimpleCNN"})):
        response = client.post(path, json=body)
        assert response.status_code == 503, response.data[:200]
        assert "torch" in response.get_json()["error_message"]
