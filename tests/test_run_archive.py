"""Exporting a run to a .zip and importing it again (also on another computer)."""

import hashlib
import io
import json
import os
import zipfile

import pandas as pd
import pytest

import easyml
from easyml.app import main
from easyml.run_archive import ArchiveError, import_archive
from easyml.trainer import MLTrainer
from easyml.utils.database_operations import get_record_by_id

ROW = {"age": "45", "income": "", "city": "NY", "owns_car": "yes", "rooms": "3"}


@pytest.fixture(scope="module")
def run(classification_csv):
    return MLTrainer(task="MLClassification", name="portable", dataset_path=classification_csv, target_column="label",
                     models=["logistic_regression", "random_forest"]).run()


@pytest.fixture(scope="module")
def archive_path(run, tmp_path_factory):
    return easyml.export_run(run.run_id, output_path=str(tmp_path_factory.mktemp("export") / "portable.zip"))


def test_archive_layout_and_no_paths(archive_path, data_dir):
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
        texts = [archive.read(n).decode("utf-8") for n in names if n.endswith(".json")]
    assert {"manifest.json", "run/config_json.json", "run/column_data.json", "run/preprocessing_pipeline.pkl",
            "run/LogisticRegression.pkl", "run/RandomForestClassifier.pkl", "run/data.csv"} <= names
    assert manifest["format"] == "easyml-run" and manifest["run"]["task"] == "MLClassification"
    assert manifest["includes_data"] and "scikit-learn" in manifest["environment"]["libraries"]
    for text in texts:
        assert data_dir not in text and data_dir.replace("\\", "\\\\") not in text


def test_round_trip_predicts_the_same(run, archive_path, classification_csv):
    new_id = easyml.import_run(archive_path, trusted=True)
    assert new_id != run.run_id
    record = get_record_by_id(new_id)
    assert record["origin"] == "import" and record["run_name"] == "portable"
    frame = pd.read_csv(classification_csv).drop(columns=["label"]).head(10)
    assert easyml.load_model(new_id).predict(frame) == easyml.load_model(run.run_id).predict(frame)
    assert easyml.load_model(new_id, "RandomForestClassifier").predict(ROW) == \
        easyml.load_model(run.run_id, "RandomForestClassifier").predict(ROW)
    with open(os.path.join(record["folder_path"], "config_json.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    assert config["imported_from"]["run_id"] == run.run_id
    assert config["dataset_path"] == os.path.join(record["folder_path"], "data.csv")


def test_import_needs_trust(archive_path):
    with pytest.raises(PermissionError):
        easyml.import_run(archive_path)


def test_export_without_data(run, tmp_path):
    path = easyml.export_run(run.run_id, output_path=str(tmp_path / "nodata.zip"), include_data=False)
    with zipfile.ZipFile(path) as archive:
        assert "run/data.csv" not in archive.namelist()
    assert get_record_by_id(easyml.import_run(path, name="no data", trusted=True))["run_name"] == "no data"


def test_text_round_trip(text_csv, tmp_path):
    text_run = MLTrainer(task="NLPTextClassification", name="topics-archive", dataset_path=text_csv,
                         target_column="label", text_column="text", models=["logistic_regression"], min_df=1).run()
    path = easyml.export_run(text_run.run_id, output_path=str(tmp_path / "topics.zip"))
    new_id = easyml.import_run(path, trusted=True)
    assert easyml.load_model(new_id).predict(["a new laptop chip"]) == ["tech"]


# --------------------------------------------------------------------------- hostile archives

def _rebuild(source, tmp_path, edit_entries=None, edit_manifest=None, extra=None, name="bad.zip"):
    """Copy ``source`` with modified entries/manifest; sha values are recomputed unless the test breaks them."""
    with zipfile.ZipFile(source) as archive:
        entries = {n: archive.read(n) for n in archive.namelist()}
    manifest = json.loads(entries.pop("manifest.json"))
    if edit_entries:
        edit_entries(entries)
    manifest["files"] = [{"name": n[4:], "size": len(b), "sha256": hashlib.sha256(b).hexdigest()}
                         for n, b in entries.items() if n.startswith("run/")]
    if edit_manifest:
        edit_manifest(manifest)
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for n, b in entries.items():
            archive.writestr(n, b)
        for info, data in (extra or []):
            archive.writestr(info, data)
    return str(path)


@pytest.mark.parametrize("bad_name", ["run/../evil.pkl", "/abs.pkl", "run/sub/x.pkl", "run/evil.py", "other.json"])
def test_unexpected_entries_are_refused(archive_path, tmp_path, bad_name):
    path = _rebuild(archive_path, tmp_path, extra=[(bad_name, b"x")])
    with pytest.raises(ArchiveError):
        import_archive(path)


def test_links_and_duplicates_are_refused(archive_path, tmp_path):
    link = zipfile.ZipInfo("run/link.pkl")
    link.external_attr = (0o120777 << 16)
    with pytest.raises(ArchiveError, match="link"):
        import_archive(_rebuild(archive_path, tmp_path, extra=[(link, b"/etc/passwd")], name="link.zip"))
    with pytest.raises(ArchiveError, match="twice"):
        import_archive(_rebuild(archive_path, tmp_path, extra=[("RUN/DATA.CSV".lower(), b"a")], name="dup.zip"))


def test_damaged_or_incomplete_archives_are_refused(archive_path, tmp_path):
    def tamper(manifest):
        manifest["files"][0]["sha256"] = "0" * 64

    with pytest.raises(ArchiveError, match="checksum"):
        import_archive(_rebuild(archive_path, tmp_path, edit_manifest=tamper, name="sha.zip"))

    def drop_model(entries):
        entries.pop("run/RandomForestClassifier.pkl")

    with pytest.raises(ArchiveError, match="missing files"):
        import_archive(_rebuild(archive_path, tmp_path, edit_entries=drop_model, name="missing.zip"))

    def newer(manifest):
        manifest["format_version"] = 99

    with pytest.raises(ArchiveError, match="newer"):
        import_archive(_rebuild(archive_path, tmp_path, edit_manifest=newer, name="newer.zip"))

    def bad_model(manifest):
        manifest["metrics"]["../evil"] = {}

    with pytest.raises(ArchiveError):
        import_archive(_rebuild(archive_path, tmp_path, edit_manifest=bad_model, name="model.zip"))

    plain = tmp_path / "plain.zip"
    with zipfile.ZipFile(plain, "w") as archive:
        archive.writestr("run/data.csv", "a,b\n1,2\n")
    with pytest.raises(ArchiveError, match="manifest"):
        import_archive(str(plain))
    with pytest.raises(ArchiveError):
        import_archive(__file__)


def test_size_limit(archive_path, monkeypatch):
    monkeypatch.setenv("EASYML_MAX_IMPORT_MB", "0.001")
    with pytest.raises(ArchiveError, match="limit"):
        import_archive(archive_path)


def test_failed_import_leaves_nothing_behind(archive_path, tmp_path, data_dir):
    runs_dir = os.path.join(data_dir, "runs")
    before = set(os.listdir(runs_dir))

    def tamper(manifest):
        manifest["files"][-1]["sha256"] = "0" * 64

    with pytest.raises(ArchiveError):
        import_archive(_rebuild(archive_path, tmp_path, edit_manifest=tamper, name="cleanup.zip"))
    assert set(os.listdir(runs_dir)) == before


# --------------------------------------------------------------------------- HTTP and CLI

def test_http_export_and_import(client, run, monkeypatch):
    response = client.get(f"/runs/{run.run_id}/export")
    assert response.status_code == 200 and response.mimetype == "application/zip"
    assert "easyml-run-" in response.headers["Content-Disposition"]
    payload = response.data

    refused = client.post("/runs/import", data={"file": (io.BytesIO(payload), "run.zip")},
                          content_type="multipart/form-data")
    assert refused.status_code == 400 and "trust" in refused.get_json()["error_message"]

    imported = client.post("/runs/import", data={"file": (io.BytesIO(payload), "run.zip"), "trusted": "true",
                                                 "name": "from browser"}, content_type="multipart/form-data")
    assert imported.status_code == 201, imported.data[:300]
    assert imported.get_json()["run_name"] == "from browser"

    junk = client.post("/runs/import", data={"file": (io.BytesIO(b"not a zip"), "x.zip"), "trusted": "true"},
                       content_type="multipart/form-data")
    assert junk.status_code == 400
    assert client.get("/runs/999999/export").status_code == 404

    monkeypatch.setenv("EASYML_MAX_IMPORT_MB", "0.001")
    too_big = client.post("/runs/import", data={"file": (io.BytesIO(payload), "run.zip"), "trusted": "true"},
                          content_type="multipart/form-data")
    assert too_big.status_code == 413


def test_cli_export_and_import(capsys, run, tmp_path):
    target = tmp_path / "cli.zip"
    assert main(["export", str(run.run_id), "-o", str(target)]) == 0
    assert capsys.readouterr().out.strip() == str(target)
    assert main(["import", str(target)]) == 1  # no terminal to confirm in
    assert "--yes" in capsys.readouterr().err
    assert main(["import", str(target), "--yes", "--name", "cli copy"]) == 0
    assert "Imported as run #" in capsys.readouterr().out
    assert main(["export", "999999"]) == 1
