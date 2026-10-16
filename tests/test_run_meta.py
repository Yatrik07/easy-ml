"""0.10.0: run names, notes and tags (API, CLI, retrain inheritance, archives, old databases)."""

import os
import sqlite3

import pytest

from easyml.trainer import MLTrainer
from easyml.utils import database_operations
from easyml.utils.run_meta import clean_tags


@pytest.fixture(scope="module")
def run(classification_csv):
    return MLTrainer(task="MLClassification", name="meta-run", dataset_path=classification_csv, target_column="label",
                     models=["logistic_regression"]).run()


def test_clean_tags():
    assert clean_tags(["  churn ", "Churn", "prod  ready", "", None and "x"]) == ["churn", "prod ready"]
    assert clean_tags("one") == ["one"]
    with pytest.raises(ValueError):
        clean_tags(["t%d" % i for i in range(11)])
    with pytest.raises(ValueError):
        clean_tags(["x" * 31])


def test_edit_name_notes_and_tags(client, run):
    url = f"/runs/{run.run_id}/meta"
    body = client.put(url, json={"run_name": "  Churn v2\tfinal ", "notes": "Best so far.\nUse for October.",
                                 "tags": ["churn", " Churn ", "prod"]}).get_json()
    assert body["run_name"] == "Churn v2 final" and body["tags"] == ["churn", "prod"]
    assert body["notes"] == "Best so far.\nUse for October." and "metrics" not in body
    listed = next(r for r in client.get("/runs").get_json()["data"] if r["id"] == run.run_id)
    assert listed["tags"] == ["churn", "prod"] and listed["notes"].startswith("Best")
    # Fields left out stay as they are; empty values clear them.
    assert client.put(url, json={"notes": ""}).get_json()["tags"] == ["churn", "prod"]
    assert client.put(url, json={"tags": []}).get_json()["tags"] == []
    for bad in ({"run_name": "  "}, {"tags": ["x" * 40]}, {"notes": "n" * 5001}, {"tags": "a,b"} and {"tags": 5}):
        assert client.put(url, json=bad).status_code == 400, bad
    assert client.put("/runs/999999/meta", json={"notes": "x"}).status_code == 404


def test_old_database_gets_the_new_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(database_operations, "db_dir", str(tmp_path))
    with sqlite3.connect(os.path.join(tmp_path, "runs.db")) as conn:  # a 0.9 database
        conn.execute("CREATE TABLE runs(id INTEGER PRIMARY KEY AUTOINCREMENT, run_name TEXT, folder_path TEXT, "
                     "timestamp TEXT, task TEXT, dataset_name TEXT, metrics_json TEXT, parent_run_id INTEGER, origin TEXT)")
        conn.execute("INSERT INTO runs(run_name, task) VALUES ('old', 'MLClassification')")
    database_operations.create_database_table("runs")
    with sqlite3.connect(os.path.join(tmp_path, "runs.db")) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(runs)")]
        assert {"notes", "tags"} <= set(columns)
        assert conn.execute("SELECT run_name, notes, tags FROM runs").fetchone() == ("old", None, None)


def test_retrain_inherits_tags_and_archives_keep_them(client, classification_csv, tmp_path):
    import easyml

    parent = MLTrainer(task="MLClassification", name="tagged", dataset_path=classification_csv, target_column="label",
                       models=["logistic_regression"]).run()
    client.put(f"/runs/{parent.run_id}/meta", json={"tags": ["churn", "q4"], "notes": "baseline"})
    child = easyml.retrain(parent.run_id, classification_csv)
    child_row = next(r for r in client.get("/runs").get_json()["data"] if r["id"] == child.run_id)
    assert child_row["tags"] == ["churn", "q4"] and child_row["notes"] == ""  # notes describe one run only

    archive = easyml.export_run(parent.run_id, str(tmp_path / "tagged.zip"))
    imported = easyml.import_run(archive, trusted=True)
    row = next(r for r in client.get("/runs").get_json()["data"] if r["id"] == imported)
    assert row["tags"] == ["churn", "q4"] and row["notes"] == "baseline"


def test_tag_cli(run, capsys):
    from easyml.app import main

    assert main(["tag", str(run.run_id), "nightly", "best"]) == 0
    assert "nightly, best" in capsys.readouterr().out
    assert main(["runs", "--tag", "NIGHTLY", "--limit", "50"]) == 0
    listing = capsys.readouterr().out
    assert "TAGS" in listing and str(run.run_id) in listing
    assert main(["tag", str(run.run_id), "best", "--remove"]) == 0
    assert "best" not in capsys.readouterr().out.split("tags:")[1]
    assert main(["runs", "--tag", "no-such-tag"]) == 0 and "No runs tagged" in capsys.readouterr().out
    assert main(["tag", "999999", "x"]) == 1
