"""Scheduled retraining: timing, data sources, holdout comparison, promote/keep, claims, hosts, HTTP, CLI."""

import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from easyml.scheduling import compare, engine, service, store
from easyml.scheduling.sources import find_data, matching_files
from easyml.scheduling.timing import describe, next_run, parse_every, parse_weekday, validate_trigger


# --------------------------------------------------------------------------- timing

def test_next_run_and_descriptions():
    at = datetime(2026, 10, 14, 10, 30)  # a Wednesday
    assert next_run({"trigger": "interval", "interval_minutes": 90}, at) == datetime(2026, 10, 14, 12, 0)
    assert next_run({"trigger": "daily", "at_time": "11:00"}, at) == datetime(2026, 10, 14, 11, 0)
    assert next_run({"trigger": "daily", "at_time": "10:30"}, at) == datetime(2026, 10, 15, 10, 30)  # not "now" again
    assert next_run({"trigger": "weekly", "weekday": 0, "at_time": "02:00"}, at) == datetime(2026, 10, 19, 2, 0)
    assert next_run({"trigger": "weekly", "weekday": 2, "at_time": "11:00"}, at) == datetime(2026, 10, 14, 11, 0)
    assert next_run({"trigger": "weekly", "weekday": 2, "at_time": "09:00"}, at) == datetime(2026, 10, 21, 9, 0)
    assert next_run({"trigger": "new_data"}, at) is None
    assert describe({"trigger": "interval", "interval_minutes": 360}) == "Every 6 hours"
    assert describe({"trigger": "interval", "interval_minutes": 1440}) == "Every day"
    assert describe({"trigger": "weekly", "weekday": 0, "at_time": "02:00"}) == "Every Monday at 02:00"
    assert parse_every("30m") == 30 and parse_every("6h") == 360 and parse_every("1d") == 1440
    assert parse_weekday("mon") == 0 and parse_weekday("Thu") == 3 and parse_weekday(6) == 6
    for bad in ({"trigger": "hourly"}, {"trigger": "interval", "interval_minutes": 1},
                {"trigger": "daily", "at_time": "25:00"}, {"trigger": "weekly", "at_time": "02:00", "weekday": "x"}):
        with pytest.raises(ValueError):
            validate_trigger(bad.pop("trigger"), **bad)


# --------------------------------------------------------------------------- sources

def test_sources_newest_settled_file(tmp_path):
    folder = tmp_path / "drop"
    folder.mkdir()
    with pytest.raises(FileNotFoundError):
        find_data(str(tmp_path / "missing"), "MLClassification")
    assert find_data(str(folder), "MLClassification", settle_seconds=0) is None
    now = time.time()
    for name, age in (("jan.csv", 300), ("feb.csv", 200), ("notes.txt", 50), ("mar.csv", 2)):
        (folder / name).write_text("a,b\n1,2\n")
        os.utime(folder / name, (now - age, now - age))
    assert [n for n, _, _ in matching_files(str(folder))] == ["mar.csv", "feb.csv", "jan.csv"]
    # mar.csv changed 2 s ago: maybe still being copied, so the newest settled file is used.
    assert find_data(str(folder), "MLClassification", now=now, settle_seconds=10).name == "feb.csv"
    assert find_data(str(folder), "MLClassification", now=now, settle_seconds=0).name == "mar.csv"
    assert find_data(str(folder), "MLClassification", pattern="j*.csv", now=now).name == "jan.csv"
    single = find_data(str(folder / "feb.csv"), "MLClassification", now=now)
    (folder / "feb.csv").write_text("a,b\n1,2\n3,4\n")
    os.utime(folder / "feb.csv", (now - 100, now - 100))
    assert find_data(str(folder / "feb.csv"), "MLClassification", now=now).signature != single.signature


# --------------------------------------------------------------------------- holdout

def _accuracy_on_holdout(run_id, model_name):
    from easyml.predictor import Predictor

    task, inputs, truth = compare.holdout_inputs(run_id)
    return compare.score(task, truth, Predictor(run_id, model_name=model_name).predict(inputs))


def test_holdout_reproduces_stored_test_scores(classification_csv, text_csv, data_dir):
    from easyml.trainer import MLTrainer
    from easyml.utils.run_summary import parse_metrics
    from easyml.utils.database_operations import get_record_by_id

    frame = pd.read_csv(classification_csv)
    frame.loc[[3, 40, 77], "label"] = np.nan  # dropped before the split: holdout rows must still map to data.csv
    path = os.path.join(data_dir, "uploads", "holdout_check.csv")
    frame.to_csv(path, index=False)
    tabular = MLTrainer(task="MLClassification", name="holdout-tab", dataset_path=path, target_column="label",
                        models=["logistic_regression", "random_forest"]).run()
    text = MLTrainer(task="NLPTextClassification", name="holdout-text", dataset_path=text_csv, target_column="label",
                     text_column="text", models=["logistic_regression"]).run()
    for run in (tabular, text):
        metrics = parse_metrics(get_record_by_id(run.run_id)["metrics_json"])
        for model, values in metrics.items():
            assert round(_accuracy_on_holdout(run.run_id, model), 5) == values["accuracy_test"], (run.run_id, model)


# --------------------------------------------------------------------------- engine

def _drift_frame(seed, signal):
    """y depends on x1 or x2 only, so a model trained on one kind is ~50 % accurate on the other."""
    rng = np.random.default_rng(seed)
    n = 400
    frame = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "shop": rng.choice(["a", "b"], n)})
    frame["bought"] = np.where(frame[signal] > 0, "yes", "no")
    return frame


@pytest.fixture(scope="module")
def drift_run(data_dir):
    from easyml.trainer import MLTrainer

    path = os.path.join(data_dir, "uploads", "drift_base.csv")
    _drift_frame(1, "x1").to_csv(path, index=False)
    return MLTrainer(task="MLClassification", name="drift-base", dataset_path=path, target_column="bought",
                     models=["logistic_regression"]).run().run_id


def _drop(folder, name, frame, age=60):
    path = os.path.join(folder, name)
    frame.to_csv(path, index=False)
    stamp = time.time() - age
    os.utime(path, (stamp, stamp))
    return path


def test_schedule_promotes_better_and_keeps_worse(drift_run, tmp_path):
    from easyml.utils.database_operations import get_record_by_id

    folder = str(tmp_path / "incoming")
    os.makedirs(folder)
    _drop(folder, "week1.csv", _drift_frame(1, "x1"), age=600)
    created = datetime.now()
    row = service.create({"run_id": drift_run, "name": "drift weekly", "source_path": folder, "trigger": "interval",
                          "interval_minutes": 60, "min_improvement": 0.05}, now=created)
    assert row["champion"]["id"] == drift_run and row["next_run_at"]

    # Due, but the folder holds the same data as when the schedule was created: nothing is trained.
    later = created + timedelta(minutes=61)
    assert engine.run_due(now=later, settle_seconds=0) == []
    current = store.get_schedule(row["id"])
    assert current["last_status"] == engine.NO_NEW_DATA and current["running_since"] is None

    # New behaviour in the data: the old model is ~50 % right, the new one ~100 % -> promoted.
    _drop(folder, "week2.csv", _drift_frame(2, "x2"), age=300)
    events = engine.run_due(now=later + timedelta(hours=2), settle_seconds=0)
    assert [e["decision"] for e in events] == ["promoted"]
    first = events[0]
    assert first["compared_on"] == compare.SAME_DATA and first["challenger_score"] > first["champion_score"] + 0.3
    assert get_record_by_id(first["challenger_run_id"])["origin"] == "schedule"
    assert get_record_by_id(first["challenger_run_id"])["parent_run_id"] == drift_run
    assert store.get_schedule(row["id"])["champion_run_id"] == first["challenger_run_id"]

    # More data of the same kind: the new model is not 5 points better, so the champion stays.
    _drop(folder, "week3.csv", _drift_frame(3, "x2"), age=120)
    second = engine.run_schedule(row["id"], trigger="time", runner="test")
    assert second["decision"] == "kept champion" and second["champion_after"] == first["challenger_run_id"]
    assert second["challenger_run_id"] != first["challenger_run_id"]
    summary = next(s for s in service.list_summaries() if s["id"] == row["id"])
    assert summary["champion"]["id"] == first["challenger_run_id"] and summary["last_event"]["decision"] == "kept champion"
    champions, decisions = store.run_badges()
    assert "drift weekly" in champions[first["challenger_run_id"]]
    assert decisions[second["challenger_run_id"]] == "kept champion"
    # Snapshots of the source files are removed after use.
    from easyml.config.paths import UPLOADS_DIR

    assert not [n for n in os.listdir(UPLOADS_DIR) if n.startswith("sched_")]
    service.delete(row["id"])


def test_new_data_trigger_failure_is_recorded_once(drift_run, tmp_path):
    source = str(tmp_path / "feed.csv")
    _drop(str(tmp_path), "feed.csv", _drift_frame(4, "x1"))
    row = service.create({"run_id": drift_run, "name": "feed watcher", "source_path": source, "trigger": "new_data"})
    assert engine.run_due(settle_seconds=0) == []  # the existing file is not new
    _drop(str(tmp_path), "feed.csv", _drift_frame(5, "x1").drop(columns=["shop"]), age=30)  # a column went missing
    assert engine.run_due(settle_seconds=0) == []
    events = store.list_events(row["id"])
    assert len(events) == 1 and events[0]["decision"] == "failed" and "shop" in events[0]["message"]
    current = store.get_schedule(row["id"])
    assert current["running_since"] is None and current["last_status"] == "failed"
    assert engine.run_due(settle_seconds=0) == [] and len(store.list_events(row["id"])) == 1  # not retried in a loop
    service.delete(row["id"])


def test_claims_are_exclusive_and_expire(drift_run, tmp_path):
    source = _drop(str(tmp_path), "claim.csv", _drift_frame(6, "x1"))
    row = service.create({"run_id": drift_run, "name": "claim test", "source_path": source, "trigger": "daily",
                          "at_time": "03:00"})
    now = datetime.now()
    assert store.claim(row["id"], "a", now) and not store.claim(row["id"], "b", now)
    with pytest.raises(engine.ScheduleBusy):
        engine.prepare(row["id"], runner="c")
    assert store.claim(row["id"], "b", now + timedelta(hours=store.STALE_HOURS + 1))  # the first one died
    results = []

    def grab(name):
        results.append(store.claim(row["id"], name, now + timedelta(hours=20)))

    store.release(row["id"])
    threads = [threading.Thread(target=grab, args=(f"t{i}",)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results.count(True) == 1
    store.release(row["id"])
    service.delete(row["id"])


def test_schedule_validation(drift_run, tmp_path):
    source = _drop(str(tmp_path), "v.csv", _drift_frame(7, "x1"))
    base = {"run_id": drift_run, "name": "valid", "source_path": source, "trigger": "daily", "at_time": "02:00"}
    for change, words in (({"name": ""}, "name"), ({"source_path": str(tmp_path / "nope.csv")}, "exist"),
                          ({"trigger": "daily", "at_time": "2am"}, "time"), ({"min_improvement": 3}, "between"),
                          ({"run_id": 999999}, "not found"), ({"recipe": "no such recipe"}, "recipe")):
        with pytest.raises((ValueError, LookupError, OSError)) as info:
            service.create({**base, **change})
        assert words.lower() in str(info.value).lower(), (change, info.value)
    created = service.create(base)
    with pytest.raises(ValueError):
        service.create(base)  # same name
    changed = service.update(created["id"], {"trigger": "weekly", "weekday": "fri", "at_time": "06:15"})
    assert changed["when"] == "Every Friday at 06:15" and changed["next_run_at"].endswith("06:15:00")
    assert service.set_enabled(created["id"], False)["next_run_at"] is None
    service.delete(created["id"])


# --------------------------------------------------------------------------- serving a schedule

def test_serve_follows_the_schedule_champion(drift_run, tmp_path):
    from easyml.serving.runner import build_app, follow_schedule

    source = _drop(str(tmp_path), "s.csv", _drift_frame(8, "x1"))
    row = service.create({"run_id": drift_run, "name": "serve follow", "source_path": source, "trigger": "new_data"})
    app, _ = build_app(drift_run, schedule_name=row["name"])
    client = app.test_client()
    assert client.get("/info").get_json()["schedule"] == "serve follow"
    _drop(str(tmp_path), "s.csv", _drift_frame(9, "x2"), age=20)
    event = engine.run_schedule(row["id"], trigger="manual")
    assert event["decision"] == "promoted"
    stop = follow_schedule(app.extensions["easyml_serve_state"], row["id"], drift_run, out=open(os.devnull, "w"),
                           every=0.05)
    try:
        deadline = time.time() + 20
        while time.time() < deadline and client.get("/info").get_json()["run_id"] != event["challenger_run_id"]:
            response = client.post("/predict", json={"features": {"x1": 1.0, "x2": -1.0, "shop": "a"}})
            assert response.status_code == 200  # requests keep working while the model is swapped
            time.sleep(0.05)
        assert client.get("/info").get_json()["run_id"] == event["challenger_run_id"]
    finally:
        stop.set()
    service.delete(row["id"])


# --------------------------------------------------------------------------- HTTP, CLI, import cost

def test_schedule_http_api(client, drift_run, tmp_path):
    folder = str(tmp_path / "http")
    os.makedirs(folder)
    _drop(folder, "a.csv", _drift_frame(10, "x1"))
    check = client.post("/schedules/check-source", json={"run_id": drift_run, "source_path": folder}).get_json()
    assert check["ok"] and check["chosen"] == "a.csv" and check["missing_columns"] == []
    _drop(folder, "b.csv", _drift_frame(11, "x1").drop(columns=["x2"]), age=10)
    check = client.post("/schedules/check-source", json={"run_id": drift_run, "source_path": folder}).get_json()
    assert check["chosen"] == "b.csv" and check["missing_columns"] == ["x2"]
    os.remove(os.path.join(folder, "b.csv"))

    created = client.post("/schedules", json={"run_id": drift_run, "name": "http one", "source_path": folder,
                                              "trigger": "interval", "interval_minutes": 120})
    assert created.status_code == 201
    sid = created.get_json()["id"]
    assert client.post("/schedules", json={"run_id": drift_run, "name": "bad", "source_path": folder,
                                           "trigger": "never"}).status_code == 400
    listing = client.get("/schedules").get_json()
    assert any(s["id"] == sid for s in listing["schedules"]) and listing["scheduler"]["studio"] is False
    assert client.post(f"/schedules/{sid}/toggle", json={"enabled": False}).get_json()["enabled"] is False
    assert client.put(f"/schedules/{sid}", json={"name": "http renamed"}).get_json()["name"] == "http renamed"

    store.claim(sid, "someone else")
    assert client.post(f"/schedules/{sid}/run").status_code == 409
    store.release(sid)
    started = client.post(f"/schedules/{sid}/run")
    assert started.status_code == 202 and started.get_json()["job_id"]
    deadline = time.time() + 60
    while time.time() < deadline and not client.get(f"/schedules/{sid}/events").get_json()["events"]:
        time.sleep(0.2)
    events = client.get(f"/schedules/{sid}/events").get_json()["events"]
    assert events and events[0]["decision"] in ("promoted", "kept champion")
    assert client.get(f"/schedules/{sid}").get_json()["events"][0]["id"] == events[0]["id"]
    runs = client.get("/runs").get_json()["data"]
    trained = next(r for r in runs if r["id"] == events[0]["challenger_run_id"])
    assert trained["origin"] == "schedule" and trained["schedule_decision"] == events[0]["decision"]
    assert client.delete(f"/schedules/{sid}").status_code == 200
    assert client.get(f"/schedules/{sid}").status_code == 404


def test_schedule_cli(drift_run, tmp_path, capsys):
    from easyml.app import main

    source = _drop(str(tmp_path), "cli.csv", _drift_frame(12, "x1"))
    assert main(["schedule", "create", str(drift_run), source, "--name", "cli nightly", "--daily", "02:00"]) == 0
    assert "Next run" in capsys.readouterr().out
    assert main(["schedule", "list"]) == 0 and "cli nightly" in capsys.readouterr().out
    assert main(["schedule", "run", "cli nightly"]) == 0
    assert "Run #" in capsys.readouterr().out
    assert main(["schedule", "events", "cli nightly"]) == 0 and "new #" in capsys.readouterr().out
    assert main(["schedule", "disable", "cli nightly"]) == 0 and "off" in capsys.readouterr().out
    assert main(["schedule", "run-due"]) == 0 and "Nothing was due" in capsys.readouterr().out
    assert main(["schedule", "delete", "cli nightly"]) == 0
    assert main(["schedule", "run", "cli nightly"]) == 1
    assert main(["serve", "--schedule", "no such schedule"]) == 1


def test_scheduling_import_is_light():
    code = ("import sys, easyml, easyml.scheduling.engine, easyml.scheduling.service; "
            "assert callable(easyml.schedules.create) and easyml.prep.Recipe; "  # lazy attributes, no explicit import
            "print(','.join(m for m in ('pandas', 'sklearn') if m in sys.modules))")
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True,
                            env={**os.environ, "EASYML_NO_CONSOLE_LOGS": "1"})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""
