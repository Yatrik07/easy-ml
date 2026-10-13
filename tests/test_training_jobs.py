"""Per-run training jobs: progress rooms, per-job cancellation and clean-up of unfinished run folders."""

import os
import threading
import time

import pytest

from easyml.route_helpers import start_training_job
from easyml.utils.cancellation import check_cancelled, get_job
from easyml.utils.database_operations import enter_record_to_table
from easyml.utils.socket_emitter import emit_progress


def _wait_finished(*jobs, timeout=60):
    deadline = time.time() + timeout
    while not all(job.finished for job in jobs):
        assert time.time() < deadline, "training job did not finish in time"
        time.sleep(0.02)


def _progress(socket_client):
    return [packet["args"][0] for packet in socket_client.get_received() if packet["name"] == "progress"]


def _start(app, fn, label="test", run_folder=None):
    with app.app_context():
        return start_training_job(fn, label, run_folder=run_folder)


def test_progress_only_reaches_clients_in_the_jobs_room(app):
    socketio = app.extensions["easyml_socketio"]
    release = threading.Event()

    def make_job(name):
        def train():
            emit_progress(f"{name}: started", pct=10)
            assert release.wait(10)
            emit_progress(f"{name}: finished", pct=100, final=True, result={"status_code": 200})
        return train

    job_a = _start(app, make_job("A"))
    job_b = _start(app, make_job("B"))
    client_a = socketio.test_client(app)
    client_b = socketio.test_client(app)
    bystander = socketio.test_client(app)
    try:
        time.sleep(0.2)  # the "started" events are emitted before anyone joins and must be replayed
        assert client_a.emit("join_run", {"job_id": job_a.job_id}, callback=True)["ok"]
        assert client_b.emit("join_run", {"job_id": job_b.job_id}, callback=True)["ok"]
        assert not bystander.emit("join_run", {"job_id": "nope"}, callback=True)["ok"]
        release.set()
        _wait_finished(job_a, job_b)

        events_a, events_b = _progress(client_a), _progress(client_b)
        assert [e["message"] for e in events_a] == ["A: started", "A: finished"]
        assert [e["message"] for e in events_b] == ["B: started", "B: finished"]
        assert {e["job_id"] for e in events_a} == {job_a.job_id}
        assert [e["seq"] for e in events_a] == [1, 2]
        assert _progress(bystander) == []
    finally:
        for socket_client in (client_a, client_b, bystander):
            socket_client.disconnect()


def test_stop_training_cancels_only_the_requested_job(app, client):
    def train():
        for _ in range(1000):
            check_cancelled()
            time.sleep(0.01)
        emit_progress("done", pct=100, final=True, result={"status_code": 200})

    job_a = _start(app, train)
    job_b = _start(app, train)

    response = client.post("/stop_training", json={"job_id": job_a.job_id})
    assert response.status_code == 200 and response.get_json()["status"] == "ok"
    _wait_finished(job_a, timeout=10)
    assert job_a.history[-1]["result"]["status"] == "cancelled"
    assert not job_b.finished and not job_b.cancel_event.is_set()

    client.post("/stop_training", json={"job_id": job_b.job_id})
    _wait_finished(job_b, timeout=10)
    # Exactly one final event per job.
    assert sum(event["done"] for event in job_a.history) == 1

    assert client.post("/stop_training", json={"job_id": job_a.job_id}).get_json()["status"] == "idle"
    assert client.post("/stop_training", json={}).status_code == 400
    assert client.post("/stop_training", json={"job_id": "missing"}).status_code == 404


@pytest.mark.parametrize("outcome", ["error", "cancelled"])
def test_unfinished_run_folder_is_removed(app, data_dir, outcome):
    folder = os.path.join(data_dir, "runs", f"orphan_{outcome}")
    os.makedirs(folder)
    with open(os.path.join(folder, "preprocessing_pipeline.pkl"), "wb") as handle:
        handle.write(b"partial")

    def train():
        raise InterruptedError("stop") if outcome == "cancelled" else RuntimeError("boom")

    job = _start(app, train, run_folder=folder)
    _wait_finished(job)
    assert job.history[-1]["result"]["status"] == outcome
    assert not os.path.exists(folder)


def test_run_folder_recorded_in_database_is_kept(app, data_dir):
    folder = os.path.join(data_dir, "runs", "already_saved")
    os.makedirs(folder)
    enter_record_to_table("runs", run_name="saved", folder_path=folder, timestamp="2026-01-01T00-00-00",
                          task="MLClassification", dataset_name="x.csv", metrics_json="{}")

    def train():
        raise RuntimeError("failed after the database insert")

    job = _start(app, train, run_folder=folder)
    _wait_finished(job)
    assert os.path.isdir(folder)


def test_failed_training_request_returns_job_and_leaves_no_folder(client, data_dir, classification_csv):
    runs_dir = os.path.join(data_dir, "runs")
    before = set(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else set()
    response = client.post("/ml_predict", json={
        "task": "MLClassification", "name": "will fail", "dataset_path": classification_csv,
        "target_column": "no_such_column", "models": ["logistic_regression"],
    })
    assert response.status_code == 202
    job = get_job(response.get_json()["job_id"])
    _wait_finished(job)
    assert job.history[-1]["result"]["status"] == "error"
    assert set(os.listdir(runs_dir)) == before


def test_trainer_removes_the_folder_it_created(classification_csv, data_dir):
    from easyml.trainer import MLTrainer

    runs_dir = os.path.join(data_dir, "runs")
    before = set(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else set()
    trainer = MLTrainer(task="MLClassification", name="lib run", dataset_path=classification_csv,
                        target_column="no_such_column", models=["logistic_regression"])
    created = trainer._created_folder
    assert created and os.path.isdir(created)
    with pytest.raises(Exception):
        trainer.run()
    assert not os.path.exists(created)
    assert set(os.listdir(runs_dir)) == before
