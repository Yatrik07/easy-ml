"""0.11: the data check compares new data with the data a run was trained on."""

import io
import os

import numpy as np
import pandas as pd
import pytest

from easyml.monitoring.datacheck import MIN_ROWS, build_profile, check_frame, check_rows, warnings_for


def _training(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"area": rng.normal(5000, 800, n), "rooms": rng.integers(1, 6, n),
                         "city": rng.choice(["NY", "LA", "SF"], n), "note": [f"free text {i}" for i in range(n)]})


@pytest.fixture(scope="module")
def profile():
    return build_profile(_training(), ["area", "rooms", "city", "note"], text_columns=["note"])


def test_same_kind_of_data_is_ok(profile):
    report = check_frame(profile, _training(seed=1))
    assert report["status"] == "ok" and report["summary"] == "The data looks like the training data."
    assert all(c["psi"] is not None and c["psi"] < 0.1 for c in report["columns"] if c["kind"] != "text")


def test_random_samples_of_the_training_data_are_rarely_flagged():
    # PSI alone flags most 30-100 row samples of the training data itself; the significance test must not.
    training = _training(n=1000, seed=3)
    training.loc[training.sample(frac=0.1, random_state=1).index, "area"] = np.nan
    prof = build_profile(training, ["area", "rooms", "city"])
    for n in (30, 60, 120):
        flagged = sum(check_frame(prof, training.sample(n, random_state=seed))["status"] != "ok" for seed in range(40))
        assert flagged <= 3, (n, flagged)
    shifted = training.sample(60, random_state=0).assign(area=lambda f: f["area"] * 1.5)
    assert check_frame(prof, shifted)["status"] == "major"


def test_shifted_columns_are_found_and_explained(profile):
    new = _training(seed=2)
    new["area"] = new["area"] * 1.6
    new.loc[: len(new) // 3, "city"] = "Boston"
    new.loc[: len(new) // 4, "rooms"] = np.nan
    report = check_frame(profile, new)
    by_name = {c["column"]: c for c in report["columns"]}
    assert report["status"] == "major"
    assert by_name["area"]["status"] == "major" and "much higher" in by_name["area"]["message"]
    assert by_name["city"]["unseen_values"] == ["Boston"] and "'Boston'" in by_name["city"]["message"]
    assert by_name["rooms"]["status"] == "major" and "empty in" in by_name["rooms"]["message"]
    assert by_name["note"]["status"] == "ok"  # free text: only how often it is empty is compared
    assert report["summary"].startswith("3 of 4 columns look different from the training data.")


def test_small_batches_are_checked_row_by_row(profile):
    rows = pd.DataFrame([{"area": "9999", "rooms": "3", "city": "Atlantis", "note": "x"},
                         {"area": "5000", "rooms": "", "city": "NY", "note": "y"}])
    assert len(rows) < MIN_ROWS
    findings = check_rows(profile, rows)
    issues = {(f["row"], f["column"], f["issue"]) for f in findings}
    assert issues == {(0, "area", "above"), (0, "city", "unseen"), (1, "rooms", "missing")}
    report = check_frame(profile, rows)
    assert not report["rows_checked_as_whole"] and report["status"] == "minor"
    one = warnings_for(profile, rows.head(1))
    assert len(one) == 2 and one[0].startswith("'area' is 9,999, higher than anything")
    assert warnings_for(profile, rows)[0].startswith("Row 1:")
    assert check_frame(profile, rows.drop(columns=["note"]))["missing_columns"] == ["note"]


def test_runs_get_a_profile_and_old_runs_one_on_demand(classification_csv):
    from easyml.monitoring.datacheck import PROFILE_FILE, reference_profile
    from easyml.trainer import MLTrainer

    run = MLTrainer(task="MLClassification", name="profiled", dataset_path=classification_csv, target_column="label",
                    models=["logistic_regression"]).run()
    saved = reference_profile(run.path_to_save)
    assert set(saved["columns"]) == {"age", "income", "city", "owns_car", "rooms"}
    os.remove(os.path.join(run.path_to_save, PROFILE_FILE))  # like a run from before 0.11
    rebuilt = reference_profile(run.path_to_save)
    assert rebuilt["columns"].keys() == saved["columns"].keys()
    assert not os.path.exists(os.path.join(run.path_to_save, PROFILE_FILE))  # never written into old runs


def test_endpoint_python_and_cli(client, classification_csv, capsys, tmp_path):
    import easyml
    from easyml.app import main
    from easyml.trainer import MLTrainer

    run = MLTrainer(task="MLClassification", name="checked", dataset_path=classification_csv, target_column="label",
                    models=["logistic_regression"]).run()
    shifted = pd.read_csv(classification_csv).drop(columns=["label"])
    shifted["income"] = shifted["income"] * 4
    body = client.post(f"/runs/{run.run_id}/data-check",
                       data={"file": (io.BytesIO(shifted.to_csv(index=False).encode()), "new.csv")}).get_json()
    assert body["status"] == "major" and any(c["column"] == "income" and c["status"] == "major" for c in body["columns"])
    rows = client.post(f"/runs/{run.run_id}/data-check", json={"rows": [{"age": 30, "city": "Paris"}]}).get_json()
    assert rows["status"] == "minor" and "income" in rows["missing_columns"]
    assert client.post(f"/runs/{run.run_id}/data-check", json={"rows": []}).status_code == 400
    assert client.post("/runs/999999/data-check", json={"rows": [{"a": 1}]}).status_code == 404

    report = easyml.load_model(run.run_id).check(shifted)
    assert not report.ok and "income" in report.summary and list(report.to_frame()["column"]).count("income") == 1

    path = tmp_path / "shifted.csv"
    shifted.to_csv(path, index=False)
    assert main(["check", str(run.run_id), str(path)]) == 0
    printed = capsys.readouterr().out
    assert "COLUMN" in printed and "income" in printed and "major" in printed
    assert main(["predict", str(run.run_id), str(path), "-o", str(tmp_path / "out.csv")]) == 0
    assert "Data check:" in capsys.readouterr().err
    assert main(["predict", str(run.run_id), str(path), "-o", str(tmp_path / "out.csv"), "--no-check"]) == 0
    assert "Data check:" not in capsys.readouterr().err


def test_schedule_events_record_the_data_check(classification_csv, tmp_path):
    from easyml.scheduling import engine, service, store
    from easyml.trainer import MLTrainer

    run = MLTrainer(task="MLClassification", name="sched-check", dataset_path=classification_csv, target_column="label",
                    models=["logistic_regression"]).run()
    folder = tmp_path / "feed"
    folder.mkdir()
    shifted = pd.read_csv(classification_csv)
    shifted["income"] = shifted["income"] * 5
    shifted.to_csv(folder / "new.csv", index=False)
    row = service.create({"run_id": run.run_id, "name": "check feed", "source_path": str(folder), "trigger": "new_data"})
    event = engine.run_schedule(row["id"], trigger="manual")
    latest = service.event_summary(store.list_events(row["id"])[0])
    assert latest["data_check"]["status"] == "major" and latest["data_check"]["columns"][0]["column"] == "income"
    assert "New data:" in event["message"]
    service.delete(row["id"])
