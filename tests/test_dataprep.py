"""Data prep: every step, recipes, preview caching, the HTTP API, the CLI and training integration."""

import io
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from easyml.dataprep import engine
from easyml.dataprep.engine import apply_steps, preview
from easyml.dataprep.recipe import Recipe
from easyml.dataprep.steps import STEP_SPECS, StepError, default_params, run_step, validate_step


def _frame():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "age": [25.0, np.nan, 40.0, 40.0, 300.0, 33.0],
        "city": [" NY", "ny ", "LA", "LA", None, "SF"],
        "joined": ["2024-01-05", "2023-12-31", "bad", "2024-02-01", None, "2022-07-07"],
        "price": ["1.5", "2", "x", "4", "5", "6"],
    })


def _run(op, frame=None, **params):
    frame = _frame() if frame is None else frame
    before = frame.copy()
    result, note = run_step(frame, validate_step({"op": op, "params": params}))
    pd.testing.assert_frame_equal(frame, before)  # steps never modify their input
    return result, note


def test_columns_steps():
    assert list(_run("drop_columns", columns=["id", "price"])[0].columns) == ["age", "city", "joined"]
    assert list(_run("keep_columns", columns=["city", "age"])[0].columns) == ["city", "age"]
    renamed, _ = _run("rename_columns", mapping={"age": "years", "city": ""})
    assert list(renamed.columns) == ["id", "years", "city", "joined", "price"]
    with pytest.raises(StepError):
        _run("rename_columns", mapping={"age": "city"})
    with pytest.raises(StepError):
        _run("drop_columns", columns=["nope"])


def test_change_type():
    numbers, note = _run("change_type", columns=["price"], to="number")
    assert numbers["price"].tolist()[:2] == [1.5, 2.0] and np.isnan(numbers["price"][2]) and "1 value" in note
    dates, _ = _run("change_type", columns=["joined"], to="date")
    assert str(dates["joined"].dtype).startswith("datetime64") and pd.isna(dates["joined"][2])
    flags = pd.DataFrame({"f": ["Yes", "no", "1", "maybe", None]})
    assert _run("change_type", flags, columns=["f"], to="boolean")[0]["f"].tolist()[:3] == [True, False, True]
    assert pd.isna(_run("change_type", flags, columns=["f"], to="boolean")[0]["f"][3])
    assert str(_run("change_type", columns=["city"], to="category")[0]["city"].dtype) == "category"


def test_row_steps():
    dup = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 3]})
    assert len(_run("remove_duplicates", dup)[0]) == 2
    assert _run("remove_duplicates", dup, columns=["a"], keep="last")[0]["b"].tolist() == [1, 3]
    assert len(_run("remove_missing")[0]) == 4
    assert len(_run("remove_missing", columns=["city", "age"], how="all")[0]) == 6
    assert _run("filter_rows", column="age", operator="gt", value="30")[0]["id"].tolist() == [3, 4, 5, 6]
    assert _run("filter_rows", column="city", operator="contains", value="ny")[0]["id"].tolist() == [1, 2]
    assert _run("filter_rows", column="city", operator="is_empty", action="remove")[0]["id"].tolist() == [1, 2, 3, 4, 6]
    assert _run("filter_rows", column="age", operator="in", value="25, 33")[0]["id"].tolist() == [1, 6]
    assert _run("sort_rows", column="age", ascending=False)[0]["id"].tolist()[:2] == [5, 3]
    with pytest.raises(StepError):
        _run("filter_rows", column="age", operator="gt", value="abc")


def test_value_steps():
    filled, note = _run("fill_missing", strategy="median")
    assert filled["age"][1] == 40.0 and "1 missing value" in note  # only number columns when none chosen
    assert _run("fill_missing", columns=["city"], strategy="value", value="Unknown")[0]["city"][4] == "Unknown"
    assert _run("fill_missing", columns=["city"], strategy="most_frequent")[0]["city"][4] == "LA"
    assert _run("fill_missing", columns=["age"], strategy="previous")[0]["age"][1] == 25.0
    with pytest.raises(StepError):
        _run("fill_missing", columns=["city"], strategy="mean")
    with pytest.raises(StepError):
        _run("fill_missing", strategy="value", value="")
    replaced, _ = _run("replace_values", column="city", mapping={"LA": "Los Angeles", "SF": ""})
    assert replaced["city"][2] == "Los Angeles" and pd.isna(replaced["city"][5])
    assert _run("replace_values", column="age", mapping={"300": "30"})[0]["age"][4] == 30.0
    cleaned, _ = _run("clean_text", columns=["city"], case="upper")
    assert cleaned["city"].tolist()[:2] == ["NY", "NY"] and pd.isna(cleaned["city"][4])  # None, or NaN on pandas 3
    mixed = pd.DataFrame({"m": [" a ", 5, None]})
    assert _run("clean_text", mixed, columns=["m"])[0]["m"].tolist()[:2] == ["a", 5]
    clipped, _ = _run("clip_outliers", columns=["age"])
    q1, q3 = _frame()["age"].quantile([0.25, 0.75])
    assert clipped["age"].max() == pytest.approx(q3 + 1.5 * (q3 - q1))
    assert _run("round_numbers", pd.DataFrame({"x": [1.234, 5.678]}), decimals=1)[0]["x"].tolist() == [1.2, 5.7]


def test_new_column_steps():
    parts, note = _run("date_parts", column="joined", parts=["year", "weekday"], drop_original=True)
    assert "joined" not in parts and parts["joined_year"].tolist()[:2] == [2024, 2023] and "1 value was not" in note
    binned, _ = _run("bin_numbers", column="age", bins=3)
    assert binned["age_bin"].notna().sum() == 5
    numbers = pd.DataFrame({"a": [6.0, 4.0], "b": [3.0, 0.0], "t": ["x", None]})
    assert _run("combine_columns", numbers, left="a", right="b", operation="divide")[0].iloc[:, -1].tolist()[0] == 2.0
    assert pd.isna(_run("combine_columns", numbers, left="a", right="b", operation="divide")[0].iloc[1, -1])
    joined, _ = _run("combine_columns", numbers, left="t", right="a", operation="concat", new_column="j",
                     separator="-")
    assert joined["j"].tolist() == ["x-6.0", "-4.0"]
    with pytest.raises(StepError):
        _run("combine_columns", numbers, left="t", right="a", operation="add")


def test_validation_and_specs():
    for op in STEP_SPECS:
        validate_step({"op": op, "params": {k: v for k, v in default_params(op).items() if v not in ("", [], {})}}) \
            if not any(f.get("required") for f in STEP_SPECS[op]["fields"]) else None
    with pytest.raises(ValueError):
        validate_step({"op": "rm -rf", "params": {}})
    with pytest.raises(ValueError):
        validate_step({"op": "drop_columns", "params": {}})
    with pytest.raises(ValueError):
        validate_step({"op": "change_type", "params": {"columns": ["a"], "to": "spaceship"}})
    with pytest.raises(ValueError):
        validate_step({"op": "drop_columns", "params": {"columns": ["a"], "evil": 1}})


def test_failed_step_is_reported_and_skipped():
    steps = [{"op": "drop_columns", "params": {"columns": ["nope"]}}, {"op": "drop_columns", "params": {"columns": ["id"]}},
             {"op": "drop_columns", "params": {"columns": ["age"]}, "enabled": False}]
    result, reports = apply_steps(_frame(), steps)
    assert reports[0]["error"] and not reports[1]["error"] and "id" not in result and "age" in result
    assert reports[2]["note"] == "" and reports[2]["enabled"] is False


def test_preview_matches_full_apply_and_cache(tmp_path):
    path = tmp_path / "big.csv"
    rng = np.random.default_rng(0)
    pd.DataFrame({"a": rng.integers(0, 100, 12000), "b": rng.choice(["x", " y", None], 12000)}).to_csv(path, index=False)
    steps = [{"op": "clean_text", "params": {"columns": ["b"]}},
             {"op": "fill_missing", "params": {"columns": ["b"], "strategy": "value", "value": "none"}},
             {"op": "filter_rows", "params": {"column": "a", "operator": "ge", "value": "50"}}]
    engine.clear_cache()
    first = preview(str(path), steps)
    assert first["sampled"] and first["sample_rows"] == 5000
    full, _ = apply_steps(pd.read_csv(path, nrows=5000), steps)
    expected = full.head(50).astype(object).where(full.head(50).notna(), None).to_dict(orient="records")
    assert first["head"] == expected and first["rows"] == len(full)
    again = preview(str(path), steps)  # served from the cached prefixes
    assert again == first
    changed = preview(str(path), steps[:2] + [{"op": "filter_rows", "params": {"column": "a", "operator": "lt",
                                                                               "value": "50"}}])
    assert changed["rows"] == 5000 - len(full)


def test_preview_dates_and_exact_sample_size(tmp_path):
    path = tmp_path / "dates.csv"
    pd.DataFrame({"day": ["2022-01-05", "2022-02-10", None], "at": ["2022-01-05 13:45", "2022-01-06", None]}).to_csv(
        path, index=False)
    result = preview(str(path), [{"op": "change_type", "params": {"columns": ["day", "at"], "to": "date"}}])
    # Written like the cleaned CSV: date only when every time is midnight; never as an HTTP date.
    assert result["head"][0] == {"day": "2022-01-05", "at": "2022-01-05 13:45:00"}
    assert result["head"][2] == {"day": None, "at": None}
    exact = tmp_path / "exact.csv"
    pd.DataFrame({"a": range(engine.PREVIEW_ROWS)}).to_csv(exact, index=False)
    assert preview(str(exact), [])["sampled"] is False


def test_recipes_round_trip(tmp_path):
    recipe = Recipe(name="Customers", steps=[{"op": "drop_columns", "params": {"columns": ["id"]}}])
    recipe_id = recipe.save()
    assert Recipe.load("customers").id == recipe_id and Recipe.load(recipe_id).steps == recipe.steps
    with pytest.raises(ValueError):
        Recipe(name="customers", steps=[]).save()
    Recipe(name="Customers", steps=[{"op": "drop_columns", "params": {"columns": ["age"]}}]).save(overwrite=True)
    assert Recipe.load("Customers").steps[0]["params"]["columns"] == ["age"]
    path = tmp_path / "r.json"
    path.write_text(recipe.to_json(), encoding="utf-8")
    assert Recipe.load(str(path)).steps == recipe.steps
    with pytest.raises(LookupError):
        Recipe.load("does not exist")
    from easyml import prep

    assert any(r["name"] == "Customers" for r in prep.list_recipes())
    assert list(prep.apply("Customers", _frame()).columns) == ["id", "city", "joined", "price"]


# --------------------------------------------------------------------------- HTTP

def _upload(client, frame, name="messy.csv"):
    data = client.post("/load_data", data={"file": (io.BytesIO(frame.to_csv(index=False).encode()), name)},
                       content_type="multipart/form-data").get_json()
    return data["saved_path"]


def test_http_steps_preview_apply_download(client, classification_csv):
    specs = client.get("/prep/steps").get_json()
    assert "Columns" in specs["categories"] and any(s["op"] == "fill_missing" for s in specs["steps"])

    raw = pd.read_csv(classification_csv)
    raw["junk"] = 1
    path = _upload(client, raw)
    steps = [{"op": "drop_columns", "params": {"columns": ["junk"]}},
             {"op": "fill_missing", "params": {"strategy": "median"}}]
    shown = client.post("/prep/preview", json={"dataset_path": path, "steps": steps}).get_json()
    assert "junk" not in shown["columns"] and shown["steps"][1]["note"].startswith("Filled")
    assert any(p["name"] == "income" and p["missing"] == 0 for p in shown["profile"])

    applied = client.post("/prep/apply", json={"dataset_path": path, "steps": steps,
                                               "original_filename": "messy.csv"}).get_json()
    assert applied["original_filename"] == "messy_clean.csv" and applied["rows"] == len(raw)
    assert os.path.basename(applied["saved_path"]).startswith("clean_")

    training = client.post("/ml_predict", json={
        "task": "MLClassification", "name": "from-prep", "dataset_path": applied["saved_path"],
        "original_filename": applied["original_filename"], "target_column": "label", "models": ["logistic_regression"],
        "recipe": {"name": "Unsaved steps", "steps": steps}})
    assert training.status_code == 202

    download = client.post("/prep/download", json={"dataset_path": path, "steps": steps, "original_filename": "m.xlsx"})
    assert download.status_code == 200 and "m_clean.csv" in download.headers["Content-Disposition"]
    assert "junk" not in pd.read_csv(io.BytesIO(download.data)).columns

    assert client.post("/prep/preview", json={"dataset_path": "C:/Windows/win.ini", "steps": []}).status_code == 403
    unfinished = client.post("/prep/preview", json={"dataset_path": path, "steps": [
        {"op": "drop_columns", "params": {}}, {"op": "drop_columns", "params": {"columns": ["junk"]}}]}).get_json()
    assert unfinished["steps"][0]["incomplete"] and "required" in unfinished["steps"][0]["error"]
    assert "junk" not in unfinished["columns"]  # later steps still run


def test_http_recipes(client):
    created = client.post("/prep/recipes", json={"name": "http recipe",
                                                  "steps": [{"op": "drop_columns", "params": {"columns": ["a"]}}]})
    assert created.status_code == 201
    recipe_id = created.get_json()["id"]
    assert client.post("/prep/recipes", json={"name": "HTTP recipe", "steps": []}).status_code == 400
    assert any(r["id"] == recipe_id for r in client.get("/prep/recipes").get_json()["recipes"])
    updated = client.put(f"/prep/recipes/{recipe_id}", json={"name": "http recipe 2",
                                                             "steps": [{"op": "sort_rows", "params": {"column": "a"}}]})
    assert updated.get_json()["name"] == "http recipe 2" and updated.get_json()["steps"][0]["op"] == "sort_rows"
    assert client.delete(f"/prep/recipes/{recipe_id}").status_code == 200
    assert client.get(f"/prep/recipes/{recipe_id}").status_code == 404


# --------------------------------------------------------------------------- CLI and training

def test_cli(capsys, tmp_path):
    from easyml.app import main

    Recipe(name="cli recipe", steps=[{"op": "drop_columns", "params": {"columns": ["id"]}}]).save(overwrite=True)
    source = tmp_path / "in.csv"
    _frame().to_csv(source, index=False)
    assert main(["prep", "list"]) == 0 and "cli recipe" in capsys.readouterr().out
    assert main(["prep", "show", "cli recipe"]) == 0 and '"drop_columns"' in capsys.readouterr().out
    output = tmp_path / "out.csv"
    assert main(["prep", "apply", "cli recipe", str(source), "-o", str(output)]) == 0
    assert "id" not in pd.read_csv(output).columns
    assert main(["prep", "apply", "missing recipe", str(source)]) == 1


def test_retrain_with_recipe_and_auto_recipe(client, classification_csv, data_dir):
    import easyml
    from easyml.trainer import MLTrainer
    from easyml.utils.database_operations import get_record_by_id

    raw = pd.read_csv(classification_csv)
    raw["junk"] = "x"
    raw_path = os.path.join(data_dir, "uploads", "raw_for_recipe.csv")
    raw.to_csv(raw_path, index=False)
    steps = [{"op": "drop_columns", "params": {"columns": ["junk"]}},
             {"op": "fill_missing", "params": {"strategy": "median"}}]
    clean_path = os.path.join(data_dir, "uploads", "clean_for_recipe.csv")
    Recipe(name="r", steps=steps).apply(raw_path, output=clean_path)
    parent = MLTrainer(task="MLClassification", name="prepped", dataset_path=clean_path, target_column="label",
                       models=["logistic_regression"], recipe={"name": "r", "steps": steps}).run()
    folder = get_record_by_id(parent.run_id)["folder_path"]
    config = json.load(open(os.path.join(folder, "config_json.json"), encoding="utf-8"))
    assert config["recipe"]["name"] == "r" and len(config["recipe"]["steps"]) == 2

    # Raw data with the extra column: the run's own recipe is re-applied automatically.
    auto = easyml.retrain(parent.run_id, raw_path)
    assert any("recipe 'r'" in w for w in auto.warnings) and not any("junk" in w for w in auto.warnings)
    # An explicitly chosen saved recipe.
    Recipe(name="saved one", steps=steps).save(overwrite=True)
    chosen = easyml.retrain(parent.run_id, raw_path, recipe="saved one")
    assert any("saved one" in w for w in chosen.warnings)
    # recipe=False (CLI / studio: "none") uses the file as it is: the extra column is just ignored.
    as_is = easyml.retrain(parent.run_id, raw_path, recipe=False)
    assert any("junk" in w for w in as_is.warnings) and not any("recipe" in w for w in as_is.warnings)
    from easyml.app import main
    assert main(["retrain", str(parent.run_id), raw_path, "--recipe", "none", "--models", "logistic_regression"]) == 0

    # The retrain dialog learns the run's recipe; a deleted recipe is a 400, not a server error.
    from easyml.retraining import run_spec
    assert run_spec(parent.run_id)["retrain"]["recipe"] == {"name": "r", "steps": 2}
    response = client.post(f"/runs/{parent.run_id}/retrain",
                           json={"dataset_path": raw_path, "recipe_id": "no such recipe"})
    assert response.status_code == 400 and "no such recipe" in response.get_json()["message"]


def test_prep_import_is_light():
    code = "import sys, easyml, easyml.prep; print(','.join(m for m in ('pandas', 'sklearn') if m in sys.modules))"
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True,
                            env={**os.environ, "EASYML_NO_CONSOLE_LOGS": "1"})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""
