"""0.11: runs trained on cleaned data clean prediction input the same way (frozen data-prep recipes)."""

import json
import os
import zipfile

import numpy as np
import pandas as pd
import pytest

from easyml.dataprep.engine import apply_file
from easyml.dataprep.frozen import LOOKS_CLEAN, FrozenRecipe, MissingInputError, read_raw_rows, sidecar_paths
from easyml.dataprep.recipe import Recipe

EVERY_STEP = [
    {"op": "drop_columns", "params": {"columns": ["id"]}},
    {"op": "clean_text", "params": {"columns": ["city"], "case": "upper"}},
    {"op": "replace_values", "params": {"column": "flag", "mapping": {"yes": "Y", "no": "N"}}},
    {"op": "change_type", "params": {"columns": ["flag"], "to": "category"}},
    {"op": "fill_missing", "params": {"columns": ["area"], "strategy": "median"}},
    {"op": "fill_missing", "params": {"columns": ["city"], "strategy": "most_frequent"}},
    {"op": "clip_outliers", "params": {"columns": ["income"], "method": "iqr"}},
    {"op": "fill_missing", "params": {"columns": ["income"], "strategy": "mean"}},
    {"op": "round_numbers", "params": {"columns": ["income"], "decimals": 1}},
    {"op": "date_parts", "params": {"column": "joined", "parts": ["year", "month", "weekday"], "drop_original": True}},
    {"op": "bin_numbers", "params": {"column": "area", "bins": 4, "method": "quantile", "new_column": "area band"}},
    {"op": "bin_numbers", "params": {"column": "rooms", "bins": 3}},
    {"op": "combine_columns", "params": {"left": "area", "right": "rooms", "operation": "divide",
                                         "new_column": "area_per_room"}},
    {"op": "rename_columns", "params": {"mapping": {"city": "town"}}},
    {"op": "filter_rows", "params": {"column": "income", "operator": "gt", "value": "0"}},
    {"op": "remove_duplicates", "params": {}},
    {"op": "sort_rows", "params": {"column": "price"}},
]


def _raw(n=240, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "id": range(n),
        "area": rng.integers(1000, 9000, n).astype(float),
        "rooms": rng.integers(1, 6, n),
        "city": rng.choice([" ny", "NY ", "la", "SF", None], n),
        "joined": pd.date_range("2024-01-01", periods=n, freq="37h").strftime("%Y-%m-%d"),
        "flag": rng.choice(["yes", "no", None], n),
        "income": rng.normal(50000, 15000, n),
    })
    frame.loc[rng.choice(n, n // 8, replace=False), "area"] = np.nan
    frame.loc[rng.choice(n, n // 10, replace=False), "income"] = np.nan
    frame.loc[3, "income"] = 9e6  # an outlier the clip step limits
    frame["price"] = frame["area"].fillna(5000) * 120 + frame["rooms"] * 9000 + rng.normal(0, 5e4, n)
    return frame


def _clean(tmp_path, frame, steps, ext=".csv"):
    source = tmp_path / ("raw" + ext)
    if ext == ".csv":
        frame.to_csv(source, index=False)
    else:
        frame.to_excel(source, index=False)
    output = tmp_path / "clean_0a1b2c3d.csv"
    _, reports = apply_file(str(source), steps, str(output), freeze=True)
    assert not [r for r in reports if r.get("error")]
    prep_path, raw_path = sidecar_paths(output)
    frozen = FrozenRecipe.load(prep_path)
    cleaned = pd.read_csv(output)
    features = [c for c in cleaned.columns if c != "price"]
    frozen.finalize(features, "price", read_raw_rows(raw_path, frozen))
    return frozen, cleaned, features, read_raw_rows(raw_path, frozen), output


def _as_text(frame):
    return frame.astype(str).where(frame.notna(), "<empty>").reset_index(drop=True)


@pytest.mark.parametrize("ext", [".csv", ".xlsx"])
def test_replaying_the_frozen_recipe_reproduces_the_cleaned_data(tmp_path, ext):
    if ext == ".xlsx":
        pytest.importorskip("openpyxl")
    frozen, cleaned, features, raw_rows, output = _clean(tmp_path, _raw(), EVERY_STEP, ext)
    assert len(raw_rows) == len(cleaned)  # raw_data.csv: one raw row per cleaned row, same order

    replayed, notes = frozen.prepare(raw_rows)
    expected = pd.read_csv(output, dtype=str)[features]
    assert notes == [] and list(replayed.columns) == features
    pd.testing.assert_frame_equal(_as_text(replayed), _as_text(expected))

    # One row typed into a form (everything is a string) gives the same cleaned row.
    row = {c: ("" if pd.isna(v) else str(v)) for c, v in raw_rows.iloc[7].items()}
    single, _ = frozen.prepare(pd.DataFrame([row]))
    assert _as_text(single).iloc[0].tolist() == _as_text(expected).iloc[7].tolist()


def test_learned_values_come_from_the_training_data(tmp_path):
    frozen, cleaned, _, raw_rows, _ = _clean(tmp_path, _raw(), EVERY_STEP)
    fill = next(s for s in frozen.steps if s["op"] == "fill_missing" and s["frozen"]["columns"] == ["area"])
    median = float(pd.read_csv(tmp_path / "raw.csv")["area"].median())
    assert fill["frozen"]["values"]["area"] == median

    # A single empty value is filled with the training median, not the median of the input.
    one = raw_rows.iloc[[0]].copy()
    one["area"] = np.nan
    prepared, _ = frozen.prepare(one)
    assert float(prepared["area"].iloc[0]) == median
    # Outside the training range: the highest range, and the clip limits from training.
    huge = raw_rows.iloc[[0]].copy()
    huge["area"], huge["income"] = 10_000_000.0, 10_000_000.0
    prepared, _ = frozen.prepare(huge)
    limits = next(s for s in frozen.steps if s["op"] == "clip_outliers")["frozen"]["limits"]["income"]
    assert prepared["area band"].iloc[0] == next(s for s in frozen.steps if s["op"] == "bin_numbers")["frozen"]["labels"][-1]
    assert float(prepared["income"].iloc[0]) == round(limits[1], 1)


def test_input_columns_rules_and_notes(tmp_path):
    frozen, cleaned, features, raw_rows, _ = _clean(tmp_path, _raw(), EVERY_STEP)
    # The model needs the original columns: no id (dropped), no target, city instead of town, joined instead of its parts.
    assert frozen.input_columns == ["area", "rooms", "city", "joined", "flag", "income"]
    assert frozen.inputs["area"] == {"type": "numeric"} and frozen.inputs["joined"] == {"type": "text"}
    assert frozen.inputs["city"]["type"] == "categorical"

    with pytest.raises(MissingInputError, match="rooms"):
        frozen.prepare(raw_rows.drop(columns=["rooms"]))
    # Data that is already clean is recognised and used as it is.
    as_is, notes = frozen.prepare(cleaned.drop(columns=["price"]).head(3))
    assert notes == [LOOKS_CLEAN] and "town" in as_is.columns
    same, notes = frozen.prepare(cleaned.head(3), cleaned=True)
    assert notes == [] and same is not None
    # A row that a training filter would have removed is predicted, with a note.
    filtered, *_ = _clean(tmp_path, _raw(), [{"op": "filter_rows", "params": {"column": "rooms", "operator": "ge",
                                                                                "value": "1"}}])
    odd = raw_rows.head(3).copy()
    odd.loc[odd.index[:2], "rooms"] = 0
    kept, notes = filtered.prepare(odd)
    assert len(kept) == 3 and len(notes) == 1 and "Filter rows" in notes[0] and "2 row(s) (1, 2)" in notes[0]


def _studio_like_run(data_dir, task="MLRegression", name="frozen-run"):
    """Clean like the Data prep page's "Train on it" (POST /prep/apply), then train on the cleaned upload."""
    from easyml.trainer import MLTrainer

    raw = _raw(300, seed=4)
    source = os.path.join(data_dir, "uploads", f"{name}_raw.csv")
    raw.to_csv(source, index=False)
    steps = EVERY_STEP[:13]
    output = os.path.join(data_dir, "uploads", "clean_" + os.urandom(4).hex() + ".csv")
    apply_file(source, steps, output, freeze=True)
    run = MLTrainer(task=task, name=name, dataset_path=output, target_column="price",
                    models=["linear_regression"], recipe={"name": "house rules", "steps": steps}).run()
    return run, raw, output


def test_studio_run_predicts_from_raw_input(client, data_dir):
    import easyml

    run, raw, output = _studio_like_run(data_dir)
    folder = run.path_to_save
    for name in ("prep.json", "raw_data.csv", "data_profile.json"):
        assert os.path.exists(os.path.join(folder, name)), name

    model = easyml.load_model(run.run_id)
    assert model.cleans_input and model.input_columns == ["area", "rooms", "city", "joined", "flag", "income"]
    clean = pd.read_csv(os.path.join(folder, "data.csv"))
    raw_rows = pd.read_csv(os.path.join(folder, "raw_data.csv"))
    from_raw = model.predict(raw_rows)
    from_clean = model.predict(clean.drop(columns=["price"]), cleaned=True)
    assert np.allclose(from_raw, from_clean)
    assert np.allclose(model.predict(clean.drop(columns=["price"])), from_clean)  # recognised as clean

    # The studio form asks for the original columns and fills them from raw rows.
    data = client.get(f"/get_run_data?run_id={run.run_id}").get_json()
    assert data["columns"] == model.input_columns and data["cleaning"]["automatic"]
    assert data["cleaning"]["name"] == "house rules" and data["column_details"]["joined"]["type"] == "text"
    sample = client.get(f"/runs/{run.run_id}/sample").get_json()
    assert set(sample["values"]) == set(model.input_columns)
    row = sample["row"]
    answer = client.post("/inference", json={"run_id": run.run_id, "model_name": "LinearRegression",
                                             "data": {k: "" if v is None else str(v) for k, v in sample["values"].items()}})
    assert answer.status_code == 200 and np.isclose(answer.get_json()["prediction"], from_clean[row])
    assert "warnings" not in answer.get_json()
    unusual = dict(sample["values"], rooms="40", city="Atlantis")
    flagged = client.post("/inference", json={"run_id": run.run_id, "model_name": "LinearRegression", "data": unusual})
    warnings = flagged.get_json()["warnings"]
    assert any("rooms" in w and "higher" in w for w in warnings) and any("Atlantis" in w for w in warnings)
    missing = client.post("/inference", json={"run_id": run.run_id, "model_name": "LinearRegression",
                                              "data": {"area": "100"}})
    assert missing.status_code == 400 and "original columns" in missing.get_json()["error_message"]

    # Batch predictions keep the original columns and add Prediction.
    raw_csv = raw_rows.head(5).to_csv(index=False).encode()
    import io

    batch = client.post("/inference_batch", data={"run_id": str(run.run_id), "model_name": "LinearRegression",
                                                 "file": (io.BytesIO(raw_csv), "new.csv")})
    result = pd.read_csv(io.BytesIO(batch.data))
    assert batch.status_code == 200 and list(result.columns) == list(raw_rows.columns) + ["Prediction"]
    assert np.allclose(result["Prediction"], from_clean[:5])


def test_explanation_uses_the_cleaned_row(client, data_dir):
    from easyml.capabilities import get_capabilities

    if not get_capabilities()["features"]["explain"]:
        pytest.skip("needs SHAP")
    run, _, _ = _studio_like_run(data_dir, name="frozen-explain")
    sample = client.get(f"/runs/{run.run_id}/sample").get_json()
    body = client.post("/explain", json={"run_id": run.run_id, "model_name": "LinearRegression",
                                         "data": sample["values"]}).get_json()
    names = {f["name"] for f in body["explanation"]["features"]}
    assert "town" in names or "area band" in names  # the model's (cleaned) columns


def test_python_recipe_argument_retrain_and_compare(data_dir):
    import easyml
    from easyml.scheduling.compare import compare, holdout_raw_inputs

    raw = _raw(300, seed=7)
    source = os.path.join(data_dir, "uploads", "py_recipe.csv")
    raw.to_csv(source, index=False)
    recipe = Recipe(name="py rules", steps=EVERY_STEP[:9])
    before = set(os.listdir(os.path.join(data_dir, "uploads")))
    run = easyml.MLTrainer(task="MLRegression", name="py-recipe", dataset_path=source, target_column="price",
                           models=["linear_regression"], recipe=recipe).run()
    assert set(os.listdir(os.path.join(data_dir, "uploads"))) == before  # the cleaned copy was removed
    config = json.load(open(os.path.join(run.path_to_save, "config_json.json"), encoding="utf-8"))
    assert config["recipe"]["name"] == "py rules" and config["original_dataset_name"] == "py_recipe.csv"
    assert easyml.load_model(run.run_id).cleans_input

    # Retraining on new raw data freezes the recipe again, on the new data.
    newer = _raw(300, seed=8)
    newer_path = os.path.join(data_dir, "uploads", "py_recipe_new.csv")
    newer.to_csv(newer_path, index=False)
    child = easyml.retrain(run.run_id, newer_path)
    child_model = easyml.load_model(child.run_id)
    assert child_model.cleans_input
    raw_rows = holdout_raw_inputs(child.run_id)
    assert raw_rows is not None and list(raw_rows.columns) == child_model.input_columns
    outcome = compare(run.run_id, child.run_id)
    assert outcome["compared_on"] == "same new data" and outcome["note"] == ""


def test_text_run_cleans_text_input(data_dir, text_csv):
    import easyml
    from easyml.trainer import MLTrainer

    raw = pd.read_csv(text_csv)
    raw["text"] = raw["text"].str.upper()  # the recipe lower-cases text again
    source = os.path.join(data_dir, "uploads", "text_upper.csv")
    raw.to_csv(source, index=False)
    steps = [{"op": "clean_text", "params": {"columns": ["text"], "case": "lower"}}]
    output = os.path.join(data_dir, "uploads", "clean_" + os.urandom(4).hex() + ".csv")
    apply_file(source, steps, output, freeze=True)
    run = MLTrainer(task="NLPTextClassification", name="text-frozen", dataset_path=output, target_column="label",
                    text_column="text", models=["logistic_regression"], lowercase=False).run()
    model = easyml.load_model(run.run_id)
    assert model.cleans_input and model.input_columns == ["text"]
    shouted = ["THE TEAM WON THE FOOTBALL MATCH", "A FASTER CHIP FOR GAMING LAPTOPS"]
    assert model.predict(shouted) == model.predict([t.lower() for t in shouted], cleaned=True)


def test_serve_export_and_archive(data_dir, tmp_path):
    import easyml
    from easyml.predictor import Predictor
    from easyml.serving.server import create_serve_app
    from easyml.utils.export_utils import create_deployment_package

    run, _, _ = _studio_like_run(data_dir, name="frozen-serve")
    predictor = Predictor(run.run_id)
    api = create_serve_app(predictor).test_client()
    info = api.get("/info").get_json()
    assert list(info["inputs"]["columns"]) == predictor.input_columns and info["inputs"]["cleaning"]["recipe"] == "house rules"
    raw_row = info["example"]["features"]
    assert set(raw_row) == set(predictor.input_columns)
    answer = api.post("/predict", json={"features": raw_row}).get_json()
    assert np.isclose(answer["prediction"], predictor.predict(raw_row)[0])
    assert "original columns" in api.get("/").get_data(as_text=True)
    odd = api.post("/predict", json={"features": dict(raw_row, rooms=99)}).get_json()
    assert any("rooms" in w for w in odd["warnings"])

    zip_path = create_deployment_package(run.run_id, "LinearRegression")
    with zipfile.ZipFile(zip_path) as package:
        names = set(package.namelist())
        main = package.read("main.py").decode()
        requirements = package.read("requirements.txt").decode()
    assert "prep.json" in names and "FrozenRecipe.load('prep.json')" in main and "prep.prepare(df)" in main
    assert f"easy-ml-studio=={easyml.__version__}" in requirements

    archive = easyml.export_run(run.run_id, str(tmp_path / "frozen.zip"))
    imported = easyml.import_run(archive, trusted=True)
    assert np.allclose(easyml.load_model(imported).predict(raw_row), predictor.predict(raw_row))
    lean = easyml.export_run(run.run_id, str(tmp_path / "lean.zip"), include_data=False)
    with zipfile.ZipFile(lean) as package:
        lean_names = set(package.namelist())
    assert "run/prep.json" in lean_names and "run/raw_data.csv" not in lean_names and "run/data.csv" not in lean_names


def test_runs_without_frozen_recipes_are_unchanged(client, classification_csv, data_dir):
    from easyml.trainer import MLTrainer
    from easyml.utils.export_utils import generate_fastapi_app, generate_requirements

    plain = MLTrainer(task="MLClassification", name="plain", dataset_path=classification_csv, target_column="label",
                      models=["logistic_regression"]).run()
    assert not os.path.exists(os.path.join(plain.path_to_save, "prep.json"))
    assert os.path.exists(os.path.join(plain.path_to_save, "data_profile.json"))
    assert client.get(f"/get_run_data?run_id={plain.run_id}").get_json()["cleaning"] is None
    main = generate_fastapi_app("MLClassification", "LogisticRegression", "model.pkl")
    assert "FrozenRecipe" not in main and "easy-ml-studio" not in generate_requirements("LogisticRegression")

    # A run cleaned before 0.11 (recipe recorded, nothing frozen): input must already be clean, as before.
    old = MLTrainer(task="MLClassification", name="old-recipe", dataset_path=classification_csv, target_column="label",
                    models=["logistic_regression"], recipe={"name": "old", "steps": [
                        {"op": "fill_missing", "params": {"strategy": "median"}}]}).run()
    cleaning = client.get(f"/get_run_data?run_id={old.run_id}").get_json()["cleaning"]
    assert cleaning == {"name": "old", "steps": ["Fill missing values"], "automatic": False}


def test_upload_clean_up_removes_side_files(tmp_path):
    import time

    from easyml.utils.storage import clean_uploads

    old = time.time() - 3 * 24 * 3600
    names = ["clean_0a1b2c3d.csv", "clean_0a1b2c3d.prep.json", "clean_0a1b2c3d.raw.csv", "notes.prep.json"]
    for name in names:
        (tmp_path / name).write_text("x")
        os.utime(tmp_path / name, (old, old))
    assert clean_uploads(folder=str(tmp_path))["deleted"] == 3
    assert sorted(os.listdir(tmp_path)) == ["notes.prep.json"]


def test_frozen_recipe_files_are_validated(tmp_path):
    frozen, *_ = _clean(tmp_path, _raw(60), EVERY_STEP[:3])
    data = json.loads(json.dumps(frozen.to_dict()))
    data["steps"][0]["op"] = "run_python"
    with pytest.raises(ValueError):
        FrozenRecipe.from_dict(data)
    data = json.loads(json.dumps(frozen.to_dict()))
    data["format_version"] = 99
    with pytest.raises(ValueError, match="newer"):
        FrozenRecipe.from_dict(data)
