"""
Retrain a saved run on new data, with the same settings (optionally cleaning it with a data-prep recipe first).

    import easyml

    result = easyml.retrain(5, "customers_october.csv")   # same settings as run #5
    print(result.run_id, result.best_model)
    result.load_model().predict("new_customers.csv")

The new run keeps the parent's input schema: the new data must contain every column the parent was
trained on, and columns the parent did not use are ignored. Preprocessing choices (which columns are
categorical, scaling, imputation, tuning level, ...) are replayed exactly. The new run records the
parent's id, so Run History can show where it came from.
"""

import os
from dataclasses import dataclass, field

from easyml.utils.database_operations import get_record_by_id
from easyml.utils.run_meta import parse_tags
from easyml.utils.run_summary import (
    display_dataset_name,
    model_label,
    parse_metrics,
    rank_models,
    settings_for_display,
    task_label,
)

TABULAR_TASKS = ("MLClassification", "MLRegression")
TEXT_TASK = "NLPTextClassification"
IMAGE_TASK = "ImageClassification"

# Saved model class name -> model key, including models whose library may not be installed here.
_TABULAR_KEYS = {
    "LogisticRegression": "logistic_regression", "LinearRegression": "linear_regression",
    "RandomForestClassifier": "random_forest", "RandomForestRegressor": "random_forest",
    "XGBClassifier": "xg_boost", "XGBRegressor": "xg_boost",
    "LGBMClassifier": "light_gbm", "LGBMRegressor": "light_gbm",
    "CatBoostClassifier": "cat_boost", "CatBoostRegressor": "cat_boost",
    "KNeighborsClassifier": "knn", "KNeighborsRegressor": "knn",
    "AdaBoostClassifier": "ada_boost", "AdaBoostRegressor": "ada_boost",
}
_TEXT_KEYS = {
    "LogisticRegression": "logistic_regression", "MultinomialNB": "naive_bayes", "ComplementNB": "complement_nb",
    "LinearSVC": "svm", "RandomForestClassifier": "random_forest", "MLPClassifier": "mlp",
    "LGBMClassifier": "light_gbm",
}
_TEXT_SETTINGS = ("lowercase", "remove_stopwords", "lemmatize", "regex_clean", "regex_pattern", "feature_mode",
                  "vectorizer_type", "max_features", "ngram_range", "min_df", "max_df", "embedding_model",
                  "embedding_batch_size", "embedding_max_length", "hyper_parameter_tuning", "imbalance_handling")
_IMAGE_SETTINGS = ("batch_size", "validation_split", "test_split", "augment", "epochs", "learning_rate",
                   "early_stopping_patience", "reduce_lr_patience", "reduce_lr_factor", "use_gpu")


class DatasetMismatchError(ValueError):
    """The new dataset lacks columns the run was trained on."""

    def __init__(self, message, missing_columns=(), required_columns=()):
        super().__init__(message)
        self.missing_columns = list(missing_columns)
        self.required_columns = list(required_columns)


@dataclass
class RetrainPlan:
    task: str
    name: str
    parent_run_id: int
    kwargs: dict
    warnings: list = field(default_factory=list)
    assumed_defaults: list = field(default_factory=list)


@dataclass
class RunResult:
    """The outcome of a retrain: the new run's id, scores and anything worth knowing about the new data."""

    run_id: int
    parent_run_id: int
    task: str
    run_name: str
    performance: dict
    warnings: list = field(default_factory=list)

    @property
    def best_model(self):
        ranked = rank_models(self.task, self.performance)
        return ranked[0]["key"] if ranked else None

    def load_model(self, model_name=None):
        from easyml.predictor import load_model

        return load_model(self.run_id, model_name=model_name)

    def __repr__(self):
        return (f"RunResult(run_id={self.run_id}, parent_run_id={self.parent_run_id}, "
                f"task={task_label(self.task)!r}, best_model={self.best_model!r})")


# --------------------------------------------------------------------------- reading a saved run

def load_run(run_id):
    """A run's database record plus its saved configuration files."""
    from easyml.core.artifacts import read_json

    record = get_record_by_id(int(run_id))
    folder = record["folder_path"]
    config = read_json(folder, "config_json.json") if folder and os.path.isdir(folder) else {}
    if not config:
        raise ValueError(f"The files of run #{run_id} are missing, so it cannot be retrained.")
    return {
        "record": record,
        "folder": folder,
        "config": config,
        "column_data": read_json(folder, "column_data.json"),
        "metrics": parse_metrics(record.get("metrics_json")),
    }


def _feature_columns(column_data):
    dtypes = column_data.get("column_dtypes") or {}
    if dtypes:
        return list(dtypes)
    return list(column_data.get("numerical_columns") or []) + list(column_data.get("categorical_columns") or [])


def _available_models(task):
    if task == "MLClassification":
        from easyml.config.GLOBAL_VARS import ui2back_mappings_classification_models as models
    elif task == "MLRegression":
        from easyml.config.GLOBAL_VARS import ui2back_mappings_regression_models as models
    elif task == TEXT_TASK:
        from easyml.nlp.nlp_config import AVAILABLE_NLP_MODELS as models
    else:
        from easyml.cv.cv_config import AVAILABLE_CV_MODELS as models
    return list(models)


def _saved_models(task, config, metrics):
    if config.get("models"):
        return list(config["models"])
    keys = _TABULAR_KEYS if task in TABULAR_TASKS else _TEXT_KEYS if task == TEXT_TASK else {}
    return list(dict.fromkeys(keys.get(name, name) for name in metrics))


def _replay_models(task, config, metrics, models, warnings):
    available = _available_models(task)
    if models:
        unknown = [m for m in models if m not in available]
        if unknown:
            raise ValueError(f"Models not available in this installation: {', '.join(unknown)}. "
                             f"Available: {', '.join(available)}")
        return list(models)
    saved = _saved_models(task, config, metrics)
    kept = [m for m in saved if m in available]
    dropped = [m for m in saved if m not in available]
    if dropped:
        warnings.append("Skipping models that are not installed here: "
                        + ", ".join(model_label(m) for m in dropped) + ".")
    if not kept:
        raise ValueError("None of the run's models can be trained in this installation.")
    return kept


def _read_table(path):
    from easyml.utils.data_loader import load_data_from_path

    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    return load_data_from_path(path)


def _require_columns(frame, required):
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise DatasetMismatchError(
            f"The new data is missing {len(missing)} column(s) the run was trained on: {', '.join(missing)}.",
            missing_columns=missing, required_columns=required)


def _check_classes(series, config, warnings):
    values = series.dropna()
    if values.nunique() < 2:
        raise ValueError(f"The target column '{series.name}' needs at least two different classes.")
    old = config.get("target_labels") or list((config.get("target_encodings") or {}).keys())
    if old:
        old_set = {str(v) for v in old}
        new_set = {str(v) for v in values.unique()}
        added, removed = sorted(new_set - old_set), sorted(old_set - new_set)
        if added:
            warnings.append("New classes in the target: " + ", ".join(added) + ".")
        if removed:
            warnings.append("Classes no longer in the target: " + ", ".join(removed) + ".")


def _image_folders(dataset_path):
    folders = [p.strip() for p in str(dataset_path or "").split(",") if p.strip()]
    if not folders:
        raise ValueError("Choose a folder of images, with one sub-folder per class.")
    classes = set()
    for folder in folders:
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")
        classes.update(entry.name for entry in os.scandir(folder) if entry.is_dir())
    if len(classes) < 2:
        raise ValueError("The image folder needs at least two class sub-folders.")
    return sorted(classes)


# --------------------------------------------------------------------------- planning

def _apply_recipe(recipe, dataset_path, warnings):
    """Clean ``dataset_path`` with ``recipe`` into a new file in the uploads folder; returns (path, summary)."""
    import uuid

    from easyml.config.paths import UPLOADS_DIR
    from easyml.dataprep.engine import apply_file
    from easyml.dataprep.recipe import Recipe

    if not isinstance(recipe, Recipe):
        try:
            recipe = Recipe.load(int(recipe) if str(recipe).isdigit() else recipe)
        except LookupError as exc:  # a deleted or mistyped recipe is bad input, not a server error
            raise ValueError(str(exc)) from exc
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    output = os.path.join(UPLOADS_DIR, f"clean_{uuid.uuid4().hex[:8]}.csv")
    _, reports = apply_file(dataset_path, recipe.steps, output, freeze=True)
    failed = [r for r in reports if r.get("error")]
    if failed:
        from easyml.dataprep.frozen import remove_sidecars

        os.remove(output)
        remove_sidecars(output)
        raise ValueError(f"Recipe '{recipe.name}' could not clean this file: "
                         + "; ".join(f"{r['label']}: {r['error']}" for r in failed))
    warnings.append(f"Cleaned the new data with recipe '{recipe.name}' ({len(recipe.steps)} steps).")
    return output, {"name": recipe.name, "steps": recipe.steps}


def plan_retrain(run_id, dataset_path, name=None, models=None, original_dataset_name=None, recipe=None):
    """
    Check the new dataset against a saved run and work out the training arguments.

    Raises:
        RunNotFoundError: unknown run.
        DatasetMismatchError: the new data lacks columns the run was trained on.
        ValueError / FileNotFoundError: unusable data or models.
        ImportError: the run needs an optional extra that is not installed.
    """
    from easyml.capabilities import get_capabilities

    run = load_run(run_id)
    record, config, metrics = run["record"], run["config"], run["metrics"]
    task = record["task"]
    run_id = int(run_id)
    warnings, assumed = [], []
    name = (str(name).strip() if name else "") or f"{record.get('run_name') or 'run'}-retrain"
    if config.get("final_of_run"):
        warnings.append(f"Run #{run_id} is a final model (trained on every row of run #{config['final_of_run']}). "
                        "The retrained run is tested on part of the new data again; train its final model once "
                        "you are happy with its scores.")
    capabilities = get_capabilities()
    recipe_info = None
    if recipe is False or (isinstance(recipe, str) and recipe.strip().lower() == "none"):
        recipe = False  # explicitly use the new file as it is
    if recipe is None and config.get("recipe") and task != IMAGE_TASK:
        # The run was trained on cleaned data: clean the new file with the same steps. If they don't fit
        # (e.g. the file was already cleaned and a column to remove is gone), use the file as it is.
        from easyml.dataprep.recipe import Recipe

        own = Recipe(name=config["recipe"].get("name") or "run recipe", steps=config["recipe"].get("steps") or [])
        original_dataset_name = original_dataset_name or os.path.basename(str(dataset_path))
        try:
            dataset_path, recipe_info = _apply_recipe(own, dataset_path, warnings)
        except ValueError:
            warnings.append(f"The run's data-prep recipe '{own.name}' did not fit this file, so it was used as is "
                            "(it looks already cleaned).")
    elif recipe:
        if task == IMAGE_TASK:
            raise ValueError("Data-prep recipes apply to tables and text, not image folders.")
        original_dataset_name = original_dataset_name or os.path.basename(str(dataset_path))
        dataset_path, recipe_info = _apply_recipe(recipe, dataset_path, warnings)

    if task in TABULAR_TASKS:
        frame = _read_table(dataset_path)
        target = config["target_column"]
        features = _feature_columns(run["column_data"])
        required = features + [target]
        _require_columns(frame, required)
        extra = [c for c in frame.columns if c not in required]
        if extra:
            warnings.append(f"Ignoring {len(extra)} column(s) the original run did not use: {', '.join(map(str, extra))}.")
        if task == "MLClassification":
            _check_classes(frame[target], config, warnings)
        else:
            import pandas as pd

            values = frame[target].dropna()
            if pd.to_numeric(values, errors="coerce").isna().any():
                raise ValueError(f"The target column '{target}' must contain numbers for a regression run.")
        kwargs = {
            "task": task,
            "name": name,
            "dataset_path": dataset_path,
            "target_column": target,
            "preprocessing": True,
            "imputation": config.get("imputation", True),
            "scaling": bool(config.get("scaling")),
            "reduce_features": bool(config.get("reduce_features")),
            "hyper_parameter_tuning": config.get("hyper_parameter_tuning") or 0,
            "imbalance_handling": config.get("imbalance_handling") or "none",
            # The resolved list, so new data does not change which columns are treated as categories.
            "categorical_columns": list(config.get("categorical_columns") or []),
            "ignore_columns": list(dict.fromkeys(list(config.get("ignore_columns") or []) + extra)),
            "models": _replay_models(task, config, metrics, models, warnings),
            "original_dataset_name": original_dataset_name or os.path.basename(dataset_path),
            "parent_run_id": run_id,
            "recipe": recipe_info,
            # Same seed and validation as the original (runs before 0.10.0: the old fixed split, no CV).
            "random_seed": config.get("random_seed", 42),
            "cv_folds": config.get("cv_folds", 0),
            "tags": parse_tags(record.get("tags")) or None,
        }

    elif task == TEXT_TASK:
        frame = _read_table(dataset_path)
        target, text_column = config["target_column"], config["text_column"]
        _require_columns(frame, [text_column, target])
        _check_classes(frame[target], config, warnings)
        if config.get("feature_mode") == "embedding" and not capabilities["features"]["embeddings"]:
            raise ImportError("This run uses sentence embeddings. Install them with: "
                              + capabilities["install_hints"]["embeddings"])
        if config.get("lemmatize") and not capabilities["features"]["lemmatize"]:
            raise ImportError("This run lemmatizes text. Set it up with: " + capabilities["install_hints"]["lemmatize"])
        if config.get("stopwords_source") == "nltk":
            warnings.append("The original run used NLTK's stopword list; the retrain uses scikit-learn's.")
        kwargs = {key: config[key] for key in _TEXT_SETTINGS if key in config}
        kwargs.update({
            "task": TEXT_TASK,
            "name": name,
            "dataset_path": dataset_path,
            "target_column": target,
            "text_column": text_column,
            "imputation": config.get("imputation", True),
            "models": _replay_models(task, config, metrics, models, warnings),
            "original_dataset_name": original_dataset_name or os.path.basename(dataset_path),
            "parent_run_id": run_id,
            "recipe": recipe_info,
            "random_seed": config.get("random_seed", 42),
            "cv_folds": config.get("cv_folds", 0),
            "tags": parse_tags(record.get("tags")) or None,
        })

    elif task == IMAGE_TASK:
        if not capabilities["features"]["cv"]:
            raise ImportError("Image models need PyTorch. Install it with: " + capabilities["install_hints"]["cv"])
        from easyml.cv.cv_config import CV_RequestBody

        classes = _image_folders(dataset_path)
        old = set(config.get("classes") or [])
        if old and set(classes) != old:
            added, removed = sorted(set(classes) - old), sorted(old - set(classes))
            if added:
                warnings.append("New classes: " + ", ".join(added) + ".")
            if removed:
                warnings.append("Classes no longer present: " + ", ".join(removed) + ".")
        defaults = CV_RequestBody.model_fields
        kwargs = {}
        for key in _IMAGE_SETTINGS:
            if key in config:
                kwargs[key] = config[key]
            else:
                kwargs[key] = defaults[key].default
                assumed.append(key)
        chosen = _replay_models(task, config, metrics, models, warnings)
        if "CustomCNN" in chosen and not config.get("custom_layers"):
            chosen = [m for m in chosen if m != "CustomCNN"]
            warnings.append("Skipping the custom CNN: its layers were not saved with the run.")
            if not chosen:
                raise ValueError("None of the run's models can be retrained.")
        if assumed:
            warnings.append("This run was trained before Easy-ML 0.6.0, so some settings use defaults: "
                            + ", ".join(assumed) + ".")
        kwargs.update({
            "task": IMAGE_TASK,
            "name": name,
            "dataset_path": dataset_path,
            "models": chosen,
            "image_size": tuple(config.get("image_size") or defaults["image_size"].default),
            "custom_layers": config.get("custom_layers"),
            "parent_run_id": run_id,
            "random_seed": config.get("random_seed", 42),
            "tags": parse_tags(record.get("tags")) or None,
        })
    else:
        raise ValueError(f"Runs of task {task!r} cannot be retrained.")

    return RetrainPlan(task=task, name=name, parent_run_id=run_id, kwargs=kwargs, warnings=warnings,
                       assumed_defaults=assumed)


def is_final_model(run_id):
    """True for a final model: a run trained on every row of an evaluated run (``final_of_run`` in its config)."""
    try:
        return bool(load_run(run_id)["config"].get("final_of_run"))
    except Exception:
        return False


def plan_final(run_id, models=None, name=None):
    """
    Plan the final model of an evaluated run: the chosen models (default: its best one) trained again on
    every row of the run's own data (training and test rows), with the run's settings and the best
    parameters its search found. No rows are left to test it on, so its test and cross-validation scores
    are the evaluated run's, marked as estimates (``estimated_from_run``).

    ``models`` are model class names (``"RandomForestClassifier"``) or keys (``"random_forest"``).

    Raises:
        RunNotFoundError, FileNotFoundError (the run's data is missing),
        ValueError (image runs, a run that already is a final model, models not in the run).
    """
    run = load_run(run_id)
    record, config, metrics = run["record"], run["config"], run["metrics"]
    task, run_id = record["task"], int(run_id)
    if task == IMAGE_TASK:
        raise ValueError("Final models are for table and text runs. Image runs need their validation images "
                         "to decide when to stop training.")
    if config.get("final_of_run"):
        raise ValueError(f"Run #{run_id} already is a final model (of run #{config['final_of_run']}).")
    data_path = os.path.join(run["folder"], "data.csv")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Run #{run_id} has no saved copy of its data, so it cannot be trained again.")

    ranked = [m["key"] for m in rank_models(task, metrics)]
    if not ranked:
        raise ValueError(f"Run #{run_id} has no trained models.")
    keys = _TABULAR_KEYS if task in TABULAR_TASKS else _TEXT_KEYS
    chosen = []
    for wanted in (list(models) if models else ranked[:1]):
        match = wanted if wanted in ranked else next((c for c in ranked if keys.get(c) == wanted), None)
        if match is None:
            raise ValueError(f"{wanted!r} is not a model of run #{run_id}. Its models: {', '.join(ranked)}.")
        if match not in chosen:
            chosen.append(match)

    plan = plan_retrain(run_id, data_path, name=(str(name).strip() if name else "") or
                        f"{record.get('run_name') or 'run'}-final", models=[keys.get(c, c) for c in chosen],
                        original_dataset_name=record.get("dataset_name"), recipe=False)
    kwargs = dict(plan.kwargs)
    kwargs.update(
        final_of_run=run_id,
        fixed_params={c: (metrics.get(c) or {}).get("best_params") or {} for c in chosen},
        estimates={c: {k: v for k, v in (metrics.get(c) or {}).items()
                       if k in ("accuracy_test", "test_metrics", "f1_test", "cv") and v is not None} for c in chosen},
        # data.csv is already clean: the recipe is the parent's, kept as a record (prep.json is copied over)
        recipe=config.get("recipe") if isinstance(config.get("recipe"), dict) else None,
    )
    warnings = [w for w in plan.warnings if not w.startswith("Ignoring")]
    return RetrainPlan(task=task, name=plan.name, parent_run_id=run_id, kwargs=kwargs, warnings=warnings)


def train_final(run_id, models=None, name=None):
    """
    Train the final model of run ``run_id`` on all its rows (see :func:`plan_final`) and return its
    :class:`RunResult`. Use it once cross-validation and the test set have shown the model is good.
    """
    return execute_plan(plan_final(run_id, models=models, name=name))


def retrain(run_id, dataset_path, name=None, models=None, recipe=None):
    """
    Train a new run on ``dataset_path`` with the same settings as run ``run_id``.

    Args:
        run_id: The run to retrain (see Run History or ``easy-ml runs``).
        dataset_path: A CSV/Excel file (tabular and text runs) or an image folder with one
            sub-folder per class (image runs).
        name: Name of the new run (default: ``"<parent name>-retrain"``).
        models: Model keys to train instead of the parent's, e.g. ``["random_forest"]``.
        recipe: A data-prep recipe (name, id or :class:`easyml.prep.Recipe`) applied to the new data first.
            By default a run trained on cleaned data cleans new files with its own recipe;
            ``recipe=False`` uses the new file as it is.

    Returns:
        RunResult
    """
    dataset_path = str(dataset_path)
    return execute_plan(plan_retrain(run_id, dataset_path, name=name, models=models, recipe=recipe))


def execute_plan(plan):
    """Train the run a :class:`RetrainPlan` describes (in this thread) and return its :class:`RunResult`."""
    if plan.task == IMAGE_TASK:
        from easyml.cv.cv_controllers import CVController

        outcome = CVController(kwargs=dict(plan.kwargs)).execute_pipeline()
        new_id, performance = outcome["run_id"], outcome["performance"]
    else:
        from easyml.trainer import MLTrainer

        body = MLTrainer(**plan.kwargs).run()
        new_id, performance = body.run_id, body.performance
    clean = {k: {m: v for m, v in values.items() if m != "model_path"} if isinstance(values, dict) else values
             for k, values in (performance or {}).items()}
    return RunResult(run_id=new_id, parent_run_id=plan.parent_run_id, task=plan.task, run_name=plan.name,
                     performance=clean, warnings=list(plan.warnings))


# --------------------------------------------------------------------------- describing a run

def run_spec(run_id):
    """What the UI needs to retrain or compare a run: its settings and the data it expects. No file paths."""
    from easyml.capabilities import get_capabilities

    record = get_record_by_id(int(run_id))
    task = record["task"]
    spec = {
        "run_id": int(run_id),
        "run_name": record.get("run_name"),
        "task": task,
        "task_label": task_label(task),
        "dataset_name": display_dataset_name(record.get("dataset_name")),
        "parent_run_id": None,
        "origin": record.get("origin") or None,
        "created_with": None,
        "settings": [],
        "retrain": {"supported": False, "reason": None},
    }
    try:
        run = load_run(run_id)
    except ValueError as exc:
        spec["retrain"]["reason"] = str(exc)
        return spec

    config, metrics = run["config"], run["metrics"]
    from easyml.utils.run_summary import summarize_run

    spec["parent_run_id"] = summarize_run(record)["parent_run_id"]
    spec["created_with"] = config.get("easyml_version")
    spec["settings"] = settings_for_display(task, config)

    capabilities = get_capabilities()
    available = set(capabilities["models"].get(task) or [])
    models = [{"key": m, "label": model_label(m), "available": m in available}
              for m in _saved_models(task, config, metrics)]
    retrain = {
        "supported": True,
        "reason": None,
        "blocked_by": None,
        "models": models,
        "default_name": f"{record.get('run_name') or 'run'}-retrain",
        "assumed_defaults": [],
        "target_column": config.get("target_column"),
        "text_column": config.get("text_column"),
        "feature_columns": [],
        "required_columns": [],
        "classes": config.get("target_labels") or config.get("classes") or list((config.get("target_encodings") or {})),
        # The data-prep recipe the run was trained with; new files are cleaned with it unless told otherwise.
        "recipe": ({"name": config["recipe"].get("name"), "steps": len(config["recipe"].get("steps") or [])}
                   if isinstance(config.get("recipe"), dict) and task != IMAGE_TASK else None),
    }
    if task in TABULAR_TASKS:
        features = _feature_columns(run["column_data"])
        retrain.update(input="table", upload_endpoint="/load_data", feature_columns=features,
                       required_columns=features + [config.get("target_column")])
    elif task == TEXT_TASK:
        retrain.update(input="text_table", upload_endpoint="/nlp/load_data",
                       required_columns=[config.get("text_column"), config.get("target_column")])
        if config.get("feature_mode") == "embedding" and not capabilities["features"]["embeddings"]:
            retrain["blocked_by"] = "embeddings"
    else:
        retrain.update(input="image_folder", upload_endpoint=None,
                       assumed_defaults=[k for k in _IMAGE_SETTINGS if k not in config])
        if not capabilities["features"]["cv"]:
            retrain["blocked_by"] = "cv"
    if not any(m["available"] for m in models):
        retrain["supported"] = False
        retrain["reason"] = "None of the run's models are installed here."
    elif retrain["blocked_by"]:
        retrain["supported"] = False
        retrain["reason"] = "This run needs an optional extra: " + capabilities["install_hints"][retrain["blocked_by"]]
    spec["retrain"] = retrain
    return spec
