import json
import os
import shutil
import tempfile
import traceback
from contextlib import contextmanager

import joblib
import numpy as np

from easyml.core.seeding import apply_seed
from easyml.config.GLOBAL_VARS import (
    CATBOOST_MODEL_NAMES,
    ML_RequestBody,
    error_code_mappings,
    get_default_ml_models,
    get_ml_model_instance,
    get_model_mappings,
    get_param_grid,
)
from easyml.core.evaluation_charts import (
    build_model_charts,
    classification_parts,
    regression_points,
    tabular_importance,
    write_evaluation,
)
from easyml.core.evaluator import get_metrics, get_train_test_metrics
from easyml.core.cancellable_search import CancellableGridSearchCV
from easyml.core.imbalance_handler import unwrap_model, with_resampling
from easyml.core.multi_target import PerTargetModel, estimator_for, weighted_mean, weights_of
from easyml.utils.cancellation import is_cancelled
from easyml.utils.holdout import write_holdout
from easyml.utils.logger import write_log
from easyml.utils.socket_emitter import emit_progress
from easyml.utils.labels import labels_in_code_order
from easyml.utils.utils import NumpyEncoder


# Models whose fit is the same (same seeding, same saved form) when it uses every core: the trees'
# seeds are drawn before the parallel part, and n_jobs is restored before predicting and saving.
_PARALLEL_FIT = {"RandomForestClassifier", "RandomForestRegressor"}
# Below this many training rows, starting the worker threads costs more than it saves.
_PARALLEL_FIT_MIN_ROWS = 1000


@contextmanager
def _all_cores(model, names, n_rows):
    """Temporarily let ``model`` use every core; its saved parameters are unchanged afterwards."""
    if type(model).__name__ not in names or not hasattr(model, "n_jobs") or n_rows < _PARALLEL_FIT_MIN_ROWS:
        yield model
        return
    previous = model.n_jobs
    model.n_jobs = -1
    try:
        yield model
    finally:
        model.n_jobs = previous


def _final_step(estimator):
    return estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator


def _model_charts(model, request_body, task, y_pred_test=None):
    """ROC/PR (classification) or scatter/residuals (regression), plus feature importance."""
    if task == "classification":
        labels = labels_in_code_order(request_body.target_encodings)
        parts = classification_parts(model, request_body.X_test, request_body.y_test, labels)
    else:
        if y_pred_test is None:
            y_pred_test = np.asarray(model.predict(request_body.X_test)).reshape(-1)
        parts = {"regression": lambda: regression_points(request_body.y_test, y_pred_test)}
    parts["importance"] = lambda: tabular_importance(model, request_body, task)
    return build_model_charts(parts)

def _cross_validate(request_body, template, model_name, task, y_train=None):
    """k-fold scores of ``template`` (with its tuned settings) with the preprocessing refit in every fold."""
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline

    from easyml.core.cross_validation import cross_validate

    folds = request_body.cv_folds
    emit_progress(f"🔁 Cross-validating {model_name} on {folds} folds...")

    def make():
        return Pipeline([("prep", clone(request_body.pipeline)), ("model", clone(template))])

    return cross_validate(make, request_body.X_train_raw, request_body.y_train if y_train is None else y_train,
                          folds, request_body.random_seed, task == "classification",
                          on_fold=lambda number, k: emit_progress(f"🔁 {model_name}: fold {number}/{k} done"))


def _fit(request_body, model, param_grid, imbalance_method, scoring, model_name, task, y_train, cv_name=None):
    """
    Fit ``model`` on ``y_train`` (with a grid search when ``param_grid`` has entries) and cross-validate it
    when asked. Returns ``(fitted bare model, best parameters, cross-validation result or None)``.
    """
    temp_dir = None
    if model_name in CATBOOST_MODEL_NAMES:
        temp_dir = tempfile.mkdtemp()
        model.set_params(train_dir=temp_dir, thread_count=1, allow_writing_files=False)

    estimator, search_grid = with_resampling(model, param_grid, imbalance_method, request_body.random_seed)
    try:
        if search_grid:
            search = CancellableGridSearchCV(estimator=estimator, param_grid=search_grid,
                                             cv=4, scoring=scoring, n_jobs=-1)
            search.fit(request_body.X_train, y_train)
            fitted, best_params = search.best_estimator_, search.best_params_
        else:
            # No tuning requested: a single fit instead of a 4-fold search over an empty grid.
            with _all_cores(_final_step(estimator), _PARALLEL_FIT, len(y_train)):
                fitted, best_params = estimator.fit(request_body.X_train, y_train), {}
        # Save and evaluate the bare model: metrics below use the original (not resampled) data.
        best_estimator, best_params = unwrap_model(fitted, best_params)
        cv_result = (_cross_validate(request_body, search.best_estimator_ if search_grid else estimator,
                                     cv_name or model_name, task, y_train)
                     if request_body.cv_folds and not request_body.final_of_run else None)
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
    return best_estimator, best_params, cv_result


# --------------------------------------------------------------------------- several targets (0.13)

def _round(value):
    return round(float(value), 5) if value is not None else None


def _fit_per_target(request_body, template, model_name, task, param_grid, scoring):
    """
    A copy of ``template`` for each target, fitted (and searched) on that target's own column: no target's
    accuracy is traded for another's. Returns ``(PerTargetModel, best params by target, CV by target)``.
    """
    from sklearn.base import clone

    targets = request_body.target_columns
    estimators, params, cv_results = [], {}, {}
    for index, target in enumerate(targets):
        if is_cancelled():
            raise InterruptedError("Training cancelled by user")
        emit_progress(f"🎯 {model_name}: predicting {target} ({index + 1}/{len(targets)})...")
        model = clone(template)
        grid = param_grid
        if request_body.final_of_run:
            model.set_params(**(((request_body.fixed_params.get(model_name) or {}).get(target)) or {}))
            grid = {}
        fitted, best_params, cv_result = _fit(request_body, model, grid, "none", scoring, model_name, task,
                                              request_body.y_train[:, index], f"{model_name} ({target})")
        estimators.append(fitted)
        params[target] = best_params
        cv_results[target] = cv_result
    return PerTargetModel(estimators, targets), params, cv_results


def _target_metrics(request_body, task, y_true, y_pred):
    """
    The single-target metrics of every target, their weighted score and, for classification, the share of
    rows with every target right. Returns ``(metrics, per-target scores)``.
    """
    from types import SimpleNamespace

    by_target, scores = {}, []
    for index, target in enumerate(request_body.target_columns):
        if task == "classification":
            encodings = request_body.target_encodings_by_column[target]
            metrics = get_metrics(SimpleNamespace(target_encodings=encodings), task, y_true[:, index],
                                  y_pred[:, index], labels=list(encodings.keys()))
            score = metrics.get("accuracy")
        else:
            metrics = get_metrics(request_body, task, y_true[:, index], y_pred[:, index])
            score = metrics.get("r2_score")
        by_target[target] = metrics
        scores.append(None if score is None else float(score))
    result = {"targets": by_target, "weighted": _round(weighted_mean(scores, weights_of(request_body)))}
    if task == "classification":
        result["exact_match"] = _round(np.mean(np.all(np.asarray(y_true) == np.asarray(y_pred), axis=1)))
    return result, scores


def _combined_cv(cv_results, request_body):
    """One cross-validation summary for a multi-target model: the weighted mean of the targets' results."""
    results = [cv_results.get(t) for t in request_body.target_columns]
    present = [r for r in results if r]
    if not present:
        return None
    failed = next((r for r in present if r.get("error")), None)
    if failed:
        return {"folds": failed["folds"], "metric": failed["metric"], "error": failed["error"]}
    weights = weights_of(request_body)
    return {"folds": present[0]["folds"], "metric": present[0]["metric"],
            "mean": _round(weighted_mean([r and r.get("mean") for r in results], weights)),
            "std": _round(weighted_mean([r and r.get("std") for r in results], weights))}


def _per_target_charts(model, request_body, task, y_pred_test):
    """The single-target charts of each target's model: ``{"targets": {target: charts}}``."""
    from types import SimpleNamespace

    charts = {}
    for index, target in enumerate(request_body.target_columns):
        estimator, y_test = estimator_for(model, index), request_body.y_test[:, index]
        if task == "classification":
            labels = labels_in_code_order(request_body.target_encodings_by_column[target])
            parts = classification_parts(estimator, request_body.X_test, y_test, labels)
        else:
            parts = {"regression": lambda y=y_test, p=y_pred_test[:, index]: regression_points(y, p)}
        body = SimpleNamespace(pipeline=request_body.pipeline, X_test_raw=request_body.X_test_raw,
                               X_train=request_body.X_train, y_test=y_test)
        parts["importance"] = lambda e=estimator, b=body: tabular_importance(e, b, task)
        charts[target] = build_model_charts(parts)
    return {"targets": charts}


def _record_per_target(request_body, task, model_name, model, params, cv_results):
    """Save a multi-target model and record its scores (weighted, per target and, for classification, exact)."""
    targets = request_body.target_columns
    model_path = os.path.join(request_body.path_to_save, f"{model_name}.pkl")
    joblib.dump(model, model_path, protocol=5)
    train, train_scores = _target_metrics(request_body, task, request_body.y_train, model.predict(request_body.X_train))
    if request_body.final_of_run:
        estimate = request_body.estimates.get(model_name) or {}
        estimated_targets = estimate.get("targets") or {}
        request_body.performance[model_name] = {
            "model_path": model_path,
            "accuracy_train": train["weighted"],
            "accuracy_test": estimate.get("accuracy_test"),
            "best_params": request_body.fixed_params.get(model_name) or {},
            "train_metrics": train,
            "test_metrics": estimate.get("test_metrics") or {},
            "targets": {t: {"accuracy_train": _round(train_scores[i]),
                            **{k: v for k, v in (estimated_targets.get(t) or {}).items() if k != "accuracy_train"}}
                        for i, t in enumerate(targets)},
            "estimated_from_run": request_body.final_of_run,
        }
        if estimate.get("cv"):
            request_body.performance[model_name]["cv"] = estimate["cv"]
        write_log(f"trained final {model_name} for {len(targets)} targets on {len(request_body.y_train)} rows",
                  "info", os.path.basename(__file__))
        return None

    y_pred_test = model.predict(request_body.X_test)
    test, test_scores = _target_metrics(request_body, task, request_body.y_test, y_pred_test)
    per_target = {}
    for index, target in enumerate(targets):
        per_target[target] = {"accuracy_train": _round(train_scores[index]), "accuracy_test": _round(test_scores[index])}
        if cv_results.get(target):
            per_target[target]["cv"] = cv_results[target]
    request_body.performance[model_name] = {
        "model_path": model_path,
        "accuracy_train": train["weighted"],
        "accuracy_test": test["weighted"],
        "best_params": params,
        "train_metrics": train,
        "test_metrics": test,
        "targets": per_target,
    }
    combined = _combined_cv(cv_results, request_body)
    if combined:
        request_body.performance[model_name]["cv"] = combined
    write_log(f"trained {model_name} for {len(targets)} targets: weighted test score {test['weighted']} - saved at "
              f"{model_path}", "info", os.path.basename(__file__))
    return y_pred_test


def _save_final_model(request_body, task, model_name, model, is_catboost):
    """
    Save a final model (trained on every row) and record its scores: the training score is measured, the
    test and cross-validation scores are the evaluated run's (``estimated_from_run``), since no rows are
    left to test it on.
    """
    labels = {'labels': list(request_body.target_encodings.keys())} if task == 'classification' else {}
    y_pred_train = np.asarray(model.predict(request_body.X_train)).reshape(-1)
    train_metrics = get_metrics(request_body, task, request_body.y_train, y_pred_train, **labels)
    accuracy_train = train_metrics.get('r2_score' if task == 'regression' else 'accuracy')
    model_path = os.path.join(request_body.path_to_save, f"{model_name}.{'cbm' if is_catboost else 'pkl'}")
    if is_catboost:
        model.save_model(model_path)
    else:
        joblib.dump(model, model_path, protocol=5)
    estimate = request_body.estimates.get(model_name) or {}
    request_body.performance[model_name] = {
        "model_path": model_path,
        "accuracy_train": round(accuracy_train, 5) if accuracy_train is not None else None,
        "accuracy_test": estimate.get("accuracy_test"),
        "best_params": request_body.fixed_params.get(model_name) or {},
        "train_metrics": train_metrics,
        "test_metrics": estimate.get("test_metrics") or {},
        "estimated_from_run": request_body.final_of_run,
    }
    if estimate.get("cv"):
        request_body.performance[model_name]["cv"] = estimate["cv"]
    write_log(f"trained final {model_name} on {len(request_body.y_train)} rows - saved at {model_path}", "info",
              os.path.basename(__file__))


def ml_model_builder_trainer_controller(request_body: ML_RequestBody):
    """
    Handles model training and evaluation for classification or regression tasks.

    This function:
    - Selects models based on user input or defaults
    - Applies hyperparameter tuning using GridSearchCV
    - Evaluates models using train/test metrics
    - Saves trained models and configuration JSONs

    Args:
        request_body (ML_RequestBody): Object containing all user configuration and data.

    Raises:
        Exception: If any error occurs during training, it logs and re-raises the exception.
    """

    try:

        write_log(f"entered ml_model_builder_trainer_controller()", "debug", os.path.basename(__file__))

        task = "classification" if "classification" in request_body.task.lower() else "regression" if "regression" in request_body.task.lower() else None

        if task is None:
            raise ValueError(f"Unsupported task '{request_body.task}'")

        mappings = get_model_mappings(task)
        unavailable = [model for model in request_body.models if model not in mappings]
        if unavailable:
            raise ValueError(f"These models are not available in this installation: {', '.join(unavailable)}")

        models_to_apply = (
            [get_ml_model_instance(task, model) for model in request_body.models]
            if request_body.models
            else get_default_ml_models(task)
        )
        models_to_apply = [apply_seed(model, request_body.random_seed) for model in models_to_apply]

        # Resampling (SMOTE/Tomek) is part of each model's pipeline, so it only ever sees training folds.
        imbalance_method = request_body.imbalance_handling if task == "classification" else "none"
        if (imbalance_method or "none").lower() != "none":
            emit_progress(f"⚖️ Balancing training data with {imbalance_method.upper()} inside each training fold...", pct=44)

        grid = get_param_grid(task, request_body.hyper_parameter_tuning)

        scoring = 'accuracy' if 'classification' in request_body.task.lower() else 'r2'
        several_targets = len(request_body.target_columns) > 1

        total_models = len(models_to_apply)
        for idx, single_model in enumerate(models_to_apply):
            model_base_pct = int(45 + (idx / max(1, total_models)) * 45)
            emit_progress(f"🤖 Training model {idx+1}/{total_models}: {type(single_model).__name__}...", pct=model_base_pct)

            # Check for cancellation before each model
            if is_cancelled():
                write_log("Training cancelled by user", "info", os.path.basename(__file__))
                raise InterruptedError("Training cancelled by user")

            model_name = type(single_model).__name__
            is_catboost = model_name in CATBOOST_MODEL_NAMES
            param_grid = grid.get(model_name, {})
            if several_targets:
                write_log(f"training - {single_model} - grid - {param_grid} - once per target", "info",
                          os.path.basename(__file__))
                model, params, cv_results = _fit_per_target(request_body, single_model, model_name, task,
                                                            param_grid, scoring)
                y_pred_test = _record_per_target(request_body, task, model_name, model, params, cv_results)
                if y_pred_test is not None:
                    emit_progress(f"📈 Preparing result charts for {model_name}...")
                    request_body.charts[model_name] = _per_target_charts(model, request_body, task, y_pred_test)
                emit_progress(f"✅ Finished {model_name}", pct=model_base_pct + int(45 / max(1, total_models)))
                continue
            if request_body.final_of_run:
                # The final model: the evaluated run's best settings, fitted once on every row (no search).
                single_model.set_params(**(request_body.fixed_params.get(model_name) or {}))
                param_grid = {}
            write_log(f"training - {single_model} - grid - {param_grid}", "info", os.path.basename(__file__))

            best_estimator, best_params, cv_result = _fit(request_body, single_model, param_grid, imbalance_method,
                                                          scoring, model_name, task, request_body.y_train)

            if request_body.final_of_run:
                _save_final_model(request_body, task, model_name, best_estimator, is_catboost)
                emit_progress(f"✅ Finished {model_name}", pct=model_base_pct + int(45 / max(1, total_models)))
                continue

            y_pred_train = best_estimator.predict(request_body.X_train).reshape(-1)
            y_pred_test = best_estimator.predict(request_body.X_test).reshape(-1)
            metrics = get_train_test_metrics(
                request_body,
                task=task,
                y_real_train=request_body.y_train, y_pred_train=y_pred_train,
                y_real_test=request_body.y_test, y_pred_test=y_pred_test,
                **({'labels' : list(request_body.target_encodings.keys())} if task == 'classification' else {})
            )
            if task == 'regression':
                accuracy_train = metrics['train_metrics'].get('r2_score', None)
                accuracy_train = round(accuracy_train,5) if accuracy_train is not None else accuracy_train
                accuracy_test = metrics['test_metrics'].get('r2_score', None)
                accuracy_test = round(accuracy_test,5) if accuracy_test is not None else accuracy_test
            else:
                accuracy_train = metrics['train_metrics'].get('accuracy', None)
                accuracy_train = round(accuracy_train,5) if accuracy_train is not None else accuracy_train
                accuracy_test = metrics['test_metrics'].get('accuracy', None)
                accuracy_test = round(accuracy_test,5) if accuracy_test is not None else accuracy_test

            if accuracy_train is None:
                write_log(f"Train accuracy None for {type(single_model).__name__} for task {request_body.name}.", level='error', module_name=os.path.basename(__file__))
            if accuracy_test is None:
                write_log(f"Test accuracy None for {type(single_model).__name__} for task {request_body.name}.", level='error', module_name=os.path.basename(__file__))

            extension = "cbm" if is_catboost else "pkl"
            model_path = os.path.join(request_body.path_to_save, f"{model_name}.{extension}")

            write_log(f"trained {single_model} with train accuracy: {accuracy_train} and test_accuracy:"
                      f" {accuracy_test} - saving at {model_path}", "info", os.path.basename(__file__))

            if is_catboost:
                best_estimator.save_model(model_path)
            else:
                joblib.dump(best_estimator, model_path, protocol=5)

            request_body.performance[model_name] = {
                "model_path": model_path,
                "accuracy_train": accuracy_train,
                "accuracy_test": accuracy_test,
                "best_params": best_params,
                'train_metrics' : metrics['train_metrics'],
                "test_metrics" : metrics['test_metrics'],
            }
            if cv_result:
                request_body.performance[model_name]["cv"] = cv_result
            emit_progress(f"📈 Preparing result charts for {model_name}...")
            request_body.charts[model_name] = _model_charts(best_estimator, request_body, task, y_pred_test)
            emit_progress(f"✅ Finished {model_name}", pct=model_base_pct + int(45 / max(1, total_models)))

        labels = labels_in_code_order(request_body.target_encodings) if task == "classification" else None
        if not request_body.final_of_run:  # a final model shows the evaluated run's charts (run_inputs)
            meta = {"n_test": len(request_body.y_test), "classes": [str(c) for c in labels] if labels else None}
            if several_targets:
                meta["targets"] = request_body.target_columns
                if task == "classification":
                    meta["target_classes"] = {t: [str(c) for c in labels_in_code_order(e)]
                                              for t, e in request_body.target_encodings_by_column.items()}
            write_evaluation(request_body.path_to_save, request_body.task, request_body.charts, meta)

        # Write column_data.json and config_json.json ONCE after all models are done
        column_data = {}
        column_data['numerical_columns'] = request_body.numeric_columns
        column_data['categorical_columns'] = request_body.categorical_columns
        # Training-time dtypes let inference cast form input (always strings) back to the right types.
        column_data['column_dtypes'] = {col: str(dtype) for col, dtype in request_body.X.dtypes.items()}
        column_data['unique_values'] = {}
        for col in request_body.categorical_columns:
            try:
                column_data['unique_values'][col] = request_body.df[col].dropna().unique().tolist()
            except Exception:
                column_data['unique_values'][col] = []

        json_path = os.path.join(request_body.path_to_save, "config_json.json")
        column_data_path = os.path.join(request_body.path_to_save, "column_data.json")
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(request_body.jsonify(), handle, cls=NumpyEncoder)
        with open(column_data_path, "w", encoding="utf-8") as handle:
            json.dump(column_data, handle, cls=NumpyEncoder)
        if request_body.X_test_raw is not None:
            write_holdout(request_body.path_to_save, request_body.X_test_raw.index, request_body.source_rows)

    except InterruptedError:
        raise
    except Exception as e:
        if request_body.status_code == 200:
           request_body.error_message = "ModelTrainingError"
           request_body.status_code = error_code_mappings[request_body.error_message]
           request_body.error_traceback = traceback.format_exc()

           write_log("error occurred- {} - {}".format(
               request_body.error_message, request_body.error_traceback),
               "error", os.path.basename(__file__))

        raise e
