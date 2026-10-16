"""
NLP Model Builder Module
=========================
Handles model selection, training via GridSearchCV, evaluation,
and persistence for NLP classification tasks.

Mirrors easyml/core/model_builder.py for NLP tasks.
"""

import os
import json
import traceback
import joblib
import scipy.sparse


from easyml.core.seeding import apply_seed
from easyml.nlp.nlp_config import (
    NLP_RequestBody, NLP_ERROR_CODES,
    get_nlp_model_instance, DEFAULT_NLP_MODELS,
    NLP_PARAM_GRID_LEVEL1, NLP_PARAM_GRID_LEVEL2,
    DENSE_ONLY_MODELS,
)
from easyml.core.cancellable_search import CancellableGridSearchCV
from easyml.core.evaluation_charts import build_model_charts, classification_parts, nlp_top_terms, write_evaluation
from easyml.core.imbalance_handler import unwrap_model, with_resampling
from easyml.nlp.nlp_evaluator import get_nlp_train_test_metrics
from easyml.utils.holdout import write_holdout
from easyml.utils.logger import write_log
from easyml.utils.labels import labels_in_code_order
from easyml.utils.utils import NumpyEncoder
from easyml.utils.socket_emitter import emit_progress
from easyml.utils.cancellation import is_cancelled

# Naive Bayes needs non-negative features (counts / tf-idf), which embeddings are not.
NON_NEGATIVE_ONLY_MODELS = {"naive_bayes", "complement_nb"}


def _to_dense(matrix):
    return matrix.toarray() if scipy.sparse.issparse(matrix) else matrix


def _cross_validate(request_body, template, class_name, dense=False):
    """
    k-fold scores of ``template``: with a fresh vectorizer per fold on the training texts, or on the
    training embeddings (fixed per text, so nothing leaks between folds).
    """
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    from easyml.core.cross_validation import cross_validate

    folds = request_body.cv_folds
    emit_progress(f"🔁 Cross-validating {class_name} on {folds} folds...")
    use_texts = request_body.feature_mode != "embedding" and request_body.vectorizer is not None \
        and request_body.X_text_train is not None

    def make():
        if not use_texts:
            return clone(template)
        steps = [("vectorize", clone(request_body.vectorizer))]
        if dense:
            steps.append(("dense", FunctionTransformer(_to_dense, accept_sparse=True)))
        return Pipeline(steps + [("model", clone(template))])

    data = request_body.X_text_train if use_texts else request_body.X_train
    return cross_validate(make, data, request_body.y_train, folds, request_body.random_seed, True,
                          on_fold=lambda number, k: emit_progress(f"🔁 {class_name}: fold {number}/{k} done"))


def nlp_model_builder_controller(request_body: NLP_RequestBody):
    """
    Trains and evaluates NLP models.

    For each selected model:
    1. Instantiates the model
    2. Runs GridSearchCV with the configured hyperparameter level
    3. Evaluates on train + test sets
    4. Saves best estimator to disk
    5. Stores metrics in request_body.performance

    Args:
        request_body (NLP_RequestBody): Data carrier with features and config.

    Raises:
        Exception: Re-raises with NLPModelTrainingError code.
    """
    try:
        write_log("entered nlp_model_builder_controller", "debug", os.path.basename(__file__))

        # Determine which models to train
        model_names = request_body.models if request_body.models else DEFAULT_NLP_MODELS

        # Select hyperparameter grid level
        if request_body.hyper_parameter_tuning == 0:
            grid = {}
        elif request_body.hyper_parameter_tuning == 1:
            grid = NLP_PARAM_GRID_LEVEL1
        else:
            grid = NLP_PARAM_GRID_LEVEL2

        # Determine if input is sparse
        is_sparse = scipy.sparse.issparse(request_body.X_train)
        dense = None  # (X_train, X_test) as dense arrays, made at most once

        # Resampling (SMOTE/Tomek) is part of each model's pipeline, so it only ever sees training folds.
        imbalance_method = request_body.imbalance_handling or "none"
        if imbalance_method.lower() != "none":
            emit_progress(f"⚖️ Balancing training data with {imbalance_method.upper()} inside each training fold...", pct=44)

        has_negative_features = (not is_sparse) and bool((request_body.X_train < 0).any())

        total_models = len(model_names)
        for idx, model_name in enumerate(model_names):
            if is_cancelled():
                raise InterruptedError("Training cancelled by user")

            if has_negative_features and model_name in NON_NEGATIVE_ONLY_MODELS:
                write_log(f"skipping {model_name}: it cannot be trained on embeddings", "warning", os.path.basename(__file__))
                emit_progress(f"⚠️ Skipped {model_name}: Naive Bayes cannot use embedding features", pct=int(45 + (idx / max(1, total_models)) * 50))
                continue

            model_base_pct = int(45 + (idx / max(1, total_models)) * 50)
            write_log(f"training NLP model: {model_name}", "info", os.path.basename(__file__))
            emit_progress(f"🤖 Training NLP model {idx+1}/{total_models}: {model_name}...", pct=model_base_pct)

            try:
                model = apply_seed(get_nlp_model_instance(model_name), request_body.random_seed)
            except ValueError as e:
                write_log(f"skipping unknown model: {model_name} - {e}", "error", os.path.basename(__file__))
                continue

            class_name = type(model).__name__

            # Handle sparse/dense compatibility
            X_train = request_body.X_train
            X_test = request_body.X_test

            if is_sparse and model_name in DENSE_ONLY_MODELS:
                if dense is None:  # converted once and shared by every model that needs dense input
                    write_log(f"converting sparse to dense for {model_name}", "info", os.path.basename(__file__))
                    dense = (X_train.toarray() if scipy.sparse.issparse(X_train) else X_train,
                             X_test.toarray() if scipy.sparse.issparse(X_test) else X_test)
                X_train, X_test = dense

            # Get param grid for this model
            param_grid = grid.get(class_name, {})

            write_log(f"GridSearchCV for {class_name} with params: {param_grid}", "info", os.path.basename(__file__))

            estimator, search_grid = with_resampling(model, param_grid, imbalance_method, request_body.random_seed)
            best_params = {}
            if search_grid:
                grid_search = CancellableGridSearchCV(
                    estimator=estimator,
                    param_grid=search_grid,
                    cv=4,
                    scoring='accuracy',
                    n_jobs=-1,
                    error_score='raise'
                )
                grid_search.fit(X_train, request_body.y_train)
                fitted, best_params = grid_search.best_estimator_, grid_search.best_params_
            else:
                # No tuning requested: a single fit instead of a 4-fold search over an empty grid.
                fitted = estimator.fit(X_train, request_body.y_train)
            # Save and evaluate the bare model: metrics below use the original (not resampled) data.
            best_estimator, best_params = unwrap_model(fitted, best_params)
            cv_result = None
            if request_body.cv_folds:
                cv_result = _cross_validate(request_body, grid_search.best_estimator_ if search_grid else estimator,
                                            class_name, dense=is_sparse and model_name in DENSE_ONLY_MODELS)

            # Predict
            y_pred_train = best_estimator.predict(X_train)
            y_pred_test = best_estimator.predict(X_test)

            # Evaluate
            labels = list(request_body.target_encodings.keys()) if request_body.target_encodings else None
            kwargs = {'labels': labels} if labels else {}

            metrics = get_nlp_train_test_metrics(
                request_body,
                y_real_train=request_body.y_train, y_pred_train=y_pred_train,
                y_real_test=request_body.y_test, y_pred_test=y_pred_test,
                **kwargs
            )

            accuracy_train = metrics['train_metrics'].get('accuracy')
            accuracy_test = metrics['test_metrics'].get('accuracy')
            accuracy_train = round(accuracy_train, 5) if accuracy_train is not None else None
            accuracy_test = round(accuracy_test, 5) if accuracy_test is not None else None

            f1_train = metrics['train_metrics'].get('f1_weighted')
            f1_test = metrics['test_metrics'].get('f1_weighted')
            f1_train = round(f1_train, 5) if f1_train is not None else None
            f1_test = round(f1_test, 5) if f1_test is not None else None

            # Save model
            model_path = os.path.join(request_body.path_to_save, f"{class_name}.pkl")
            joblib.dump(best_estimator, model_path, protocol=5)

            write_log(
                f"trained {class_name}: train_acc={accuracy_train}, test_acc={accuracy_test}, "
                f"train_f1={f1_train}, test_f1={f1_test} - saved to {model_path}",
                "info", os.path.basename(__file__)
            )

            # Store performance
            request_body.performance[class_name] = {
                "model_path": model_path,
                "accuracy_train": accuracy_train,
                "accuracy_test": accuracy_test,
                "f1_train": f1_train,
                "f1_test": f1_test,
                "train_metrics": metrics['train_metrics'],
                "test_metrics": metrics['test_metrics'],
                "best_params": best_params,
            }
            if cv_result:
                request_body.performance[class_name]["cv"] = cv_result

            class_labels = labels_in_code_order(request_body.target_encodings)
            parts = classification_parts(best_estimator, X_test, request_body.y_test, class_labels)
            parts["top_terms"] = lambda: nlp_top_terms(best_estimator, request_body.vectorizer, class_labels)
            request_body.charts[class_name] = build_model_charts(parts)

            emit_progress(f"✅ Finished NLP model: {class_name} (test_acc={accuracy_test})", pct=model_base_pct + int(50/max(1, total_models)))

        labels = labels_in_code_order(request_body.target_encodings)
        write_evaluation(request_body.path_to_save, request_body.task, request_body.charts,
                         {"n_test": len(request_body.y_test), "classes": [str(c) for c in labels or []]})

        # Save run config and column data
        config_path = os.path.join(request_body.path_to_save, "config_json.json")
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(request_body.jsonify(), handle, cls=NumpyEncoder)

        column_data = {
            "text_column": request_body.text_column,
            "target_column": request_body.target_column,
            "task_type": "nlp_classification",
        }
        column_data_path = os.path.join(request_body.path_to_save, "column_data.json")
        with open(column_data_path, "w", encoding="utf-8") as handle:
            json.dump(column_data, handle, cls=NumpyEncoder)
        if request_body.test_rows is not None:
            write_holdout(request_body.path_to_save, request_body.test_rows, request_body.source_rows)

        write_log("nlp_model_builder_controller completed", "info", os.path.basename(__file__))

    except InterruptedError:
        raise
    except Exception as e:
        if request_body.status_code == 200:
            request_body.error_message = "NLPModelTrainingError"
            request_body.status_code = NLP_ERROR_CODES[request_body.error_message]
            request_body.error_traceback = traceback.format_exc()
            write_log(f"error occurred - {request_body.error_message} - {request_body.error_traceback}",
                      "error", os.path.basename(__file__))
        raise e
