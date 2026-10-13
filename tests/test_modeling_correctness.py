"""Resampling stays inside the training folds; preprocessing scales before PCA and caps one-hot width."""

import os

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from easyml.core.imbalance_handler import unwrap_model, with_resampling
from easyml.core.pre_processing import MAX_ONEHOT_CATEGORIES
from easyml.route_helpers import sanitize_run_name
from easyml.trainer import MLTrainer


class _RecordingClassifier(ClassifierMixin, BaseEstimator):
    """Remembers the class counts of every training set it is fitted on."""

    fitted_class_counts = []

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y):
        type(self).fitted_class_counts.append(tuple(np.bincount(y)))
        self.model_ = LogisticRegression(C=self.C).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)


def _imbalanced(n_major=90, n_minor=10, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(0, 1, (n_major, 3)), rng.normal(2, 1, (n_minor, 3))])
    y = np.array([0] * n_major + [1] * n_minor)
    return X, y


def test_resampling_only_touches_training_folds():
    X, y = _imbalanced()
    scored_fold_sizes = []

    def scorer(estimator, X_val, y_val):
        scored_fold_sizes.append((len(y_val), int(np.sum(y_val == 1))))
        return float(np.mean(estimator.predict(X_val) == y_val))

    _RecordingClassifier.fitted_class_counts = []
    estimator, grid = with_resampling(_RecordingClassifier(), {"C": [0.1, 1.0]}, "smote")
    assert grid == {"model__C": [0.1, 1.0]}
    search = GridSearchCV(estimator, grid, cv=4, scoring=scorer, n_jobs=1).fit(X, y)

    # Validation folds are original data: 100 samples / 4 folds, 10 minority samples spread over them.
    assert scored_fold_sizes and all(size == (25, 2) or size == (25, 3) for size in scored_fold_sizes)
    # Every fit (folds and final refit) saw a balanced, oversampled training set.
    assert all(counts[0] == counts[1] for counts in _RecordingClassifier.fitted_class_counts)

    model, params = unwrap_model(search.best_estimator_, search.best_params_)
    assert isinstance(model, _RecordingClassifier)
    assert set(params) == {"C"}


def test_no_resampling_leaves_estimator_alone():
    model = LogisticRegression()
    assert with_resampling(model, {"C": [1]}, "none") == (model, {"C": [1]})
    assert unwrap_model(model, {"C": 1}) == (model, {"C": 1})


def test_smote_falls_back_when_a_class_has_one_sample():
    X, y = _imbalanced(n_major=20, n_minor=1)
    estimator, _ = with_resampling(LogisticRegression(), {}, "smote")
    estimator.fit(X, y)  # must not raise


def _train(data_dir, **kwargs):
    name = kwargs.pop("name")
    path = os.path.join(data_dir, "runs", sanitize_run_name(name))
    os.makedirs(path, exist_ok=True)
    return MLTrainer(name=name, path_to_save=path, timestamp="2026-01-01 00-00-00", **kwargs).run()


@pytest.fixture(scope="module")
def imbalanced_csv(data_dir):
    X, y = _imbalanced(n_major=120, n_minor=20, seed=1)
    df = pd.DataFrame(X, columns=["a", "b", "c"]).assign(label=np.where(y == 1, "rare", "common"))
    path = os.path.join(data_dir, "uploads", "imbalanced.csv")
    df.to_csv(path, index=False)
    return path


@pytest.mark.parametrize("method", ["smote", "tomek", "smotetomek"])
def test_tabular_train_metrics_use_original_training_data(client, data_dir, imbalanced_csv, method):
    run = _train(data_dir, name=f"imb_{method}", task="MLClassification", dataset_path=imbalanced_csv,
                 target_column="label", models=["logistic_regression", "random_forest"],
                 hyper_parameter_tuning=1, imbalance_handling=method)
    n_train = len(run.y_train)
    assert n_train == 112  # 80% of 140, not the resampled size
    for name, perf in run.performance.items():
        assert np.asarray(perf["train_metrics"]["confusion_matrix"]).sum() == n_train
        assert not any(key.startswith("model__") for key in perf["best_params"])
        saved = joblib.load(perf["model_path"])
        assert type(saved).__name__ == name  # the bare model is saved, not the resampling pipeline

    response = client.post("/inference", json={"run_id": run.run_id, "model_name": "LogisticRegression",
                                               "data": {"a": "2.5", "b": "2", "c": "2"}})
    assert response.status_code == 200, response.get_json()


def test_nlp_train_metrics_use_original_training_data(data_dir):
    rows = ([{"text": "the team won the football match", "label": "sports"}] * 30
            + [{"text": "the new laptop has a fast processor", "label": "tech"}] * 8)
    path = os.path.join(data_dir, "uploads", "imbalanced_text.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    run = _train(data_dir, name="nlp_imb", task="NLPTextClassification", dataset_path=path, target_column="label",
                 text_column="text", models=["logistic_regression"], min_df=1, hyper_parameter_tuning=1,
                 imbalance_handling="smote")
    perf = run.performance["LogisticRegression"]
    assert np.asarray(perf["train_metrics"]["confusion_matrix"]).sum() == len(run.y_train)
    assert type(joblib.load(perf["model_path"])).__name__ == "LogisticRegression"


def test_pca_runs_on_standardised_features_even_without_scaling(client, data_dir, regression_csv):
    run = _train(data_dir, name="pca_unscaled", task="MLRegression", dataset_path=regression_csv, target_column="y",
                 models=["linear_regression"], scaling=False, reduce_features=True)
    pipeline = joblib.load(os.path.join(run.path_to_save, "preprocessing_pipeline.pkl"))
    assert list(pipeline.named_steps) == ["column_transformer", "pca_scaler", "pca"]

    scaled = _train(data_dir, name="pca_scaled", task="MLRegression", dataset_path=regression_csv, target_column="y",
                    models=["linear_regression"], scaling=True, reduce_features=True)
    pipeline = joblib.load(os.path.join(scaled.path_to_save, "preprocessing_pipeline.pkl"))
    assert list(pipeline.named_steps) == ["column_transformer", "pca"]

    response = client.post("/inference", json={"run_id": run.run_id, "model_name": "LinearRegression",
                                               "data": {"x1": "1.0", "x2": "b"}})
    assert response.status_code == 200, response.get_json()


def test_high_cardinality_columns_are_capped(client, data_dir):
    rng = np.random.default_rng(3)
    n = 400
    df = pd.DataFrame({
        "customer_code": [f"C{i:04d}" for i in range(n)],  # every value unique
        "city": rng.choice(["NY", "LA", "SF"], n),
        "spend": rng.normal(100, 20, n),
    })
    df["label"] = np.where(df["spend"] > 100, "high", "low")
    path = os.path.join(data_dir, "uploads", "high_cardinality.csv")
    df.to_csv(path, index=False)

    run = _train(data_dir, name="high_card", task="MLClassification", dataset_path=path, target_column="label",
                 models=["logistic_regression"])
    pipeline = joblib.load(os.path.join(run.path_to_save, "preprocessing_pipeline.pkl"))
    width = pipeline.transform(df.drop(columns=["label"])).shape[1]
    assert width <= MAX_ONEHOT_CATEGORIES + 3 + 1  # capped code column + 3 cities + spend

    response = client.post("/inference", json={"run_id": run.run_id, "model_name": "LogisticRegression",
                                               "data": {"customer_code": "NEVER-SEEN", "city": "NY", "spend": "150"}})
    assert response.status_code == 200, response.get_json()
    assert response.get_json()["prediction"] in {"high", "low"}
