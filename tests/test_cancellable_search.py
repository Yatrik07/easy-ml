"""A stop request reaches inside a hyperparameter search, and the search still matches GridSearchCV."""

import threading
import time

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from easyml.core.cancellable_search import CancellableGridSearchCV
from easyml.core.imbalance_handler import with_resampling
from easyml.utils.cancellation import TrainingJob, reset_current_job, set_current_job

RF_GRID = {"n_estimators": [5, 10], "max_depth": [2, 4, None], "min_samples_leaf": [1, 3]}


class _SlowClassifier(ClassifierMixin, BaseEstimator):
    """Takes ``delay`` seconds to fit, so a search over it runs long enough to be stopped."""

    def __init__(self, delay=0.3, c=0):
        self.delay = delay
        self.c = c

    def fit(self, X, y):
        time.sleep(self.delay)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


def _assert_same_search(ours, reference):
    assert ours.best_params_ == reference.best_params_
    assert ours.best_score_ == reference.best_score_
    assert ours.best_index_ == reference.best_index_
    assert ours.cv_results_["params"] == reference.cv_results_["params"]
    for key in ("mean_test_score", "std_test_score", "rank_test_score"):
        np.testing.assert_array_equal(ours.cv_results_[key], reference.cv_results_[key])


@pytest.mark.parametrize("n_jobs", [1, 2, 8])
def test_results_match_grid_search(n_jobs):
    X, y = make_classification(n_samples=200, n_features=6, random_state=0)
    estimator = RandomForestClassifier(random_state=0)
    reference = GridSearchCV(estimator, RF_GRID, cv=4, scoring="accuracy", n_jobs=n_jobs).fit(X, y)
    ours = CancellableGridSearchCV(estimator, RF_GRID, cv=4, scoring="accuracy", n_jobs=n_jobs).fit(X, y)

    _assert_same_search(ours, reference)
    # Refit on the full training data with the winning parameters.
    np.testing.assert_array_equal(ours.best_estimator_.predict(X), reference.best_estimator_.predict(X))


def test_results_match_grid_search_with_resampling():
    pytest.importorskip("imblearn")
    X, y = make_classification(n_samples=200, n_features=6, weights=[0.85], random_state=1)
    estimator, grid = with_resampling(RandomForestClassifier(random_state=0), RF_GRID, "smote")
    reference = GridSearchCV(estimator, grid, cv=4, scoring="accuracy", n_jobs=2).fit(X, y)
    ours = CancellableGridSearchCV(estimator, grid, cv=4, scoring="accuracy", n_jobs=2).fit(X, y)

    _assert_same_search(ours, reference)
    np.testing.assert_array_equal(ours.best_estimator_.predict(X), reference.best_estimator_.predict(X))


def test_cancelling_the_job_stops_a_running_search():
    X, y = make_classification(n_samples=80, n_features=4, random_state=0)
    # 30 candidates x 4 folds x 0.3 s on two workers: about 18 s if the search ran to the end.
    search = CancellableGridSearchCV(_SlowClassifier(), {"c": list(range(30))}, cv=4, n_jobs=2)
    job = TrainingJob("slow search")
    outcome = {}

    def run():
        token = set_current_job(job)
        try:
            search.fit(X, y)
            outcome["result"] = "finished"
        except InterruptedError as exc:
            outcome["result"] = exc
        finally:
            reset_current_job(token)
            outcome["stopped_at"] = time.monotonic()

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    time.sleep(2)
    assert worker.is_alive(), "the search finished before it could be cancelled"
    job.cancel()
    cancelled_at = time.monotonic()
    worker.join(timeout=30)

    assert not worker.is_alive()
    assert isinstance(outcome["result"], InterruptedError)
    assert outcome["stopped_at"] - cancelled_at < 3
    assert not hasattr(search, "best_estimator_")


def test_search_outside_a_job_is_never_cancelled():
    X, y = make_classification(n_samples=80, n_features=4, random_state=0)
    search = CancellableGridSearchCV(_SlowClassifier(delay=0), {"c": [0, 1, 2]}, cv=4, n_jobs=1).fit(X, y)
    assert len(search.cv_results_["params"]) == 3
