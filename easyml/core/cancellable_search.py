"""
Grid search that a training job can stop part-way through.

``GridSearchCV`` dispatches every candidate x fold fit to its worker processes in one go, so a stop
request would only be noticed once the whole search had finished. The worker processes cannot see
the job's cancel event, so this search hands the candidates to scikit-learn in batches of about one
round of workers and checks for cancellation in the calling (job) thread between batches.

scikit-learn accumulates the results of every batch, so ``cv_results_``, ``best_params_`` and the
refit on the full training data are the same as a plain ``GridSearchCV``.
"""

from joblib import effective_n_jobs
from sklearn.model_selection import GridSearchCV, ParameterGrid, check_cv

from easyml.utils.cancellation import check_cancelled


class CancellableGridSearchCV(GridSearchCV):
    """``GridSearchCV`` that raises ``InterruptedError`` between batches once its job is cancelled."""

    def _batch_size(self):
        try:
            n_splits = check_cv(self.cv).get_n_splits()
        except (TypeError, ValueError):
            n_splits = 1
        return max(1, effective_n_jobs(self.n_jobs) // max(1, n_splits))

    def _run_search(self, evaluate_candidates):
        candidates = list(ParameterGrid(self.param_grid))
        batch_size = self._batch_size()
        for start in range(0, len(candidates), batch_size):
            check_cancelled()
            evaluate_candidates(candidates[start:start + batch_size])
        # Don't spend a refit on a run that was stopped during the last batch.
        check_cancelled()
