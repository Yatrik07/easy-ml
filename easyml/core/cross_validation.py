"""
k-fold cross-validation of one model on a run's training data (optional, ``cv_folds``).

The held-out test set stays untouched and keeps ranking the models; cross-validation adds how stable a
model's score is across different parts of the training data. Each fold fits a fresh copy of the whole
estimator (preprocessing or vectorizer, resampling and model), so nothing from a validation fold leaks
into the model that is scored on it.
"""

import numpy as np

from easyml.utils.cancellation import check_cancelled


def _folds(y, k, seed, classification):
    from sklearn.model_selection import KFold, StratifiedKFold

    if classification:
        counts = np.unique(np.asarray(y), return_counts=True)[1]
        if counts.min() >= k:
            return StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return KFold(n_splits=k, shuffle=True, random_state=seed)


def _take(data, index):
    return data.iloc[index] if hasattr(data, "iloc") else data[index]


def cross_validate(make_estimator, X, y, k, seed, classification, on_fold=None):
    """
    Score ``make_estimator()`` (a fresh, unfitted estimator) on ``k`` folds of ``(X, y)``.

    Returns ``{"folds", "metric", "mean", "std", "scores"}``; on a failure ``{"folds", "metric", "error"}``
    (the run itself still succeeds). A stop request (``InterruptedError``) is raised between folds.
    """
    from sklearn.metrics import accuracy_score, r2_score

    metric = "accuracy" if classification else "r2"
    y = np.asarray(y)
    if len(y) < 2 * k:
        return {"folds": k, "metric": metric, "error": f"Too few training rows ({len(y)}) for {k} folds."}
    scores = []
    try:
        for number, (train, valid) in enumerate(_folds(y, k, seed, classification).split(np.zeros(len(y)), y), 1):
            check_cancelled()
            estimator = make_estimator()
            estimator.fit(_take(X, train), y[train])
            predicted = np.asarray(estimator.predict(_take(X, valid))).reshape(-1)
            score = accuracy_score(y[valid], predicted) if classification else r2_score(y[valid], predicted)
            scores.append(round(float(score), 5))
            if on_fold:
                on_fold(number, k)
    except InterruptedError:
        raise
    except Exception as exc:  # e.g. a class missing from one fold: report it, keep the run
        return {"folds": k, "metric": metric, "error": f"Cross-validation failed: {exc}"}
    return {"folds": k, "metric": metric, "mean": round(float(np.mean(scores)), 5),
            "std": round(float(np.std(scores)), 5), "scores": scores}
