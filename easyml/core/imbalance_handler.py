"""
Class-imbalance handling (SMOTE, Tomek links, SMOTETomek) for tabular and NLP classification.

Resampling is a step of an ``imblearn`` pipeline wrapped around the model, so inside a grid search
it runs on the training folds only: synthetic samples never reach a validation fold, and the saved
model is evaluated (train and test metrics) on the original, un-resampled data.
"""

import os

import numpy as np

from easyml.utils.logger import write_log

RESAMPLING_METHODS = ("smote", "tomek", "smotetomek")
MODEL_STEP = "model"


def _resample(X, y, method):
    """Resample one training set; falls back to the original data when a class is too small."""
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import TomekLinks

    if method == "tomek":
        sampler = TomekLinks()
    else:
        # SMOTE interpolates between a sample and its neighbours of the same class, so the smallest
        # class needs at least two samples, and k_neighbors may not exceed its size minus one.
        smallest = int(np.unique(y, return_counts=True)[1].min())
        if smallest < 2:
            write_log(f"{method}: smallest class has {smallest} sample(s); training on the original data",
                      "warning", os.path.basename(__file__))
            return X, y
        smote = SMOTE(random_state=42, k_neighbors=min(5, smallest - 1))
        sampler = smote if method == "smote" else SMOTETomek(random_state=42, smote=smote)

    try:
        return sampler.fit_resample(X, y)
    except ValueError as exc:
        write_log(f"{method} resampling failed ({exc}); training on the original data", "warning", os.path.basename(__file__))
        return X, y


def make_resampler(method):
    """Return an imblearn sampler for ``method``, or None when no resampling is requested."""
    method = (method or "none").lower()
    if method == "none":
        return None
    if method not in RESAMPLING_METHODS:
        raise ValueError(f"Unknown imbalance handling method '{method}'. Use one of: none, {', '.join(RESAMPLING_METHODS)}.")
    try:
        from imblearn import FunctionSampler
    except ImportError:
        write_log("imbalanced-learn is not installed; skipping imbalance handling.", "error", os.path.basename(__file__))
        return None
    return FunctionSampler(func=_resample, validate=False, kw_args={"method": method})


def with_resampling(estimator, param_grid, method):
    """
    Wrap ``estimator`` so it resamples its training data before fitting.

    Returns ``(estimator, param_grid)``: unchanged when no resampling is requested, otherwise an
    imblearn ``Pipeline`` and the grid re-keyed to the pipeline's model step.
    """
    resampler = make_resampler(method)
    if resampler is None:
        return estimator, param_grid
    from imblearn.pipeline import Pipeline

    pipeline = Pipeline([("resample", resampler), (MODEL_STEP, estimator)])
    return pipeline, {f"{MODEL_STEP}__{key}": values for key, values in (param_grid or {}).items()}


def unwrap_model(fitted, best_params=None):
    """Return the fitted model (without the resampling step) and its best params with plain names."""
    prefix = f"{MODEL_STEP}__"
    params = {key[len(prefix):] if key.startswith(prefix) else key: value for key, value in (best_params or {}).items()}
    named_steps = getattr(fitted, "named_steps", None)
    if named_steps and MODEL_STEP in named_steps:
        return named_steps[MODEL_STEP], params
    return fitted, params
