"""
The random seed of a run: the same data, settings and seed always give the same models and scores.

It drives the train/test split, resampling (SMOTE), cross-validation folds and every model that has a
seed of its own (random forests, AdaBoost, MLP, XGBoost, LightGBM, CatBoost; for images, PyTorch).
"""

from easyml.config.constants import CATBOOST_MODEL_NAMES

DEFAULT_SEED = 42
MAX_SEED = 2 ** 31 - 1


def validate_seed(value):
    """A seed as an int between 0 and 2**31 - 1 (default 42 when empty)."""
    if value is None or value == "":
        return DEFAULT_SEED
    try:
        seed = int(value)
    except (TypeError, ValueError):
        raise ValueError("The random seed must be a whole number.") from None
    if isinstance(value, float) and value != seed:
        raise ValueError("The random seed must be a whole number.")
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"The random seed must be between 0 and {MAX_SEED}.")
    return seed


MAX_CV_FOLDS = 20


def validate_cv_folds(value):
    """0 (off), or between 2 and 20 folds."""
    if value is None or value == "" or value is False:
        return 0
    try:
        folds = int(value)
    except (TypeError, ValueError):
        raise ValueError("Cross-validation folds must be a whole number (0 turns it off).") from None
    if folds != 0 and not 2 <= folds <= MAX_CV_FOLDS:
        raise ValueError(f"Cross-validation needs between 2 and {MAX_CV_FOLDS} folds (0 turns it off).")
    return folds


def apply_seed(model, seed):
    """Give ``model`` the run's seed when it has one (``random_state``, or CatBoost's ``random_seed``)."""
    if type(model).__name__ in CATBOOST_MODEL_NAMES:  # CatBoost's get_params() only lists explicit settings
        model.set_params(random_seed=seed)
    elif "random_state" in model.get_params(deep=False):
        model.set_params(random_state=seed)
    return model
