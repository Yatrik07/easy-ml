"""
Input column weights (Easy-ML 0.14): how much the models rely on each input column.

A weight is a hint from what you know about the data: 2 asks the models to lean on a column more, 0.5 less.
Models are scored on the columns they predict, never on their inputs, so a weight changes which inputs a
model uses, not how its error is measured, and it can lower the test score.

How each model follows it (``weight_effect``):

* **KNN**: the column's processed values are multiplied by its weight (a step in the run's preprocessing
  pipeline), so it counts more in distances.
* **LightGBM** (``feature_contri``) multiplies the gain of splits on the column.
* **CatBoost** (``feature_weights``, the same idea) and **XGBoost** (``feature_weights``: the column is sampled
  more or less often when columns are picked for a split, ``colsample_bynode`` 0.8 in weighted runs) get the
  weights too, but they barely change what the model relies on.
* **Logistic regression** sees the multiplied values too, but that only changes how much its regularisation
  holds the column back, which is slight unless the data is small.

Measured by permutation (how much the predictions depend on the column) on two noisy copies of one signal,
with the first copy weighted 1 / 2 / 4 / 10, the share of reliance on it was:

* KNN 0.50 / 0.72 / 0.85 / 0.92 and LightGBM 0.40 / 0.77 / 0.78 / 0.79: they follow;
* CatBoost 0.54 / 0.55 / 0.54 / 0.56 and XGBoost 0.60 / 0.58 / 0.58 / 0.60: they barely change.

When the columns each carry different information, any good model needs all of them, so weights matter
little for every model.
* **Random Forest, AdaBoost and linear regression** have no such setting and are not steered by weights:
  they only follow "Leave out". (Trees can still shift slightly, because scaling a column changes float
  rounding at split points: measured on Housing.csv, R² 0.60393 vs 0.6036.)

A run without weights is built exactly as before.
"""

from functools import partial

import numpy as np

MIN_WEIGHT, MAX_WEIGHT = 0.1, 10.0
BOOSTERS = {"LGBMClassifier", "LGBMRegressor", "CatBoostClassifier", "CatBoostRegressor",
            "XGBClassifier", "XGBRegressor"}
FOLLOWS_WEIGHTS = {"LGBMClassifier", "LGBMRegressor", "KNeighborsClassifier", "KNeighborsRegressor"}
SLIGHT = {"LogisticRegression", "CatBoostClassifier", "CatBoostRegressor", "XGBClassifier", "XGBRegressor"}
XGB_COLUMN_SAMPLE = 0.8
# The same, by the model keys the studio and MLTrainer use (for the UI before training).
WEIGHT_EFFECT_BY_KEY = {"light_gbm": "follows", "cat_boost": "slight", "xg_boost": "slight", "knn": "follows",
                        "logistic_regression": "slight", "random_forest": "none", "ada_boost": "none",
                        "linear_regression": "none"}


def clean_feature_weights(weights):
    """``{column: weight}`` with only the weights other than 1; ValueError for anything out of range."""
    import math

    cleaned = {}
    for column, value in dict(weights or {}).items():
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"The weight of '{column}' must be a number.")
        if not math.isfinite(value) or not MIN_WEIGHT <= value <= MAX_WEIGHT:
            raise ValueError(f"The weight of '{column}' must be between {MIN_WEIGHT:g} and {MAX_WEIGHT:g} "
                             "(leave a column out to drop it).")
        if value != 1:
            cleaned[str(column)] = value
    return cleaned


def weight_effect(model_name):
    """``"follows"``, ``"slight"`` or ``"none"``: how much a model (class name) follows column weights."""
    if model_name in FOLLOWS_WEIGHTS:
        return "follows"
    return "slight" if model_name in SLIGHT else "none"


def follows_weights(model_name):
    return weight_effect(model_name) == "follows"


def weight_effects_by_class():
    """``{model class: effect}`` for every tabular model class."""
    names = ["LinearRegression", "LogisticRegression", "RandomForestClassifier", "RandomForestRegressor",
             "AdaBoostClassifier", "AdaBoostRegressor"] + sorted(FOLLOWS_WEIGHTS | SLIGHT)
    return {name: weight_effect(name) for name in names}


def weighted_transformers(transformers, weights):
    """
    The ColumnTransformer entries with one entry per column, in the same order, and a "multiply by the
    weight" step after the column's own steps when its weight is not 1. One column per entry gives the same
    output as the grouped entries (imputers, encoders and scalers work column by column), so unweighted
    columns are unchanged and every output of a one-hot encoded column gets its weight.
    """
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    entries = []
    for name, pipeline, columns in transformers:
        for index, column in enumerate(columns):
            steps = list(clone(pipeline).steps)
            weight = weights.get(column, 1)
            if weight != 1:
                steps.append(("weight", FunctionTransformer(partial(np.multiply, float(weight)),
                                                            feature_names_out="one-to-one")))
            entries.append((f"{name}_{index}", Pipeline(steps), [column]))
    return entries


def output_weights(pipeline, raw_columns, weights):
    """The weight of every column the fitted preprocessing outputs (one-hot columns share their column's)."""
    from easyml.core.evaluation_charts import _output_columns

    owners = _output_columns(pipeline, list(raw_columns))
    if owners is None:
        raise ValueError("Column weights could not be matched to the processed columns.")
    return [float(weights.get(owner, 1)) for owner in owners]


def booster_params(model_name, vector):
    """The settings that make a gradient-boosting model follow ``vector`` (one weight per processed column)."""
    if model_name.startswith("LGBM"):
        return {"feature_contri": list(vector)}
    if model_name.startswith("CatBoost"):
        # "index:weight" text, the same as a list for CatBoost: newer versions copy a list or dict they are given,
        # which scikit-learn's clone (grid search, cross-validation) refuses.
        return {"feature_weights": ",".join(f"{i}:{w:g}" for i, w in enumerate(vector) if w != 1) or None}
    if model_name.startswith("XGB"):
        return {"feature_weights": np.asarray(vector, dtype=float), "colsample_bynode": XGB_COLUMN_SAMPLE}
    return {}


class FoldModel:
    """
    Cross-validation of a booster in a weighted run: the preprocessing is refit in each fold, which can
    change the number of processed columns (one-hot categories), so the weights are matched again per fold.
    """

    def __init__(self, prep, model, model_name, raw_columns, weights):
        self.prep, self.model, self.model_name = prep, model, model_name
        self.raw_columns, self.weights = raw_columns, weights

    def fit(self, X, y):
        transformed = self.prep.fit_transform(X)
        final = self.model.steps[-1][1] if hasattr(self.model, "steps") else self.model
        final.set_params(**booster_params(self.model_name, output_weights(self.prep, self.raw_columns, self.weights)))
        self.model.fit(transformed, y)
        return self

    def predict(self, X):
        return self.model.predict(self.prep.transform(X))
