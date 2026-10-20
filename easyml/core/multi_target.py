"""
Several target columns in one tabular run (Easy-ML 0.13).

Every target gets its own model, trained (and tuned, when tuning is on) on that target's own error, so the
accuracy of one target is never traded for another's. The target weights only decide how the targets'
scores add up to the run's score, which ranks the models. The "best model for each target" predicts every
target with the model that did best on it.

A run with one target never comes here: its files and answers are exactly those of earlier versions.
"""

import math
import os

import numpy as np

BEST_FOR_EACH_TARGET = "best_for_each_target"
MIN_WEIGHT, MAX_WEIGHT = 0.1, 10.0


def targets_of(source):
    """The target columns of a run: from its config (a dict) or a request body. One target: ``[name]``."""
    if source is None:
        return []
    if isinstance(source, dict):
        columns, single = source.get("target_columns"), source.get("target_column")
    else:
        columns, single = getattr(source, "target_columns", None), getattr(source, "target_column", None)
    if columns:
        return [str(c) for c in columns]
    return [single] if single else []


def is_multi(source):
    return len(targets_of(source)) > 1


def clean_targets(target_column, target_columns=None):
    """
    ``(first target, all targets)`` from what a caller passed: a column name, or a list of them (as
    ``target_column`` or ``target_columns``). ``all targets`` is empty for a run with a single target.
    """
    chosen = target_columns if target_columns else target_column
    if isinstance(chosen, (list, tuple)):
        columns = [c for c in chosen if c is not None and str(c) != ""]
    else:
        columns = [chosen] if chosen else []
    if len(set(columns)) != len(columns):
        raise ValueError("Each column to predict can only be chosen once.")
    return (columns[0] if columns else ""), (list(columns) if len(columns) > 1 else [])


def clean_weights(weights, targets):
    """``{target: weight}`` for every target (1 when not given); ValueError for anything else."""
    weights = dict(weights or {})
    unknown = [str(k) for k in weights if k not in targets]
    if unknown:
        raise ValueError(f"Target weights are only for the columns to predict, not {', '.join(unknown)}.")
    cleaned = {}
    for target in targets:
        value = weights.get(target, 1)
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"The weight of '{target}' must be a number.")
        if not math.isfinite(value) or not MIN_WEIGHT <= value <= MAX_WEIGHT:
            raise ValueError(f"The weight of '{target}' must be between {MIN_WEIGHT:g} and {MAX_WEIGHT:g}.")
        cleaned[target] = value
    return cleaned


def weights_of(source):
    """The target weights in target order (1 when none were set)."""
    targets = targets_of(source)
    stored = (source.get("target_weights") if isinstance(source, dict)
              else getattr(source, "target_weights", None)) or {}
    return [float(stored.get(t, 1)) for t in targets]


def weighted_mean(values, weights):
    """Mean of the values that are not None, each counted ``weight`` times (None when there are none)."""
    pairs = [(float(v), float(w)) for v, w in zip(values, weights) if v is not None]
    total = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / total if total else None


class PerTargetModel:
    """
    One fitted model per target column, used together: ``predict`` returns one column per target.
    ``estimators_[i]`` is the model of ``targets[i]``, an ordinary single-target model.
    """

    def __init__(self, estimators, targets, model_names=None):
        self.estimators_ = list(estimators)
        self.targets = list(targets)
        self.model_names = list(model_names) if model_names else [type(e).__name__ for e in self.estimators_]

    def predict(self, X):
        return np.column_stack([np.asarray(e.predict(X)).reshape(-1) for e in self.estimators_])

    def __repr__(self):
        return f"PerTargetModel({', '.join(f'{t}: {n}' for t, n in zip(self.targets, self.model_names))})"


def estimator_for(model, index):
    """The single-target model of target number ``index``."""
    return model.estimators_[index]


# --------------------------------------------------------------------------- scores

def target_scores(performance, targets):
    """``{target: [(model, test score), ...] best first}`` from a multi-target run's metrics."""
    ranked = {}
    for target in targets:
        scores = []
        for model, values in (performance or {}).items():
            if model == BEST_FOR_EACH_TARGET or not isinstance(values, dict):
                continue
            score = ((values.get("targets") or {}).get(target) or {}).get("accuracy_test")
            if score is not None:
                scores.append((model, float(score)))
        ranked[target] = sorted(scores, key=lambda item: -item[1])  # stable: ties keep training order
    return ranked


def winners(performance, targets):
    """``{target: model}``: the model with the best test score on each target."""
    return {t: s[0][0] for t, s in target_scores(performance, targets).items() if s}


# --------------------------------------------------------------------------- the combined model

def best_for_each_target(folder, task):
    """
    The run's "best model for each target": a :class:`PerTargetModel` that predicts every target with the
    saved model that scored best on it. Built from the saved models; nothing is trained.
    """
    from easyml.core.artifacts import load_tabular_model, read_json

    config = read_json(folder, "config_json.json")
    targets = targets_of(config)
    chosen = winners(config.get("performance") or {}, targets)
    if len(targets) < 2 or len(chosen) != len(targets):
        raise FileNotFoundError("This run has no best model for each target.")
    estimators, names = [], []
    for index, target in enumerate(targets):
        saved = load_tabular_model(folder, chosen[target], task)
        estimators.append(estimator_for(saved, index))
        names.append(chosen[target])
    return PerTargetModel(estimators, targets, names)


def _config(folder):
    from easyml.core.artifacts import read_json

    return read_json(folder, "config_json.json") if folder and os.path.isdir(folder) else {}


def run_targets(folder):
    """The target columns of a run with several of them (``[]`` for any other run)."""
    targets = targets_of(_config(folder))
    return targets if len(targets) > 1 else []


def combined_summary(folder):
    """``{target: model}`` behind the best model for each target, or None for other runs."""
    config = _config(folder)
    targets = targets_of(config)
    if len(targets) < 2:
        return None
    chosen = winners(config.get("performance") or {}, targets)
    return chosen if len(chosen) == len(targets) else None
