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


# --------------------------------------------------------------------------- notes per target (0.14.2)

NOT_PREDICTABLE_R2 = 0.1      # regression: below this best test R², the models predict about the column's average
NOT_PREDICTABLE_MARGIN = 0.02  # classification: no better than always answering the most frequent class
OVERFIT_GAP = 0.3             # train score this much above the test score
RELATED_CORR = 0.3            # |r| between a column's test errors and another target's values


def _fmt(value, task):
    return f"{value:.0%}" if task == "classification" else f"{value:.2f}"


def target_notes(performance, targets, task, y_test=None, test_predictions=None):
    """
    What to tell the user about each column to predict, from the scores of every model (and, for numbers, how
    the best model's test errors relate to the other targets): ``{target: [{"kind", "text", ...}]}``, only
    targets with notes.

    * ``not_predictable``: even the best model is about as good as predicting the column's average (regression,
      best test R² < 0.1) or its most frequent class (classification). Weights cannot change this.
    * ``overfits``: the best model scores far higher on its training rows than on the test rows.
    * ``depends_on_target``: the best model's test errors follow another column that is also predicted (|r| ≥
      0.3), so that column would explain part of what the models miss; but a target cannot be an input.
    """
    notes = {}
    scores = target_scores(performance, targets)
    for index, target in enumerate(targets):
        found = []
        ranked = scores.get(target) or []
        if ranked:
            model, best = ranked[0]
            entry = (performance.get(model) or {})
            per_target = (entry.get("targets") or {}).get(target) or {}
            metrics = ((entry.get("test_metrics") or {}).get("targets") or {}).get(target) or {}
            baseline = metrics.get("baseline")
            if task == "classification":
                weak = baseline is not None and best <= baseline + NOT_PREDICTABLE_MARGIN
                why = (f"the best model is right {_fmt(best, task)} of the time, about as often as always answering "
                       f"its most frequent value ({_fmt(baseline, task) if baseline is not None else '?'})")
            else:
                weak = best < NOT_PREDICTABLE_R2
                spread = metrics.get("spread")
                why = (f"the models predict close to its average (best R² {best:.2f}"
                       + (f", predictions vary {spread:.0%} as much as the real values" if spread is not None else "")
                       + ")")
            if weak:
                found.append({"kind": "not_predictable", "model": model, "best": best,
                              "text": f"{target} cannot be predicted from these inputs: {why}. Weights cannot change "
                                      "this; add input columns that relate to it."})
            train = per_target.get("accuracy_train")
            if train is not None and train - best > OVERFIT_GAP:
                found.append({"kind": "overfits", "model": model, "train": train, "test": best,
                              "text": f"{target}: the best model scores {_fmt(train, task)} on its training rows but "
                                      f"{_fmt(best, task)} on the test rows, so it mostly memorized the training data."})
        if ranked and task != "classification" and y_test is not None and ranked[0][0] in (test_predictions or {}):
            related = _strongest_partner(y_test, test_predictions[ranked[0][0]], index, targets)
            if related:
                other, corr = related
                found.append({"kind": "depends_on_target", "other": other, "corr": corr,
                              "text": f"{other} would explain about {corr * corr:.0%} of what the models get wrong for "
                                      f"{target} (r {corr:.2f}), but it is also predicted here, so it cannot be used as "
                                      f"an input. If you will know {other} when predicting, train {target} in its own "
                                      f"run with {other} as an input."})
        if found:
            notes[target] = found
    return notes


def _strongest_partner(y_test, predicted, index, targets):
    """
    The other target whose true values follow target ``index``'s test errors most (|r| ≥ 0.3), as
    ``(name, r)``, or None: that target holds information the inputs lack.
    """
    truth = np.asarray(y_test, dtype=float)
    errors = truth[:, index] - np.asarray(predicted, dtype=float)[:, index]
    best = None
    for other, column in enumerate(targets):
        if other == index or np.std(errors) == 0 or np.std(truth[:, other]) == 0:
            continue
        r = float(np.corrcoef(errors, truth[:, other])[0, 1])
        if abs(r) >= RELATED_CORR and (best is None or abs(r) > abs(best[1])):
            best = (column, round(r, 2))
    return best


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
