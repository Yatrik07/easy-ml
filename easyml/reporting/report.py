"""
A run as one self-contained HTML file to send to others: summary, settings, leaderboard, per-model
metrics and charts. Inline CSS and SVG only (no scripts, no external files or links), prints well.
"""

import os
from datetime import datetime
from html import escape

from easyml.reporting import svg

REGRESSION_LABELS = {"r2_score": "R²", "mae": "MAE", "mse": "MSE", "rmse": "RMSE", "mape": "MAPE"}

CSS = """
:root { --text:#18181b; --muted:#52525b; --subtle:#71717a; --line:#e4e4e7; --soft:#f4f4f5; --accent:#2563eb; }
* { box-sizing: border-box; }
body { margin: 0; font: 14px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: var(--text); background: #fff; }
main { max-width: 980px; margin: 0 auto; padding: 32px 24px 48px; }
h1 { font-size: 26px; margin: 4px 0 2px; } h2 { font-size: 18px; margin: 32px 0 10px; border-bottom: 1px solid var(--line); padding-bottom: 6px; }
h3 { font-size: 15px; margin: 18px 0 8px; } .eyebrow { color: var(--accent); font-size: 12px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; }
.muted { color: var(--muted); } .subtle { color: var(--subtle); font-size: 12px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin: 16px 0; }
.tile { border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; } .tile b { display: block; font-size: 20px; }
.tile span { color: var(--subtle); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 12px; } th, td { border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: left; vertical-align: top; }
th { background: var(--soft); font-size: 12px; color: var(--muted); } td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.best td { font-weight: 600; } .tag { display: inline-block; border: 1px solid var(--line); border-radius: 999px; padding: 0 8px; margin: 0 4px 4px 0; font-size: 12px; }
.notes { white-space: pre-wrap; background: var(--soft); border-radius: 8px; padding: 10px 12px; }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr)); gap: 14px; }
figure { margin: 0; border: 1px solid var(--line); border-radius: 8px; padding: 10px; } figcaption { font-weight: 600; margin-bottom: 4px; }
svg.chart { width: 100%; height: auto; display: block; } .grid { stroke: #ececef; } .axis { stroke: #a1a1aa; } .ref { stroke: #a1a1aa; stroke-dasharray: 4 4; }
.tick, .label { fill: var(--muted); font-size: 11px; } .line { fill: none; stroke-width: 2; } .dot { fill: var(--accent); fill-opacity: .55; }
.s0 { stroke: #2563eb; fill: #2563eb; } .s1 { stroke: #16a34a; fill: #16a34a; } .s2 { stroke: #dc2626; fill: #dc2626; } .s3 { stroke: #d97706; fill: #d97706; }
.s4 { stroke: #7c3aed; fill: #7c3aed; } .s5 { stroke: #0891b2; fill: #0891b2; } .s6 { stroke: #db2777; fill: #db2777; } .s7 { stroke: #65a30d; fill: #65a30d; }
.line.s0, .line.s1, .line.s2, .line.s3, .line.s4, .line.s5, .line.s6, .line.s7 { fill: none; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 12px; font-size: 12px; color: var(--muted); margin-top: 4px; }
.swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: -1px; }
.cm td { text-align: center; } .cm td.hit { background: #dbeafe; font-weight: 600; }
.model { break-inside: avoid; } footer { margin-top: 40px; color: var(--subtle); font-size: 12px; border-top: 1px solid var(--line); padding-top: 10px; }
@media print { main { padding: 0; } h2 { break-after: avoid; } figure, table { break-inside: avoid; } }
"""


def _e(value):
    return escape("" if value is None else str(value))


def _score(value, metric):
    if value is None:
        return "—"
    if metric.get("kind") == "percent":
        return f"{value * 100:.1f}%"
    return f"{value:.4f}"


def _number(value):
    if isinstance(value, (int, float)):
        return f"{value:,.4f}".rstrip("0").rstrip(".") if isinstance(value, float) else f"{value:,}"
    return _e(value)


def _cv_text(cv, metric):
    if not cv:
        return ""
    if cv.get("mean") is None:
        return _e(cv.get("error") or "—")
    spread = f"{cv['std'] * 100:.1f}" if metric.get("kind") == "percent" else f"{cv['std']:.3f}"
    return f"{_score(cv['mean'], metric)} ± {spread}"


def _table(headers, rows, numeric=()):
    head = "".join(f'<th class="num">{_e(h)}</th>' if i in numeric else f"<th>{_e(h)}</th>" for i, h in enumerate(headers))
    body = []
    for row in rows:
        cls = ' class="best"' if row and row[0] == "1" else ""
        cells = "".join(f'<td class="num">{c}</td>' if i in numeric else f"<td>{c}</td>" for i, c in enumerate(row))
        body.append(f"<tr{cls}>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _confusion(test_metrics):
    matrix, order = test_metrics.get("confusion_matrix"), test_metrics.get("confusion_matrix_order")
    if not matrix or not order or len(order) > 15:
        return ""
    head = "".join(f"<th>{_e(label)}</th>" for label in order)
    rows = []
    for i, (label, counts) in enumerate(zip(order, matrix)):
        cells = "".join(f'<td class="{"hit" if i == j else ""}">{int(c)}</td>' for j, c in enumerate(counts))
        rows.append(f"<tr><th>{_e(label)}</th>{cells}</tr>")
    return ('<h3>Confusion matrix</h3><p class="subtle">Rows: actual class · columns: predicted class (test set)</p>'
            f'<table class="cm"><thead><tr><th></th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>')


def _class_report(test_metrics):
    report = test_metrics.get("classification_report")
    if not isinstance(report, dict):
        return ""
    rows = []
    for label, values in report.items():
        if not isinstance(values, dict):
            continue
        rows.append([_e(label)] + [f"{values.get(k, 0):.3f}" for k in ("precision", "recall", "f1-score")]
                    + [str(int(values.get("support", 0)))])
    return "<h3>Per class</h3>" + _table(["Class", "Precision", "Recall", "F1", "Rows"], rows, numeric=(1, 2, 3, 4))


def _charts(charts, task):
    figures = []

    def add(caption, body):
        if body:
            figures.append(f"<figure><figcaption>{_e(caption)}</figcaption>{body}</figure>")

    roc = charts.get("roc")
    if roc and roc.get("curves"):
        series = [{"label": f"{c['label']} (AUC {c['auc']:.3f})", "xs": c["fpr"], "ys": c["tpr"]} for c in roc["curves"]]
        add("ROC curve", svg.line_chart(series, "False positive rate", "True positive rate", "ROC curve", diagonal=True))
    pr = charts.get("pr")
    if pr and pr.get("curves"):
        series = [{"label": f"{c['label']} (AP {c['ap']:.3f})", "xs": c["recall"], "ys": c["precision"]} for c in pr["curves"]]
        add("Precision-recall", svg.line_chart(series, "Recall", "Precision", "Precision-recall curve"))
    scatter = charts.get("scatter")
    if scatter and scatter.get("actual"):
        add("Predicted vs actual", svg.scatter(scatter["actual"], scatter["predicted"], "Predicted vs actual"))
    histogram = charts.get("residual_histogram")
    if histogram and histogram.get("counts"):
        add("Errors", svg.histogram(histogram["edges"], histogram["counts"], "Prediction errors"))
    importance = charts.get("importance")
    if importance and importance.get("features"):
        items = [(f["name"], float(f["value"])) for f in importance["features"]][:15]
        caption = ("Score lost when a column is shuffled" if importance.get("method") == "permutation"
                   else "Feature importance")
        add(caption, svg.bars(items, caption))
    terms = charts.get("top_terms")
    if terms:
        groups = terms.get("by_class") or ([{"label": "All classes", "terms": terms["overall"]}] if terms.get("overall") else [])
        body = "".join(f"<p><b>{_e(g['label'])}</b>: {_e(', '.join(t['term'] for t in g['terms'][:15]))}</p>" for g in groups)
        add("Words the model relies on", f"<div>{body}</div>" if body else "")
    return f'<div class="charts">{"".join(figures)}</div>' if figures else ""


def _by_target(targets, ranked, config, metric):
    """Each model's test score per column to predict, the best one per column marked (runs with several)."""
    weights = config.get("target_weights") or {}
    best = {t: max((m["targets"].get(t) for m in ranked if (m.get("targets") or {}).get(t) is not None), default=None)
            for t in targets}
    headers = ["Model"] + [f"{t} (weight {float(weights.get(t, 1)):g})" for t in targets]
    rows = []
    for m in ranked:
        scores = m.get("targets") or {}
        rows.append([_e(m["label"])] + [
            _score(scores.get(t), metric) + (" ★" if scores.get(t) is not None and scores.get(t) == best[t] else "")
            for t in targets])
    note = ("<p class=\"subtle\">Each column to predict has its own model in every row above, trained and tuned on "
            "that column alone, so no column's accuracy is traded for another's. The run's score is the "
            "weighted average; ★ marks the best model for each column, which together make the "
            "&ldquo;best model for each target&rdquo;.</p>")
    return "<h3>By column to predict</h3>" + _table(headers, rows, numeric=tuple(range(1, len(headers)))) + note


def _model_targets(entry, charts, task, metric):
    """A multi-target model's section body: per column, its metrics, settings and charts."""
    parts = []
    test = (entry.get("test_metrics") or {}).get("targets") or {}
    if (entry.get("test_metrics") or {}).get("exact_match") is not None:
        parts.append(f'<p class="muted">All columns right: {_score(entry["test_metrics"]["exact_match"], metric)} '
                     "of the test rows.</p>")
    for target, scores in (entry.get("targets") or {}).items():
        parts.append(f"<h4>{_e(target)}</h4>")
        facts = [f"Test {metric['name']} {_score(scores.get('accuracy_test'), metric)}",
                 f"train {_score(scores.get('accuracy_train'), metric)}"]
        if scores.get("cv"):
            facts.append(f"cross-validation {_cv_text(scores['cv'], metric)}")
        parts.append(f'<p class="muted">{" · ".join(facts)}</p>')
        metrics = test.get(target) or {}
        if task != "MLRegression":
            parts.append(_confusion(metrics))
            parts.append(_class_report(metrics))
        else:
            values = [[_e(REGRESSION_LABELS.get(k, k)), _number(v)] for k, v in metrics.items() if isinstance(v, (int, float))]
            if values:
                parts.append(_table(["Test metric", "Value"], values, numeric=(1,)))
        params = (entry.get("best_params") or {}).get(target)
        if params:
            parts.append(f'<p class="subtle">Tuned settings: {_e(", ".join(f"{k} = {v}" for k, v in params.items()))}</p>')
        parts.append(_charts(charts.get(target) or {}, task))
    return "".join(parts)


def build_report(run_id):
    """``(html, file_name)`` for run ``run_id``. Raises RunNotFoundError."""
    from easyml import __version__
    from easyml.core.artifacts import read_json
    from easyml.core.evaluation_charts import read_evaluation
    from easyml.utils.database_operations import get_record_by_id
    from easyml.utils.run_summary import (
        model_label,
        parse_metrics,
        rank_models,
        settings_for_display,
        summarize_run,
    )
    from easyml.utils.utils import sanitize_run_name

    record = get_record_by_id(int(run_id))
    run = summarize_run(record)
    task, metric = run["task"], run["metric"]
    folder = record.get("folder_path") or ""
    try:
        config = read_json(folder, "config_json.json")
    except Exception:
        config = {}
    try:
        evaluation = read_evaluation(folder) or {}
    except Exception:
        evaluation = {}
    metrics = parse_metrics(record.get("metrics_json"))
    ranked = rank_models(task, metrics)
    classification = task != "MLRegression"
    created = (run.get("timestamp") or "").replace("T", " ")

    out = [f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
           f"content=\"width=device-width, initial-scale=1\"><title>{_e(run['run_name'])} · Easy-ML report</title>"
           f"<style>{CSS}</style></head><body><main>"]
    out.append(f'<div class="eyebrow">Easy-ML Studio · run #{run["id"]} · {_e(run["task_label"])}</div>')
    out.append(f"<h1>{_e(run['run_name'])}</h1>")
    out.append(f'<p class="muted">Trained {_e(created)} on {_e(run.get("dataset_name") or "its dataset")}'
               + (f" with Easy-ML {_e(config.get('easyml_version'))}" if config.get("easyml_version") else "") + ".</p>")
    if run.get("tags"):
        out.append("<p>" + "".join(f'<span class="tag">{_e(t)}</span>' for t in run["tags"]) + "</p>")
    if run.get("notes"):
        out.append(f'<div class="notes">{_e(run["notes"])}</div>')

    final_of = run.get("parent_run_id") if run.get("origin") == "final" else None
    targets = run.get("targets") or []
    weighted = " (weighted)" if targets else ""  # several columns to predict: scores are weighted averages
    tiles = [("Best model", _e(run.get("best_model_label") or "—")),
             (f"Expected {metric['name']}{weighted}" if final_of else f"Test {metric['name']}{weighted}",
              _score(run.get("best_score"), metric)),
             ("Models trained", str(len(ranked)))]
    if targets:
        tiles.insert(2, ("Columns to predict", _e(", ".join(targets))))
    if evaluation.get("n_test"):
        tiles.append((f"Test rows (run #{int(final_of)})" if final_of else "Test rows", f"{int(evaluation['n_test']):,}"))
    out.append('<div class="tiles">' + "".join(f"<div class=\"tile\"><span>{_e(k)}</span><b>{v}</b></div>" for k, v in tiles) + "</div>")

    lineage = []
    if final_of:
        lineage.append(f"This is the final model of run #{int(final_of)}, trained on all its rows (training and test). "
                       f"No rows were left to test it on, so its test and cross-validation scores and charts are "
                       f"run #{int(final_of)}'s: estimates, measured on data the model had not seen.")
    elif run.get("parent_run_id"):
        how = "scheduled retrain" if run.get("origin") == "schedule" else "retrained"
        lineage.append(f"This run was {how} from run #{int(run['parent_run_id'])}.")
    if run.get("origin") == "import":
        lineage.append("This run was imported from a run archive.")
    if lineage:
        out.append("<p class=\"muted\">" + " ".join(lineage) + "</p>")

    out.append("<h2>Leaderboard</h2>")
    has_cv = any(isinstance(metrics.get(m["key"]), dict) and metrics[m["key"]].get("cv") for m in ranked)
    headers = (["#", "Model", f"Test {metric['name']}{weighted}", f"Train {metric['name']}{weighted}"]
               + (["Cross-validation"] if has_cv else []))
    rows = []
    for i, m in enumerate(ranked, 1):
        entry = metrics.get(m["key"]) if isinstance(metrics.get(m["key"]), dict) else {}
        row = [str(i), _e(m["label"]), _score(m["test_score"], metric), _score(m["train_score"], metric)]
        if has_cv:
            row.append(_cv_text(entry.get("cv"), metric))
        rows.append(row)
    out.append(_table(headers, rows, numeric=tuple(range(2, len(headers)))))
    out.append(f'<p class="subtle">Models are ranked by their {_e(metric["name"].lower())} on the test set, '
               "rows the models never trained on." + (" Cross-validation: mean ± spread over folds of the training data."
                                                     if has_cv else "") + "</p>")
    if targets:
        out.append(_by_target(targets, ranked, config, metric))

    settings = settings_for_display(task, config)
    if settings:
        out.append("<h2>Settings</h2>")
        out.append(_table(["Setting", "Value"], [[_e(s["label"]), _e(s["value"])] for s in settings]))
    recipe = config.get("recipe")
    if isinstance(recipe, dict) and recipe.get("steps"):
        try:
            from easyml.dataprep.steps import STEP_SPECS

            names = [STEP_SPECS.get(step.get("op"), {}).get("label", step.get("op")) for step in recipe["steps"]]
        except Exception:
            names = [str(step.get("op")) for step in recipe["steps"]]
        out.append(f"<h3>Data prep recipe: {_e(recipe.get('name'))}</h3><ol>"
                   + "".join(f"<li>{_e(n)}</li>" for n in names) + "</ol>")
        if os.path.exists(os.path.join(folder, "prep.json")):
            out.append("<p>Predictions clean raw input with this recipe automatically, using the values it learned "
                       "from the training data (fill values, limits, ranges).</p>")
        else:
            out.append("<p>Trained before Easy-ML 0.11: input for predictions must already be cleaned.</p>")

    eval_models = (evaluation.get("models") or {}) if isinstance(evaluation, dict) else {}
    out.append("<h2>Models</h2>")
    for m in ranked:
        entry = metrics.get(m["key"]) if isinstance(metrics.get(m["key"]), dict) else {}
        test_metrics = entry.get("test_metrics") or {}
        out.append(f'<section class="model"><h3>{_e(model_label(m["key"]))}</h3>')
        facts = [f"Test {metric['name']} {_score(m['test_score'], metric)}", f"train {_score(m['train_score'], metric)}"]
        if entry.get("cv"):
            facts.append(f"cross-validation {_cv_text(entry['cv'], metric)}")
        out.append(f'<p class="muted">{" · ".join(facts)}</p>')
        if entry.get("targets"):  # several columns to predict: each target's own model, metrics and charts
            out.append(_model_targets(entry, (eval_models.get(m["key"]) or {}).get("targets") or {}, task, metric))
            out.append("</section>")
            continue
        if classification:
            out.append(_confusion(test_metrics))
            out.append(_class_report(test_metrics))
        else:
            values = [[_e(REGRESSION_LABELS.get(k, k)), _number(v)] for k, v in test_metrics.items()
                      if isinstance(v, (int, float))]
            if values:
                out.append(_table(["Test metric", "Value"], values, numeric=(1,)))
        cv = entry.get("cv") or {}
        if cv.get("scores"):
            out.append(f'<p class="subtle">{int(cv["folds"])} folds: '
                       f'{_e(", ".join(_score(s, metric) for s in cv["scores"]))}</p>')
        if entry.get("best_params"):
            params = ", ".join(f"{k} = {v}" for k, v in entry["best_params"].items())
            out.append(f'<p class="subtle">Tuned settings: {_e(params)}</p>')
        out.append(_charts(eval_models.get(m["key"]) or {}, task))
        out.append("</section>")

    out.append(f"<footer>Made with Easy-ML Studio {_e(__version__)} on {_e(datetime.now().strftime('%Y-%m-%d %H:%M'))}. "
               "Scores are from the run's own test set.</footer></main></body></html>")
    name = f"easyml-run-{int(run_id)}-{sanitize_run_name(run.get('run_name') or 'run')}.html"
    return "".join(out), name


def write_report(run_id, path=None):
    """Write the report of run ``run_id`` to ``path`` (default: its own name in the current folder). Returns the path."""
    html, name = build_report(run_id)
    path = os.path.abspath(path or name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(html)
    return path
