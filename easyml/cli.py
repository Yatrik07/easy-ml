"""
`easy-ml runs` and `easy-ml predict`: use trained models from the command line.

    easy-ml runs                                    # list runs with their best model
    easy-ml predict 5 new_data.csv -o out.csv       # tabular / text CSV: adds a Prediction column
    easy-ml predict 5 "great phone, fast delivery"  # text: prints the label
    easy-ml predict 7 photos/                       # images: one row per image
    easy-ml retrain 5 october.csv --name churn-oct  # same settings as run #5, new data
"""

import csv
import json
import os
import re
import sys


def _quiet_logs():
    # Logs would mix with the CSV written to stdout.
    os.environ["EASYML_NO_CONSOLE_LOGS"] = "1"


def _format_score(value, metric):
    if value is None:
        return "-"
    return f"{value:.4f}" if metric["kind"] == "decimal" else f"{value * 100:.1f}%"


def list_runs(task=None, limit=20, as_json=False, out=None):
    _quiet_logs()
    from easyml.utils.database_operations import create_database_table, get_data_from_database
    from easyml.utils.run_summary import summarize_run

    out = out or sys.stdout
    create_database_table("runs")
    result = get_data_from_database()
    if result.get("status_code") != 200:
        print("Could not read the runs database.", file=sys.stderr)
        return 1
    rows = sorted(result["data"], key=lambda r: r["id"], reverse=True)
    summaries = [summarize_run(r) for r in rows if not task or r["task"] == task][:limit]

    if as_json:
        json.dump(summaries, out, indent=2, default=str)
        out.write("\n")
        return 0
    if not summaries:
        print("No runs yet. Train a model with `easy-ml` first.", file=out)
        return 0

    table = [("ID", "NAME", "TASK", "DATASET", "BEST MODEL", "SCORE", "CREATED")]
    for s in summaries:
        table.append((
            str(s["id"]), s["run_name"] or "", s["task_label"], s["dataset_name"] or "",
            s["best_model"] or "-", _format_score(s["best_score"], s["metric"]),
            (s["timestamp"] or "").replace("T", " "),
        ))
    widths = [min(max(len(row[i]) for row in table), 32) for i in range(len(table[0]))]
    for row in table:
        print("  ".join(cell[:widths[i]].ljust(widths[i]) for i, cell in enumerate(row)).rstrip(), file=out)
    return 0


def predict(run_id, source, model=None, output=None, text_column=None, out=None):
    _quiet_logs()
    from easyml.predictor import IMAGE_TASK, TABULAR_TASKS, TEXT_TASK, Predictor, image_paths
    from easyml.utils.database_operations import RunNotFoundError

    out = out or sys.stdout
    try:
        predictor = Predictor(run_id, model_name=model)
    except RunNotFoundError:
        print(f"Run {run_id} was not found. List your runs with `easy-ml runs`.", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    is_file = os.path.isfile(source)
    try:
        if predictor.task in TABULAR_TASKS or (predictor.task == TEXT_TASK and is_file):
            from easyml.utils.data_loader import load_data_from_path

            if not is_file:
                print(f"File not found: {source}", file=sys.stderr)
                return 1
            frame = load_data_from_path(source)
            if predictor.task == TEXT_TASK:
                predictions, confidences = predictor._predict_text(source, text_column=text_column)
            else:
                predictions, confidences = predictor.predict_with_confidence(frame)
            frame = frame.copy()
            frame["Prediction"] = predictions
            if confidences is not None:
                frame["Confidence"] = [round(c, 4) for c in confidences]
            if output:
                frame.to_csv(output, index=False)
                print(f"Wrote {len(frame)} predictions to {output}", file=sys.stderr)
            else:
                frame.to_csv(out, index=False, lineterminator="\n")
            return 0

        if predictor.task == TEXT_TASK:
            labels, confidences = predictor.predict_with_confidence(source)
            confidence = f" ({confidences[0] * 100:.1f}%)" if confidences else ""
            print(f"{labels[0]}{confidence}", file=out)
            return 0

        if predictor.task == IMAGE_TASK:
            paths = image_paths(source)
            labels, confidences = predictor.predict_with_confidence(paths)
            handle = open(output, "w", newline="", encoding="utf-8") if output else out
            try:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(["path", "prediction", "confidence"])
                for path, label, conf in zip(paths, labels, confidences):
                    writer.writerow([path, label, round(conf, 4)])
            finally:
                if output:
                    handle.close()
                    print(f"Wrote {len(paths)} predictions to {output}", file=sys.stderr)
            return 0
    except (ValueError, TypeError, FileNotFoundError, ImportError) as exc:
        print(f"Prediction failed: {exc}", file=sys.stderr)
        return 1

    print(f"Unsupported task {predictor.task!r}", file=sys.stderr)
    return 1


def _console_text(message, stream):
    """Progress messages start with an emoji; drop it and anything the console cannot print."""
    text = re.sub(r"^[^\w(\[]+", "", str(message)).strip()
    encoding = getattr(stream, "encoding", None) or "utf-8"
    return text.encode(encoding, "replace").decode(encoding)


def _print_leaderboard(task, performance, out):
    from easyml.utils.run_summary import primary_metric, rank_models

    metric = primary_metric(task)
    ranked = rank_models(task, performance)
    width = max([len(m["label"]) for m in ranked] + [5])
    print(f"  {'MODEL'.ljust(width)}  TEST {metric['name']}", file=out)
    for m in ranked:
        print(f"  {m['label'].ljust(width)}  {_format_score(m['test_score'], metric)}", file=out)


def retrain(run_id, data, name=None, models=None, as_json=False, out=None):
    _quiet_logs()
    from easyml.retraining import DatasetMismatchError, retrain as retrain_run
    from easyml.utils.database_operations import RunNotFoundError
    from easyml.utils.socket_emitter import progress_listener

    out = out or sys.stdout

    def show(entry):
        pct = entry.get("pct")
        prefix = f"[{pct:>3}%] " if isinstance(pct, int) else ""
        print(prefix + _console_text(entry.get("message", ""), sys.stderr), file=sys.stderr)

    try:
        with progress_listener(show):
            result = retrain_run(run_id, data, name=name, models=models)
    except RunNotFoundError:
        print(f"Run {run_id} was not found. List your runs with `easy-ml runs`.", file=sys.stderr)
        return 1
    except DatasetMismatchError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (ValueError, TypeError, FileNotFoundError, ImportError) as exc:
        print(f"Retrain failed: {exc}", file=sys.stderr)
        return 1

    if as_json:
        json.dump({"run_id": result.run_id, "parent_run_id": result.parent_run_id, "task": result.task,
                   "run_name": result.run_name, "best_model": result.best_model, "warnings": result.warnings,
                   "performance": result.performance}, out, indent=2, default=str)
        out.write("\n")
        return 0
    for warning in result.warnings:
        print(f"Note: {warning}", file=sys.stderr)
    print(f"New run #{result.run_id} ({result.run_name}), retrained from run #{result.parent_run_id}:", file=out)
    _print_leaderboard(result.task, result.performance, out)
    return 0
