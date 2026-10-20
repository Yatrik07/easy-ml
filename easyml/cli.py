"""
`easy-ml runs` and `easy-ml predict`: use trained models from the command line.

    easy-ml runs                                    # list runs with their best model
    easy-ml predict 5 new_data.csv -o out.csv       # tabular / text CSV: adds a Prediction column
    easy-ml predict 5 "great phone, fast delivery"  # text: prints the label
    easy-ml predict 7 photos/                       # images: one row per image
    easy-ml retrain 5 october.csv --name churn-oct  # same settings as run #5, new data
    easy-ml export 5 -o churn.zip                   # move a run to another computer...
    easy-ml import churn.zip                        # ...and load it there
    easy-ml serve 5 --port 8000                     # a REST API for run #5
    easy-ml prep apply customers new.csv -o clean.csv
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


def list_runs(task=None, limit=20, as_json=False, out=None, tag=None):
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
    summaries = [summarize_run(r) for r in rows if not task or r["task"] == task]
    if tag:
        wanted = tag.strip().lower()
        summaries = [s for s in summaries if any(t.lower() == wanted for t in s["tags"])]
    summaries = summaries[:limit]

    if as_json:
        json.dump(summaries, out, indent=2, default=str)
        out.write("\n")
        return 0
    if not summaries:
        print(f"No runs tagged {tag!r}." if tag else "No runs yet. Train a model with `easy-ml` first.", file=out)
        return 0

    with_tags = any(s["tags"] for s in summaries)
    table = [("ID", "NAME", "TASK", "DATASET", "BEST MODEL", "SCORE", "CREATED") + (("TAGS",) if with_tags else ())]
    for s in summaries:
        table.append((
            str(s["id"]), s["run_name"] or "", s["task_label"], s["dataset_name"] or "",
            s["best_model"] or "-",
            _format_score(s["best_score"], s["metric"]) + (" (est.)" if s.get("origin") == "final" else ""),
            (s["timestamp"] or "").replace("T", " "),
        ) + ((", ".join(s["tags"]),) if with_tags else ()))
    widths = [min(max(len(row[i]) for row in table), 32) for i in range(len(table[0]))]
    for row in table:
        print("  ".join(cell[:widths[i]].ljust(widths[i]) for i, cell in enumerate(row)).rstrip(), file=out)
    return 0


def report(run_id, output=None, out=None):
    """``easy-ml report RUN_ID [-o file.html]``: the run as one HTML file to share."""
    _quiet_logs()
    from easyml.reporting.report import write_report
    from easyml.utils.database_operations import RunNotFoundError

    out = out or sys.stdout
    try:
        path = write_report(run_id, output)
    except RunNotFoundError:
        print(f"Run {run_id} was not found. List your runs with `easy-ml runs`.", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Could not write the report: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {path}", file=out)
    return 0


def tag(run_id, tags, remove=False, out=None):
    """``easy-ml tag RUN_ID TAG...``: add tags to a run (or remove them with ``--remove``)."""
    _quiet_logs()
    from easyml.utils.database_operations import RunNotFoundError, get_record_by_id
    from easyml.utils.run_meta import parse_tags, update_run_meta

    out = out or sys.stdout
    try:
        current = parse_tags(get_record_by_id(int(run_id)).get("tags"))
        if remove:
            drop = {t.strip().lower() for t in tags}
            new = [t for t in current if t.lower() not in drop]
        else:
            new = current + list(tags)
        run = update_run_meta(run_id, tags=new)
    except RunNotFoundError:
        print(f"Run {run_id} was not found. List your runs with `easy-ml runs`.", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Run #{run['id']} tags: {', '.join(run['tags']) or '(none)'}", file=out)
    return 0


def _report_check(predictor, frame, notes=()):
    """After predicting a file: the cleaning notes and, when it found something, the data check (on stderr)."""
    for note in notes:
        print(f"Note: {note}", file=sys.stderr)
    try:
        report = predictor.check(frame)
    except Exception:  # the check only informs; it never fails a prediction
        return
    if not report.ok:
        print(f"Data check: {report.summary} (details: easy-ml check {predictor.run_id} <file>)", file=sys.stderr)


def check(run_id, source, as_json=False, out=None):
    """Compare a CSV/Excel file with the data a run was trained on."""
    _quiet_logs()
    from easyml.predictor import IMAGE_TASK, Predictor
    from easyml.utils.database_operations import RunNotFoundError

    out = out or sys.stdout
    try:
        predictor = Predictor(run_id)
        if predictor.task == IMAGE_TASK:
            raise ValueError("The data check works on tables and text, not images.")
        if not os.path.isfile(source):
            raise FileNotFoundError(f"File not found: {source}")
        report = predictor.check(source)
    except RunNotFoundError:
        print(f"Run {run_id} was not found. List your runs with `easy-ml runs`.", file=sys.stderr)
        return 1
    except (ValueError, FileNotFoundError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if as_json:
        print(json.dumps(dict(report), indent=2, default=str), file=out)
        return 0
    print(report.summary, file=out)
    if report["columns"]:
        print("", file=out)
        rows = [(c["column"], c["status"], "" if c["psi"] is None else f"{c['psi']:.3f}", c["message"] or "")
                for c in report["columns"]]
        width = max(len("COLUMN"), *(len(str(r[0])) for r in rows))
        print(f"{'COLUMN':<{width}}  STATUS  PSI     DETAILS", file=out)
        for name, status, psi, message in rows:
            print(f"{str(name):<{width}}  {status:<6}  {psi:<6}  {message}", file=out)
    if report["missing_columns"]:
        print(f"\nNot in the file: {', '.join(report['missing_columns'])}", file=out)
    return 0


def predict(run_id, source, model=None, output=None, text_column=None, out=None, check_data=True):
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
            notes = []
            if predictor.task == TEXT_TASK:
                predictions, confidences = predictor._predict_text(source, text_column=text_column, frame=frame)
            else:
                predictions, confidences, notes = predictor.predict_with_notes(frame)
            if check_data:
                _report_check(predictor, frame, notes)
            frame = frame.copy()
            if getattr(predictor, "targets", None):  # several targets: a column per target
                from easyml.core.inference import prediction_columns

                for column, values in prediction_columns(predictor.targets, predictions, confidences).items():
                    frame[column] = values
            else:
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


def final(run_id, models=None, name=None, as_json=False, out=None):
    """Train the final model of a run on all its data (training and test rows)."""
    _quiet_logs()
    from easyml.retraining import train_final
    from easyml.utils.database_operations import RunNotFoundError

    out = out or sys.stdout
    try:
        result = train_final(run_id, models=models, name=name)
    except RunNotFoundError:
        print(f"Run {run_id} was not found. List your runs with `easy-ml runs`.", file=sys.stderr)
        return 1
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if as_json:
        print(json.dumps({"run_id": result.run_id, "final_of_run": result.parent_run_id, "name": result.run_name,
                          "models": list(result.performance), "warnings": result.warnings}, indent=2), file=out)
        return 0
    for warning in result.warnings:
        print(f"Note: {warning}", file=sys.stderr)
    print(f"Run #{result.run_id} ({result.run_name}): the final model of run #{result.parent_run_id}, trained on all "
          f"its rows. Its expected score is run #{result.parent_run_id}'s test score.", file=out)
    print(f"Use it: easy-ml predict {result.run_id} new_data.csv", file=out)
    return 0


def retrain(run_id, data, name=None, models=None, as_json=False, out=None, recipe=None):
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
            result = retrain_run(run_id, data, name=name, models=models, recipe=recipe)
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


def export_run(run_id, output=None, include_data=True, out=None):
    _quiet_logs()
    from easyml.run_archive import export_run as export_archive
    from easyml.utils.database_operations import RunNotFoundError

    out = out or sys.stdout
    try:
        path = export_archive(run_id, output_path=output, include_data=include_data)
    except RunNotFoundError:
        print(f"Run {run_id} was not found. List your runs with `easy-ml runs`.", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    print(path, file=out)
    return 0


def import_run(path, name=None, assume_yes=False, out=None):
    _quiet_logs()
    from easyml.run_archive import TRUST_WARNING, import_archive

    out = out or sys.stdout
    if not assume_yes:
        print(TRUST_WARNING, file=sys.stderr)
        if not sys.stdin or not sys.stdin.isatty():
            print("Add --yes to import without this question.", file=sys.stderr)
            return 1
        if input("Import this run? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Not imported.", file=sys.stderr)
            return 1
    try:
        run_id, notes = import_archive(path, name=name)
    except (OSError, ValueError) as exc:
        print(f"Import failed: {exc}", file=sys.stderr)
        return 1
    for note in notes:
        print(f"Note: {note}", file=sys.stderr)
    print(f"Imported as run #{run_id}. Try: easy-ml predict {run_id} <file>", file=out)
    return 0


def serve(run_id, model=None, host="127.0.0.1", port=8000, threads=4, api_key=None, cors_origins=None, max_mb=20,
          access_log=False, schedule=None):
    _quiet_logs()
    from easyml.serving.runner import serve as run_service
    from easyml.utils.database_operations import RunNotFoundError

    if run_id is not None and schedule is not None:
        print("Give either a run id or --schedule, not both.", file=sys.stderr)
        return 1
    try:
        run_service(run_id, model_name=model, host=host, port=port, threads=threads, api_key=api_key,
                    cors_origins=cors_origins, max_mb=max_mb, access_log=access_log, schedule=schedule)
    except RunNotFoundError:
        print(f"Run {run_id} was not found. List your runs with `easy-ml runs`.", file=sys.stderr)
        return 1
    except LookupError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (ValueError, OSError, ImportError) as exc:
        print(f"Could not start the service: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    return 0


def prep(command, recipe=None, source=None, output=None, out=None):
    _quiet_logs()
    from easyml.dataprep.recipe import Recipe, list_recipes

    out = out or sys.stdout
    try:
        if command == "list" or command is None:
            recipes = list_recipes()
            if not recipes:
                print("No recipes yet. Save one from the studio's Data prep page.", file=out)
                return 0
            print(f"{'ID':<4}  {'NAME':<32}  {'STEPS':>5}  UPDATED", file=out)
            for r in recipes:
                print(f"{r['id']:<4}  {r['name'][:32]:<32}  {r['steps']:>5}  {(r['updated_at'] or '').replace('T', ' ')}",
                      file=out)
            return 0
        loaded = Recipe.load(int(recipe) if str(recipe).isdigit() else recipe)
        if command == "show":
            out.write(loaded.to_json() + "\n")
            return 0
        if command == "apply":
            if not os.path.isfile(source):
                print(f"File not found: {source}", file=sys.stderr)
                return 1
            frame = loaded.apply(source, output=output)
            if output:
                print(f"Wrote {len(frame):,} rows x {len(frame.columns)} columns to {output}", file=sys.stderr)
            else:
                frame.to_csv(out, index=False, lineterminator="\n")
            return 0
    except LookupError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (ValueError, OSError) as exc:
        print(f"Data prep failed: {exc}", file=sys.stderr)
        return 1
    print("Use: easy-ml prep list | show RECIPE | apply RECIPE INPUT [-o OUTPUT]", file=sys.stderr)
    return 1


def _when_text(value):
    return (value or "").replace("T", " ")[:16] or "-"


def _score_text(value):
    return "n/a" if value is None else f"{value:.4f}"


def schedule(args, out=None):
    """``easy-ml schedule ...``: see ``easy-ml schedule --help``."""
    _quiet_logs()
    from easyml import schedules
    from easyml.scheduling.engine import ScheduleBusy
    from easyml.utils.socket_emitter import progress_listener

    out = out or sys.stdout
    command = args.schedule_command
    try:
        if command in (None, "list"):
            rows = schedules.list_schedules()
            if getattr(args, "json", False):
                json.dump(rows, out, indent=2, default=str)
                out.write("\n")
                return 0
            if not rows:
                print("No schedules yet. Create one with `easy-ml schedule create RUN_ID SOURCE --name NAME --daily 02:00`.",
                      file=out)
                return 0
            print(f"{'ID':<4}  {'NAME':<24}  {'IN USE':<7}  {'WHEN':<26}  {'NEXT RUN':<16}  LAST RESULT", file=out)
            for r in rows:
                state = "off" if not r["enabled"] else "running" if r["running"] else (r["last_status"] or "-")
                print(f"{r['id']:<4}  {r['name'][:24]:<24}  {'#' + str(r['champion']['id']):<7}  {r['when'][:26]:<26}  "
                      f"{_when_text(r['next_run_at']):<16}  {state}", file=out)
            return 0
        if command == "create":
            trigger = {"every": args.every, "daily": args.daily, "on_new_data": args.on_new_data,
                       "weekly": " ".join(args.weekly) if args.weekly else None}
            row = schedules.create(args.run_id, args.source, args.name, pattern=args.pattern, recipe=args.recipe,
                                   min_improvement=args.min_improvement, enabled=not args.disabled, **trigger)
            print(f"Created schedule '{row['name']}' (#{row['id']}): retrain run #{args.run_id}, {row['when'].lower()}.",
                  file=out)
            if row["next_run_at"]:
                print(f"Next run: {_when_text(row['next_run_at'])}.", file=out)
            print("Schedules run while Easy-ML Studio is open, or with `easy-ml schedule worker`.", file=out)
            return 0
        if command == "run":
            def show(entry):
                pct = entry.get("pct")
                prefix = f"[{pct:>3}%] " if isinstance(pct, int) else ""
                print(prefix + _console_text(entry.get("message", ""), sys.stderr), file=sys.stderr)

            with progress_listener(show):
                event = schedules.run_now(args.schedule)
            print(f"{event['decision']}: {event['message']}", file=out)
            return 0
        if command in ("enable", "disable"):
            row = schedules.enable(args.schedule) if command == "enable" else schedules.disable(args.schedule)
            print(f"Schedule '{row['name']}' is {'on' if row['enabled'] else 'off'}.", file=out)
            return 0
        if command == "delete":
            schedules.delete(args.schedule)
            print("Deleted. The runs it trained stay in Run History.", file=out)
            return 0
        if command == "events":
            rows = schedules.events(args.schedule, limit=args.limit)
            if args.json:
                json.dump(rows, out, indent=2, default=str)
                out.write("\n")
                return 0
            if not rows:
                print("Nothing has happened yet.", file=out)
                return 0
            for e in rows:
                new = f"#{e['challenger_run_id']}" if e["challenger_run_id"] else "-"
                print(f"{_when_text(e['started_at'])}  {e['decision']:<13}  new {new:<6}  "
                      f"{e['metric'] or ''} {_score_text(e['challenger_score'])} vs {_score_text(e['champion_score'])}  "
                      f"{e['data_file'] or ''}", file=out)
                if e["decision"] == "failed" and e["message"]:
                    print(f"    {e['message']}", file=out)
            return 0
        if command == "worker":
            return schedules.worker(poll_seconds=max(5, args.poll))
        if command == "run-due":
            events = schedules.run_due()
            print(f"Ran {len(events)} schedule(s)." if events else "Nothing was due.", file=out)
            for e in events:
                print(f"  {e['decision']}: {e['message']}", file=out)
            return 0
    except ScheduleBusy as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (LookupError, ValueError, OSError, ImportError) as exc:
        print(f"Schedule {command} failed: {exc}", file=sys.stderr)
        return 1
    except InterruptedError:
        return 1
    print("Use: easy-ml schedule list | create | run | enable | disable | delete | events | worker | run-due",
          file=sys.stderr)
    return 1
