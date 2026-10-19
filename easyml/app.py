"""
Command-line entry point: ``easy-ml`` (or ``python -m easyml``).

    easy-ml                      start the studio on http://127.0.0.1:5010 and open the browser
    easy-ml --port 8080 --no-browser
    easy-ml download-models      fetch the spaCy English model used for lemmatization
    easy-ml runs                 list trained runs
    easy-ml predict 5 data.csv   predict with run #5 (see `easy-ml predict --help`)
    easy-ml check 5 data.csv     does data.csv look like the data run #5 was trained on?
    easy-ml retrain 5 new.csv    train run #5's settings on new data
    easy-ml final 5              train run #5's best model again on all its data, for deployment
    easy-ml export 5             save run #5 as a .zip; `easy-ml import file.zip` loads it elsewhere
    easy-ml serve 5              serve run #5 as a REST API on http://127.0.0.1:8000
    easy-ml prep apply customers new.csv -o clean.csv    re-apply a saved data-prep recipe
    easy-ml schedule worker      run scheduled retraining without the studio (see `easy-ml schedule --help`)
    easy-ml report 5             save run #5 as one HTML file to share
"""

import argparse
import logging
import os
import socket
import subprocess
import sys
import threading
import webbrowser

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5010


def _build_parser():
    from easyml import __version__

    parser = argparse.ArgumentParser(prog="easy-ml", description="Easy-ML Studio: no-code machine learning in your browser.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--host", default=os.environ.get("EASYML_HOST", DEFAULT_HOST),
                        help="Interface to bind (default: 127.0.0.1). Use 0.0.0.0 to expose the studio on your network.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("EASYML_PORT", DEFAULT_PORT)),
                        help="Port to listen on (default: 5010).")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically.")
    parser.add_argument("--no-console", action="store_true", help="Do not print application logs to the console.")
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode (development only).")
    parser.add_argument("--no-scheduler", action="store_true",
                        help="Do not run scheduled retraining while the studio is open (or set EASYML_SCHEDULER=0).")

    subcommands = parser.add_subparsers(dest="command")
    subcommands.add_parser("download-models", help="Download the spaCy English model used for NLP lemmatization.")

    runs = subcommands.add_parser("runs", help="List trained runs with their best model.")
    runs.add_argument("--task", choices=["MLClassification", "MLRegression", "NLPTextClassification", "ImageClassification"],
                      help="Only show runs of this task.")
    runs.add_argument("--limit", type=int, default=20, help="Maximum number of runs to show (default: 20).")
    runs.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    runs.add_argument("--tag", help="Only runs with this tag.")

    report = subcommands.add_parser("report", help="Save a run as one HTML file to share (metrics, charts, settings).")
    report.add_argument("run_id", type=int, help="Run to report on (see `easy-ml runs`).")
    report.add_argument("-o", "--output", help="File to write (default: easyml-run-<id>-<name>.html here).")

    tag = subcommands.add_parser("tag", help="Add tags to a run (or remove them with --remove).")
    tag.add_argument("run_id", type=int, help="Run to tag (see `easy-ml runs`).")
    tag.add_argument("tags", nargs="+", help="One or more tags, e.g. churn production.")
    tag.add_argument("--remove", action="store_true", help="Remove these tags instead of adding them.")

    predict = subcommands.add_parser("predict", help="Make predictions with a trained run.")
    predict.add_argument("run_id", type=int, help="Run number (see `easy-ml runs`).")
    predict.add_argument("input", help="CSV/Excel file, text, image file or image folder.")
    predict.add_argument("--model", help="Model class name (default: the run's best model).")
    predict.add_argument("-o", "--output", help="Write the predictions to this CSV file instead of the console.")
    predict.add_argument("--text-column", help="Text column of the input CSV (text runs; default: the training column).")
    predict.add_argument("--no-check", action="store_true",
                         help="Do not check whether the file looks like the training data.")

    check = subcommands.add_parser("check", help="Check whether a file looks like the data a run was trained on.")
    check.add_argument("run_id", type=int, help="Run to compare with (see `easy-ml runs`).")
    check.add_argument("input", help="CSV or Excel file.")
    check.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    retrain = subcommands.add_parser("retrain", help="Train a run again on new data, with the same settings.")
    retrain.add_argument("run_id", type=int, help="Run to retrain (see `easy-ml runs`).")
    retrain.add_argument("input", help="New CSV/Excel file, or image folder for image runs.")
    retrain.add_argument("--name", help="Name of the new run (default: '<run name>-retrain').")
    retrain.add_argument("--models", help="Comma-separated model keys to train instead of the run's models.")
    retrain.add_argument("--recipe", help="Data-prep recipe (name or id) to clean the new data with first; 'none' uses "
                                          "the file as it is. Default: the recipe the run was trained with, if any.")
    retrain.add_argument("--json", action="store_true", help="Print the result as JSON.")

    final = subcommands.add_parser("final", help="Train a run's final model on all its data (training + test rows).")
    final.add_argument("run_id", type=int, help="The evaluated run (see `easy-ml runs`).")
    final.add_argument("--model", action="append", dest="models", metavar="MODEL",
                       help="Model class name or key to train (repeatable; default: the run's best model).")
    final.add_argument("--name", help="Name of the new run (default: '<run name>-final').")
    final.add_argument("--json", action="store_true", help="Print the result as JSON.")

    export = subcommands.add_parser("export", help="Save a run as a .zip to move it to another computer.")
    export.add_argument("run_id", type=int, help="Run to export (see `easy-ml runs`).")
    export.add_argument("-o", "--output", help="Archive path (default: easyml-run-<id>-<name>.zip here).")
    export.add_argument("--no-data", action="store_true", help="Leave out the training data (data.csv).")

    import_ = subcommands.add_parser("import", help="Load a run archive made by `easy-ml export`.")
    import_.add_argument("file", help="The .zip archive.")
    import_.add_argument("--name", help="Name for the imported run (default: its original name).")
    import_.add_argument("--yes", action="store_true", help="Do not ask for confirmation (only for trusted files).")

    serve = subcommands.add_parser("serve", help="Serve a run as a REST API that other programs can call.")
    serve.add_argument("run_id", type=int, nargs="?", help="Run to serve (see `easy-ml runs`).")
    serve.add_argument("--schedule", metavar="NAME",
                       help="Serve the model a retraining schedule currently uses, switching when it promotes a new one.")
    serve.add_argument("--model", help="Model class name (default: the run's best model).")
    serve.add_argument("--host", dest="serve_host", default="127.0.0.1", metavar="HOST",
                       help="Interface to listen on (default 127.0.0.1: this computer only; 0.0.0.0: your network).")
    serve.add_argument("--port", dest="serve_port", type=int, default=8000, metavar="PORT", help="Port (default 8000).")
    serve.add_argument("--threads", type=int, default=4, help="Requests handled in parallel (default 4).")
    serve.add_argument("--api-key", metavar="KEY", help="Require this key (default: the EASYML_SERVE_KEY environment variable).")
    serve.add_argument("--cors-origin", action="append", default=[], metavar="URL",
                       help="Let web pages from this origin call the API from a browser (repeatable, * for any).")
    serve.add_argument("--max-mb", type=float, default=20, help="Largest request accepted, in MB (default 20).")
    serve.add_argument("--access-log", action="store_true", help="Print one line per request.")

    prep = subcommands.add_parser("prep", help="Data-prep recipes: list, show, apply to a file.")
    prep_commands = prep.add_subparsers(dest="prep_command")
    prep_commands.add_parser("list", help="List saved recipes.")
    show = prep_commands.add_parser("show", help="Print a recipe's steps (as JSON).")
    show.add_argument("recipe", help="Recipe name or id.")
    apply_ = prep_commands.add_parser("apply", help="Apply a recipe to a CSV/Excel file.")
    apply_.add_argument("recipe", help="Recipe name, id, or a recipe .json file.")
    apply_.add_argument("input", help="CSV or Excel file to clean.")
    apply_.add_argument("-o", "--output", help="Write the cleaned CSV here (default: print it).")

    schedule = subcommands.add_parser("schedule", help="Scheduled retraining: list, create, run, worker, ...")
    schedule_commands = schedule.add_subparsers(dest="schedule_command")
    listing = schedule_commands.add_parser("list", help="List schedules.")
    listing.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    create = schedule_commands.add_parser("create", help="Retrain a run automatically on new data.")
    create.add_argument("run_id", type=int, help="Run to keep fresh (see `easy-ml runs`).")
    create.add_argument("source", help="Data file to watch, a folder (its newest matching file is used), or for "
                                       "image runs a folder of class folders.")
    create.add_argument("--name", required=True, help="Name of the schedule.")
    when = create.add_mutually_exclusive_group(required=True)
    when.add_argument("--every", metavar="INTERVAL", help="Retrain every INTERVAL, e.g. 30m, 6h, 1d.")
    when.add_argument("--daily", metavar="HH:MM", help="Retrain every day at this time.")
    when.add_argument("--weekly", nargs=2, metavar=("DAY", "HH:MM"), help="Retrain every week, e.g. --weekly mon 02:00.")
    when.add_argument("--on-new-data", action="store_true", help="Retrain whenever the data changes or a new file appears.")
    create.add_argument("--pattern", help="File names to use in a folder (default: *.csv;*.xlsx;*.xls).")
    create.add_argument("--recipe", help="Data-prep recipe to clean the data with ('none': as it is). "
                                         "Default: the recipe the run was trained with, if any.")
    create.add_argument("--min-improvement", type=float, default=0.0,
                        help="Use the new model only if it scores at least this much better (0.01 = 1 point).")
    create.add_argument("--disabled", action="store_true", help="Create it switched off.")
    for command, text in (("run", "Retrain now on the current data."), ("enable", "Switch a schedule on."),
                          ("disable", "Switch a schedule off."), ("delete", "Delete a schedule (its runs are kept).")):
        sub = schedule_commands.add_parser(command, help=text)
        sub.add_argument("schedule", help="Schedule name or id.")
    events = schedule_commands.add_parser("events", help="What a schedule did: promoted, kept, failed.")
    events.add_argument("schedule", help="Schedule name or id.")
    events.add_argument("--limit", type=int, default=20, help="Number of events (default 20).")
    events.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    worker = schedule_commands.add_parser("worker", help="Run schedules until stopped, without the studio.")
    worker.add_argument("--poll", type=int, default=30, help="Seconds between checks (default 30).")
    schedule_commands.add_parser("run-due", help="Run the schedules that are due now once, then exit (for cron or "
                                                 "Windows Task Scheduler).")
    return parser


def download_models():
    try:
        import spacy  # noqa: F401
    except ImportError:
        print('spaCy is not installed. Install the NLP extra first: pip install "easy-ml-studio[nlp]"')
        return 1
    print("Downloading spaCy model 'en_core_web_sm'...")
    return subprocess.call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])


def _port_in_use(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1" if host in ("0.0.0.0", "") else host, port)) == 0


def _clean_old_uploads(days=30):
    """Uploads are only needed while they are explored or trained on (runs keep a copy): drop month-old ones."""
    from easyml.utils.storage import clean_uploads

    try:
        clean_uploads(older_than_hours=days * 24)
    except Exception:  # housekeeping must never stop the studio from starting
        pass


def run_server(host=DEFAULT_HOST, port=DEFAULT_PORT, open_browser=True, console_logs=True, debug=False, scheduler=True):
    # Must be decided before the logger creates its handlers.
    if console_logs:
        os.environ.setdefault("EASYML_CONSOLE_LOG_LEVEL", "INFO")
    else:
        os.environ["EASYML_NO_CONSOLE_LOGS"] = "1"

    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    if _port_in_use(host, port):
        print(f"Port {port} is already in use. Is Easy-ML already running? Try: easy-ml --port {port + 1}")
        return 1

    from easyml.app_factory import create_app
    from easyml.utils.logger import write_log

    app = create_app()
    socketio = app.extensions["easyml_socketio"]

    display_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    url = f"http://{display_host}:{port}"
    print(f"\n  Easy-ML Studio is running at {url}\n  Press CTRL+C to stop.\n")
    write_log(f"easy-ml server started on {host}:{port}", "info", "app.py")

    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    _clean_old_uploads()
    if scheduler:  # scheduled retraining runs while the studio is open (not in tests or create_app users)
        from easyml.scheduling.host import start_studio_scheduler

        start_studio_scheduler(socketio)

    socketio.run(app, host=host, port=port, debug=debug, use_reloader=False, allow_unsafe_werkzeug=True)
    return 0


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.command == "download-models":
        return download_models()
    if args.command == "runs":
        from easyml.cli import list_runs
        return list_runs(task=args.task, limit=args.limit, as_json=args.json, tag=args.tag)
    if args.command == "report":
        from easyml.cli import report
        return report(args.run_id, output=args.output)
    if args.command == "tag":
        from easyml.cli import tag
        return tag(args.run_id, args.tags, remove=args.remove)
    if args.command == "predict":
        from easyml.cli import predict
        return predict(args.run_id, args.input, model=args.model, output=args.output, text_column=args.text_column,
                       check_data=not args.no_check)
    if args.command == "check":
        from easyml.cli import check
        return check(args.run_id, args.input, as_json=args.json)
    if args.command == "final":
        from easyml.cli import final
        return final(args.run_id, models=args.models, name=args.name, as_json=args.json)
    if args.command == "retrain":
        from easyml.cli import retrain
        models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
        return retrain(args.run_id, args.input, name=args.name, models=models, as_json=args.json, recipe=args.recipe)
    if args.command == "export":
        from easyml.cli import export_run
        return export_run(args.run_id, output=args.output, include_data=not args.no_data)
    if args.command == "import":
        from easyml.cli import import_run
        return import_run(args.file, name=args.name, assume_yes=args.yes)
    if args.command == "prep":
        from easyml.cli import prep
        return prep(args.prep_command, recipe=getattr(args, "recipe", None), source=getattr(args, "input", None),
                    output=getattr(args, "output", None))
    if args.command in ("serve", "schedule"):
        # Long-running commands write their own log files, so they never rotate the studio's files.
        os.environ.setdefault("EASYML_LOG_NAME", "serve" if args.command == "serve" else "scheduler")
    if args.command == "serve":
        from easyml.cli import serve
        return serve(args.run_id, model=args.model, host=args.serve_host, port=args.serve_port, threads=args.threads,
                     api_key=args.api_key, cors_origins=args.cors_origin, max_mb=args.max_mb,
                     access_log=args.access_log, schedule=args.schedule)
    if args.command == "schedule":
        from easyml.cli import schedule
        return schedule(args)
    return run_server(
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
        console_logs=not args.no_console,
        debug=args.debug,
        scheduler=not args.no_scheduler,
    )


if __name__ == "__main__":
    sys.exit(main())
