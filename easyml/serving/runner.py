"""Start the prediction service: ``easy-ml serve`` and ``easyml.serve``."""

import os
import socket
import sys
import threading

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
FOLLOW_SECONDS = 15


def _port_in_use(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1" if host in ("0.0.0.0", "") else host, port)) == 0


def build_app(run_id, model_name=None, api_key=None, cors_origins=(), max_mb=20, access_log=False, schedule_name=None):
    """Load the model (once, up front) and return ``(app, predictor)``."""
    from easyml.predictor import Predictor
    from easyml.serving.server import ServeState, create_serve_app, model_info

    predictor = Predictor(run_id, model_name=model_name).warm_up()
    info = model_info(predictor)
    if schedule_name:
        info["schedule"] = schedule_name
    app = create_serve_app(predictor, api_key=api_key, cors_origins=cors_origins, max_mb=max_mb,
                           access_log=access_log, state=ServeState(predictor, info))
    return app, predictor


def follow_schedule(state, schedule_id, run_id, out=None, every=FOLLOW_SECONDS):
    """
    Keep ``state`` on the schedule's current model: every ``every`` seconds read the champion (one small
    database read) and, when it changed, load the new model first and then swap it in, so no request
    waits or fails. Returns an Event that stops the thread.
    """
    from easyml.predictor import Predictor
    from easyml.scheduling import store
    from easyml.serving.server import model_info

    out = out or sys.stdout
    stop = threading.Event()

    def loop():
        current, failed = int(run_id), None
        while not stop.wait(every):
            champion = None
            try:
                row = store.get_schedule(schedule_id)
            except LookupError:
                print(f"  The schedule was deleted; still serving run #{current}.", file=out, flush=True)
                return
            except Exception:  # e.g. the database is busy: try again on the next tick
                continue
            try:
                champion = int(row.get("champion_run_id") or row["base_run_id"])
                if champion in (current, failed):
                    continue
                predictor = Predictor(champion).warm_up()
                info = model_info(predictor)
                info["schedule"] = row["name"]
                state.swap(predictor, info)
                current, failed = champion, None
                print(f"  Schedule '{row['name']}' promoted run #{champion}: now serving {predictor.model_name}.",
                      file=out, flush=True)
            except Exception as exc:  # keep serving the current model; retry when the champion changes again
                failed = champion
                print(f"  Could not switch to the schedule's new model: {exc}", file=out, flush=True)

    threading.Thread(target=loop, name="easyml-serve-follow", daemon=True).start()
    return stop


def serve(run_id=None, model_name=None, host=DEFAULT_HOST, port=DEFAULT_PORT, threads=4, api_key=None,
          cors_origins=None, max_mb=20, access_log=False, out=None, schedule=None):
    """
    Serve a trained run as a REST API until stopped (Ctrl+C).

    Args:
        run_id: The run to serve (see Run History or ``easy-ml runs``).
        model_name: A model of the run (default: its best model).
        host: Interface to listen on. The default only accepts requests from this computer;
            use "0.0.0.0" to accept requests from your network.
        port: Port (default 8000, like the exported API).
        threads: Requests handled in parallel.
        api_key: Require this key on /info and /predict (default: the EASYML_SERVE_KEY variable).
        cors_origins: Web pages (origins) allowed to call the API from a browser.
        max_mb: Largest request accepted, in MB.
        access_log: Print one line per request.
        schedule: Instead of ``run_id``, a retraining schedule (name or id): serve the model it currently
            uses and switch to each new model it promotes, without a restart.
    """
    out = out or sys.stdout
    api_key = api_key or os.environ.get("EASYML_SERVE_KEY") or None
    row = None
    if schedule is not None:
        from easyml.scheduling import store

        row = store.get_schedule(schedule)
        run_id, model_name = int(row.get("champion_run_id") or row["base_run_id"]), None
    elif run_id is None:
        raise ValueError("Give a run id, or --schedule NAME to serve a schedule's current model.")
    if _port_in_use(host, port):
        raise OSError(f"Port {port} is already in use. Choose another one with --port.")

    app, predictor = build_app(run_id, model_name=model_name, api_key=api_key, cors_origins=cors_origins or (),
                               max_mb=max_mb, access_log=access_log, schedule_name=row["name"] if row else None)
    if row is not None:
        follow_schedule(app.extensions["easyml_serve_state"], row["id"], run_id, out=out)
    shown_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    url = f"http://{shown_host}:{port}"
    print(f"\n  Serving {predictor.model_name} from run #{predictor.run_id} ({predictor.run_name})", file=out)
    print(f"  API:  {url}/predict      Help and examples: {url}/", file=out)
    if row is not None:
        print(f"  Following schedule '{row['name']}': a better model it trains is served automatically.", file=out)
    if api_key:
        print("  API key required (Authorization: Bearer <key> or X-API-Key).", file=out)
    if host in ("0.0.0.0", ""):
        print("  Listening on every network interface: anyone who can reach this computer can use it.", file=out)
    print("  Press CTRL+C to stop.\n", file=out, flush=True)

    try:
        from waitress import serve as waitress_serve
    except ImportError:  # waitress is a dependency; this only happens in unusual installs
        print("  waitress is not installed, using Flask's development server.", file=out)
        app.run(host=host, port=port, threaded=True, use_reloader=False)
        return
    waitress_serve(app, host=host, port=port, threads=max(1, int(threads)), ident="easy-ml",
                   clear_untrusted_proxy_headers=True)
