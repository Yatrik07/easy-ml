"""
Command-line entry point: ``easy-ml`` (or ``python -m easyml``).

    easy-ml                      start the studio on http://127.0.0.1:5010 and open the browser
    easy-ml --port 8080 --no-browser
    easy-ml download-models      fetch the spaCy English model used for lemmatization
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

    subcommands = parser.add_subparsers(dest="command")
    subcommands.add_parser("download-models", help="Download the spaCy English model used for NLP lemmatization.")
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


def run_server(host=DEFAULT_HOST, port=DEFAULT_PORT, open_browser=True, console_logs=True, debug=False):
    # Must be decided before the logger creates its handlers.
    if console_logs:
        os.environ.setdefault("EASYML_CONSOLE_LOG_LEVEL", "INFO")
    else:
        os.environ["EASYML_NO_CONSOLE_LOGS"] = "1"

    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    from easyml.app_factory import create_app
    from easyml.utils.logger import write_log

    if _port_in_use(host, port):
        print(f"Port {port} is already in use. Is Easy-ML already running? Try: easy-ml --port {port + 1}")
        return 1

    app = create_app()
    socketio = app.extensions["easyml_socketio"]

    display_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    url = f"http://{display_host}:{port}"
    print(f"\n  Easy-ML Studio is running at {url}\n  Press CTRL+C to stop.\n")
    write_log(f"easy-ml server started on {host}:{port}", "info", "app.py")

    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    socketio.run(app, host=host, port=port, debug=debug, use_reloader=False, allow_unsafe_werkzeug=True)
    return 0


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.command == "download-models":
        return download_models()
    return run_server(
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
        console_logs=not args.no_console,
        debug=args.debug,
    )


if __name__ == "__main__":
    sys.exit(main())
