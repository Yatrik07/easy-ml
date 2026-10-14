import os

from flask import Blueprint, Response, current_app, send_from_directory

from easyml import __version__

frontend_bp = Blueprint("easyml_frontend", __name__)


def _dist_dir():
    return current_app.config["EASYML_FRONTEND_DIR"]


def _use_dev_vue():
    """The development build of Vue prints template warnings; use it while working on the UI."""
    return current_app.debug or os.environ.get("EASYML_FRONTEND_DEBUG") == "1"


@frontend_bp.route("/")
def serve_index():
    with open(os.path.join(_dist_dir(), "index.html"), "r", encoding="utf-8") as handle:
        html = handle.read()
    html = html.replace("__EASYML_VERSION__", __version__)
    html = html.replace("__VUE_BUILD__", "vue.global.js" if _use_dev_vue() else "vue.global.prod.js")
    response = Response(html, mimetype="text/html")
    response.headers["Cache-Control"] = "no-cache"
    return response


@frontend_bp.route("/favicon.ico")
def serve_favicon():
    return send_from_directory(os.path.join(_dist_dir(), "static"), "favicon.svg", mimetype="image/svg+xml")
