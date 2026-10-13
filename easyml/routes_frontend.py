import os

from flask import Blueprint, current_app, send_from_directory

frontend_bp = Blueprint("easyml_frontend", __name__)


def _dist_dir():
    return current_app.config["EASYML_FRONTEND_DIR"]


@frontend_bp.route("/")
def serve_index():
    return send_from_directory(_dist_dir(), "index.html")


@frontend_bp.route("/favicon.ico")
def serve_favicon():
    return send_from_directory(os.path.join(_dist_dir(), "static"), "favicon.svg", mimetype="image/svg+xml")
