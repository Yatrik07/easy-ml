import mimetypes
import os
from urllib.parse import urlsplit

from flask import Flask, jsonify, request

from easyml.config.paths import DATA_DIR, DB_DIR, EDA_DIR, LOGS_DIR, PACKAGE_ROOT, RUNS_DIR, UPLOADS_DIR
from easyml.routes_core import core_bp
from easyml.routes_cv import cv_bp
from easyml.routes_frontend import frontend_bp
from easyml.routes_nlp import nlp_bp
from easyml.routes_runs import runs_bp
from easyml.routes_settings import settings_bp
from easyml.utils.database_operations import create_database_table
from easyml.utils.socket_emitter import init_socketio
from easyml.utils.utils import NumpyJSONProvider

FRONTEND_DIR = os.path.join(PACKAGE_ROOT, "unified_frontend", "dist")
mimetypes.add_type("font/woff2", ".woff2")
# Windows can map .js to text/plain in the registry; browsers refuse to run ES modules served that way.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def _parse_allowed_origins(allowed_origins):
    if allowed_origins is None:
        allowed_origins = [o.strip() for o in os.environ.get("EASYML_ALLOWED_ORIGINS", "").split(",") if o.strip()]
    return [o.rstrip("/") for o in allowed_origins]


def create_app(allowed_origins=None):
    """
    Build the Easy-ML Flask application.

    Args:
        allowed_origins (list, optional): Extra browser origins (e.g. ``http://localhost:5173``) allowed
            to call the API and open the progress socket. Defaults to the ``EASYML_ALLOWED_ORIGINS``
            environment variable (comma-separated). The page served by this app is always allowed.
    """
    os.environ.setdefault("LIGHTGBM_VERBOSE", "0")
    extra_origins = _parse_allowed_origins(allowed_origins)

    app = Flask(__name__, static_folder=os.path.join(FRONTEND_DIR, "static"), static_url_path="/static")
    app.json = NumpyJSONProvider(app)
    app.config.update(
        EASYML_DATA_DIR=DATA_DIR,
        EASYML_DB_DIR=DB_DIR,
        EASYML_RUNS_DIR=RUNS_DIR,
        EASYML_LOGS_DIR=LOGS_DIR,
        EASYML_UPLOADS_DIR=UPLOADS_DIR,
        EASYML_EDA_DIR=EDA_DIR,
        EASYML_FRONTEND_DIR=FRONTEND_DIR,
        EASYML_ALLOWED_ORIGINS=extra_origins,
    )

    @app.before_request
    def reject_cross_site_writes():
        """
        Browsers attach an Origin header to cross-site requests. Refusing state-changing requests
        from other sites stops any web page from training, deleting data or shutting the server down.
        Requests without an Origin (curl, Python clients, tests) are not affected.
        """
        if request.method not in UNSAFE_METHODS:
            return None
        origin = request.headers.get("Origin")
        if not origin:
            return None
        if urlsplit(origin).netloc == request.host or origin.rstrip("/") in app.config["EASYML_ALLOWED_ORIGINS"]:
            return None
        return jsonify({"status_code": 403, "error_message": "Cross-origin requests are not allowed."}), 403

    @app.after_request
    def allow_configured_origins(response):
        origin = request.headers.get("Origin")
        if origin and origin.rstrip("/") in app.config["EASYML_ALLOWED_ORIGINS"]:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
            response.headers["Vary"] = "Origin"
        return response

    def socket_origin_allowed(origin, environ=None):
        if not origin:
            return True
        host = (environ or {}).get("HTTP_HOST", "")
        return urlsplit(origin).netloc == host or origin.rstrip("/") in extra_origins

    socketio = init_socketio(app, allowed_origins=socket_origin_allowed)
    app.extensions["easyml_socketio"] = socketio

    create_database_table("runs")

    app.register_blueprint(core_bp)
    app.register_blueprint(cv_bp)
    app.register_blueprint(nlp_bp)
    app.register_blueprint(runs_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(frontend_bp)

    return app
