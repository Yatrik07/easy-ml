"""
The prediction service: a small Flask app around one :class:`easyml.predictor.Predictor`.

    GET  /          a page with the model's inputs and copy-paste examples
    GET  /health    {"status": "ok"}
    GET  /info      the model, its task, inputs and an example
    POST /predict   one row, many rows, text(s) or image(s)  (see ``predict``)

Request bodies (``POST /predict``):

    tabular  {"features": {"age": 45, "city": "NY"}}          -> {"prediction": ..., "confidence": ...}
             {"rows": [{...}, {...}]}                          -> {"predictions": [...], "confidences": [...]}
             (+ "warnings": [...] when the input does not look like the training data)
    text     {"text": "great phone"}  /  {"texts": ["...", "..."]}
    image    multipart "file" (one or several)  /  {"image_base64": "..."}  /  {"images_base64": [...]}

Regression answers have no confidence, exactly like the exported API.
"""

import base64
import binascii
import hmac
import io
import threading

from flask import Flask, jsonify, request

from easyml.utils.utils import NumpyJSONProvider

OPEN_PATHS = {"/", "/health"}  # never need the API key


class ServeState:
    """The predictor being served, swappable at runtime (a schedule's new champion)."""

    def __init__(self, predictor, info):
        self._lock = threading.Lock()
        self._predictor = predictor
        self._info = info

    def current(self):
        with self._lock:
            return self._predictor, self._info

    def swap(self, predictor, info):
        with self._lock:
            self._predictor, self._info = predictor, info


RAW_DTYPES = {"int": "int64", "float": "float64", "bool": "bool", "datetime": "datetime64[ns]"}


class BadRequest(ValueError):
    """The request body does not have a shape this model accepts."""


def model_info(predictor):
    """What /info and the docs page show: task, model, inputs and an example request."""
    from easyml import __version__
    from easyml.core.artifacts import load_target_decoder, read_json
    from easyml.utils.run_summary import model_label, task_label

    info = {
        "run_id": predictor.run_id,
        "run_name": predictor.run_name,
        "task": predictor.task,
        "task_label": task_label(predictor.task),
        "model": predictor.model_name,
        "model_label": model_label(predictor.model_name),
        "available_models": predictor.available_models,
        "easyml_version": __version__,
        "classes": None,
        "inputs": {},
        "example": {},
    }
    folder = predictor._folder
    if predictor.task in ("MLClassification", "MLRegression"):
        from easyml.core.artifacts import load_frozen_recipe
        from easyml.utils.export_utils import example_features

        prep = load_frozen_recipe(folder)
        if prep is not None:
            # Trained on cleaned data: callers send the original columns, the service cleans them.
            kinds = prep.data.get("raw_kinds") or {}
            columns = {c: RAW_DTYPES.get(kinds.get(c), "object") for c in prep.input_columns}
            info["inputs"] = {"kind": "features", "columns": columns,
                              "cleaning": {"recipe": prep.name, "steps": prep.summary()["steps"]}}
        else:
            columns = read_json(folder, "column_data.json").get("column_dtypes") or {}
            info["inputs"] = {"kind": "features", "columns": columns}
        info["example"] = {"features": example_features(folder)}
        if predictor.task == "MLClassification":
            decoder = load_target_decoder(folder)
            info["classes"] = [decoder[i] for i in sorted(decoder)]
    elif predictor.is_text:
        config = read_json(folder, "config_json.json")
        info["inputs"] = {"kind": "text", "text_column": config.get("text_column")}
        info["example"] = {"text": "Your text to classify"}
        info["classes"] = config.get("target_labels") or list((config.get("target_encodings") or {}))
    else:
        config = read_json(folder, "config_json.json")
        info["inputs"] = {"kind": "image", "image_size": config.get("image_size")}
        info["example"] = {"image_base64": "<base64 of a .jpg or .png>"}
        info["classes"] = config.get("classes")
    return info


def _single(predictions, confidences, with_confidence):
    body = {"prediction": predictions[0]}
    if with_confidence:
        body["confidence"] = confidences[0] if confidences else None
    return body


def _many(predictions, confidences, with_confidence):
    body = {"predictions": predictions}
    if with_confidence:
        body["confidences"] = confidences if confidences else [None] * len(predictions)
    return body


def _decode_base64(text):
    if not isinstance(text, str) or not text:
        raise BadRequest("image_base64 must be a non-empty base64 string.")
    if text.startswith("data:") and "," in text:
        text = text.split(",", 1)[1]
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise BadRequest("image_base64 is not valid base64.") from exc


def run_prediction(predictor, body, files):
    """Turn one request into predictions. Returns the JSON-ready answer; raises BadRequest on bad input."""
    with_confidence = predictor.task != "MLRegression"

    if predictor.is_image:
        if files:
            sources = [io.BytesIO(f.read()) for f in files]
            single = len(sources) == 1
        elif isinstance(body, dict) and "image_base64" in body:
            sources, single = [io.BytesIO(_decode_base64(body["image_base64"]))], True
        elif isinstance(body, dict) and isinstance(body.get("images_base64"), list):
            sources, single = [io.BytesIO(_decode_base64(t)) for t in body["images_base64"]], False
        else:
            raise BadRequest('Send an image as multipart "file", or JSON {"image_base64": "..."} '
                             'or {"images_base64": [...]}.')
        if not sources:
            raise BadRequest("No image was sent.")
        predictions, confidences = predictor.predict_with_confidence(sources)

    elif predictor.is_text:
        if isinstance(body, dict) and isinstance(body.get("text"), str):
            texts, single = [body["text"]], True
        elif isinstance(body, dict) and isinstance(body.get("texts"), list) and body["texts"]:
            texts, single = [str(t) for t in body["texts"]], False
        else:
            raise BadRequest('Send JSON {"text": "..."} or {"texts": ["...", "..."]}.')
        predictions, confidences = predictor.predict_with_confidence(texts)

    else:
        if isinstance(body, dict) and isinstance(body.get("features"), dict):
            rows, single = [body["features"]], True
        elif isinstance(body, dict) and isinstance(body.get("rows"), list) and body["rows"] \
                and all(isinstance(r, dict) for r in body["rows"]):
            rows, single = body["rows"], False
        else:
            raise BadRequest('Send JSON {"features": {"column": value, ...}} for one row, '
                             'or {"rows": [{...}, {...}]} for several.')
        predictions, confidences, notes = predictor.predict_with_notes(rows)
        answer = (_single if single else _many)(predictions, confidences, with_confidence)
        warnings = notes + predictor.data_warnings(rows)
        if warnings:  # only when there is something to say, so ordinary answers stay as they were
            answer["warnings"] = warnings
        return answer

    return (_single if single else _many)(predictions, confidences, with_confidence)


def create_serve_app(predictor, api_key=None, cors_origins=(), max_mb=20, access_log=False, state=None):
    """The Flask app serving ``predictor`` (or ``state``, whose predictor can be swapped)."""
    from easyml.serving.docs_page import render_docs

    state = state or ServeState(predictor, model_info(predictor))
    app = Flask("easyml.serve")
    app.json = NumpyJSONProvider(app)
    app.config["MAX_CONTENT_LENGTH"] = int(max_mb * 1024 * 1024) if max_mb else None
    app.extensions["easyml_serve_state"] = state
    allowed = {origin.rstrip("/") for origin in (cors_origins or ())}

    def error(status, message):
        return jsonify({"error": message}), status

    @app.before_request
    def check_key():
        if request.method == "OPTIONS" or not api_key or request.path in OPEN_PATHS:
            return None
        sent = request.headers.get("X-API-Key") or ""
        auth = request.headers.get("Authorization") or ""
        if auth.lower().startswith("bearer "):
            sent = sent or auth[7:].strip()
        if not hmac.compare_digest(sent.encode(), api_key.encode()):
            return error(401, "Missing or wrong API key. Send it as 'Authorization: Bearer <key>' or 'X-API-Key'.")
        return None

    @app.after_request
    def cors_and_log(response):
        origin = (request.headers.get("Origin") or "").rstrip("/")
        if origin and ("*" in allowed or origin in allowed):
            response.headers["Access-Control-Allow-Origin"] = "*" if "*" in allowed else origin
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-API-Key"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Vary"] = "Origin"
        if access_log:
            print(f"{request.remote_addr} {request.method} {request.path} {response.status_code}", flush=True)
        return response

    @app.get("/")
    def docs():
        _, info = state.current()
        return render_docs(info, request.host_url.rstrip("/"), needs_key=bool(api_key))

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/info")
    def info():
        return jsonify(state.current()[1])

    @app.route("/predict", methods=["POST", "OPTIONS"])
    def predict():
        if request.method == "OPTIONS":
            return "", 204
        current, _ = state.current()
        body = request.get_json(silent=True) if request.is_json else None
        files = request.files.getlist("file") if request.files else []
        if body is None and not files:
            if request.is_json and request.get_data():
                return error(400, "The request body is not valid JSON. (In Windows PowerShell, curl.exe loses the "
                                  "double quotes inside -d '...'; use Invoke-RestMethod, or a file: -d @body.json)")
            return error(400, "Send a JSON body (Content-Type: application/json) or an image file.")
        try:
            return jsonify(run_prediction(current, body, files))
        except BadRequest as exc:
            return error(400, str(exc))
        except (ValueError, TypeError, KeyError) as exc:
            return error(400, f"Could not predict: {exc}")
        except Exception as exc:  # unexpected: report without taking the service down
            from easyml.utils.logger import write_log

            write_log(f"serve: prediction failed: {exc!r}", "error", "serving.server")
            return error(500, "Prediction failed on the server. See the Easy-ML error log.")

    @app.errorhandler(404)
    def not_found(_):
        return error(404, "Not found. Endpoints: GET /, GET /health, GET /info, POST /predict.")

    @app.errorhandler(405)
    def wrong_method(_):
        return error(405, "Method not allowed for this endpoint.")

    @app.errorhandler(413)
    def too_large(_):
        return error(413, f"Request is larger than the {max_mb} MB limit (--max-mb).")

    return app
