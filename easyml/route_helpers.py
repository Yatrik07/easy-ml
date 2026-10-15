import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime
from io import BytesIO, StringIO

import pandas as pd
from flask import current_app, jsonify, request

from easyml.config.paths import DATA_DIR, RUNS_DIR, UPLOADS_DIR
from easyml.utils.cancellation import active_jobs, register_job, reset_current_job, set_current_job
from easyml.utils.database_operations import RunNotFoundError, folder_has_run
from easyml.utils.logger import write_log
from easyml.utils.socket_emitter import emit_progress
from easyml.utils.utils import clean_display_filename, sanitize_run_name  # noqa: F401 (re-exported)

TEMP_UPLOADS_DIR = os.path.join(tempfile.gettempdir(), "easy-ml", "uploads")


def ensure_writable_dir(preferred_dir):
    candidates = [
        preferred_dir,
        TEMP_UPLOADS_DIR,
    ]

    for candidate in candidates:
        try:
            os.makedirs(candidate, exist_ok=True)
            probe_path = os.path.join(candidate, ".write_test")
            with open(probe_path, "a", encoding="utf-8"):
                pass
            try:
                os.remove(probe_path)
            except OSError:
                pass
            return candidate
        except OSError:
            continue

    raise PermissionError("No writable upload directory is available.")


def _is_within(path, root):
    try:
        return os.path.commonpath([os.path.realpath(path), os.path.realpath(root)]) == os.path.realpath(root)
    except ValueError:  # different drives on Windows
        return False


def resolve_data_file(path):
    """
    Return ``path`` if it points to an existing file inside the Easy-ML data directory.

    Training and preview endpoints only ever receive paths that /load_data produced, so anything
    outside the data directory is rejected instead of being read from an arbitrary location.
    """
    if not path or not isinstance(path, str):
        raise ValueError("A dataset path is required.")
    allowed_roots = [current_app.config.get("EASYML_DATA_DIR", DATA_DIR), TEMP_UPLOADS_DIR]
    if not any(_is_within(path, root) for root in allowed_roots):
        raise PermissionError("Dataset path must come from an upload made through Easy-ML.")
    if not os.path.isfile(path):
        raise FileNotFoundError("The uploaded dataset no longer exists. Please upload it again.")
    return path


def create_run_folder(name, timestamp_format="%Y-%m-%d %H-%M-%S"):
    """Create a unique folder for a run and return ``(path, timestamp)``."""
    timestamp = datetime.now().strftime(timestamp_format)
    runs_dir = current_app.config.get("EASYML_RUNS_DIR", RUNS_DIR)
    base = os.path.join(runs_dir, f"{sanitize_run_name(name)}_{timestamp}")
    path = base
    while os.path.exists(path):
        path = f"{base}_{uuid.uuid4().hex[:4]}"
    os.makedirs(path)
    return path, timestamp


def _read_uploaded_table(uploaded_file):
    raw_bytes = uploaded_file.stream.read()
    filename = (uploaded_file.filename or "").lower()

    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(BytesIO(raw_bytes))

    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return pd.read_csv(StringIO(raw_bytes.decode(encoding), newline=None))
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    return pd.read_csv(StringIO(raw_bytes.decode("utf-8", errors="replace"), newline=None))


def save_dataframe_request(file_prefix="data"):
    """
    Persist a CSV/XLSX request payload into the uploads directory and return a DataFrame + path.

    Accepts a multipart ``file`` upload, or a JSON body ``{"path": ...}`` pointing to a local
    CSV/XLSX file (handy when scripting against the local server).
    """

    upload_dir = ensure_writable_dir(current_app.config.get("EASYML_UPLOADS_DIR", UPLOADS_DIR))
    unique_save_path = os.path.join(upload_dir, f"{file_prefix}_{uuid.uuid4().hex[:8]}.csv")

    if "file" in request.files:
        data = _read_uploaded_table(request.files["file"])
        data.to_csv(unique_save_path, index=False)
        return data, unique_save_path

    if request.is_json and "path" in request.json:
        source_path = request.json["path"]
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"File not found: {source_path}")
        if source_path.lower().endswith((".xlsx", ".xls")):
            data = pd.read_excel(source_path)
            data.to_csv(unique_save_path, index=False)
            return data, unique_save_path
        shutil.copyfile(source_path, unique_save_path)
        return pd.read_csv(unique_save_path), unique_save_path

    raise ValueError("Invalid request: must include either a file upload or a JSON body with 'path'.")


def uploaded_original_filename():
    """Original file name of the dataset in the current /load_data request."""
    if "file" in request.files:
        return clean_display_filename(request.files["file"].filename)
    if request.is_json and isinstance(request.json, dict):
        return clean_display_filename(request.json.get("path"))
    return None


def load_class_distribution(dataset_path, target_column):
    path = resolve_data_file(dataset_path)
    if target_column not in pd.read_csv(path, nrows=0).columns:
        raise ValueError(f"Column '{target_column}' not found in dataset")

    target = pd.read_csv(path, usecols=[target_column])[target_column].dropna()
    counts = target.value_counts()
    total = len(target)
    distribution = [
        {
            "class": str(label),
            "count": int(count),
            "percentage": round((count / total) * 100, 1),
        }
        for label, count in counts.items()
    ]

    return {
        "target_column": target_column,
        "total_samples": total,
        "num_classes": len(counts),
        "distribution": distribution,
        "is_imbalanced": bool(len(counts) and (counts.max() / counts.min()) > 3),
    }


def start_background_task(fn):
    socketio = current_app.extensions["easyml_socketio"]
    socketio.start_background_task(fn)


def training_in_progress():
    return bool(active_jobs())


def remove_orphan_run_folder(path):
    """Delete a run folder this server created if no database record points at it."""
    if not path or not os.path.isdir(path):
        return
    try:
        if folder_has_run(path):
            return
        shutil.rmtree(path, ignore_errors=True)
        write_log(f"Removed run folder of unfinished run: {path}", "info", os.path.basename(__file__))
    except Exception as exc:
        write_log(f"Could not remove run folder {path}: {exc}", "warning", os.path.basename(__file__))


def start_training_job(train_fn, label, run_folder=None):
    """
    Run ``train_fn`` in the background as its own job and return the job.

    Progress events go only to clients that joined the job's Socket.IO room, and
    ``/stop_training`` cancels this job alone. Successful pipelines emit their own final "done"
    event with the results payload; on failure or cancellation exactly one final event is emitted
    here and ``run_folder`` (created for this run) is removed unless the run reached the database.
    """
    job = register_job(label)

    def runner():
        token = set_current_job(job)
        try:
            train_fn()
        except InterruptedError:
            write_log(f"{label} training cancelled by user", "info", os.path.basename(__file__))
            remove_orphan_run_folder(run_folder)
            emit_progress("⚠️ Training cancelled by user", pct=100, final=True,
                          result={"status": "cancelled", "message": "Training was cancelled by the user."})
        except Exception as exc:
            write_log(f"{label} training failed: {exc}", "error", os.path.basename(__file__))
            remove_orphan_run_folder(run_folder)
            emit_progress(f"❌ Error: {exc}", pct=100, final=True,
                          result={"status": "error", "error": str(exc), "message": str(exc)})
        finally:
            job.finished = True
            reset_current_job(token)

    start_background_task(runner)
    return job


def error_response(exc, default_status=500):
    """Map an exception to a JSON error response with a meaningful HTTP status."""
    if isinstance(exc, RunNotFoundError):
        status = 404
    elif isinstance(exc, PermissionError):
        status = 403
    elif isinstance(exc, (FileNotFoundError,)):
        status = 404
    elif isinstance(exc, (ValueError, KeyError, TypeError)):
        status = 400
    else:
        status = default_status
    message = str(exc)
    if isinstance(exc, KeyError):
        message = f"Missing field: {exc.args[0]}"
    return jsonify({"status": "error", "status_code": status, "error_message": message, "message": message}), status


def optional_import_error_response(feature_name, install_hint):
    return jsonify(
        {
            "status_code": 503,
            "error_message": f"{feature_name} is not available in this installation.",
            "hint": install_hint,
        }
    ), 503


def parse_metrics_json(metrics_raw):
    if isinstance(metrics_raw, str):
        try:
            return json.loads(metrics_raw)
        except Exception:
            return {}
    if isinstance(metrics_raw, dict):
        return metrics_raw
    return {}
