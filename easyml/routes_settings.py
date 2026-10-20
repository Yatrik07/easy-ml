import os
import shutil
import sys
import threading

from flask import Blueprint, jsonify, request

from easyml import __version__, project_info
from easyml.capabilities import get_capabilities
from easyml.config.paths import DATA_DIR, DB_DIR, EDA_DIR, LOGS_DIR, RUNS_DIR, UPLOADS_DIR
from easyml.utils.database_operations import create_database_table
from easyml.utils.logger import close_all_file_handlers, reinit_file_handlers, write_log


settings_bp = Blueprint("easyml_settings", __name__)


def _directory_size(root):
    """
    Total size of the files under ``root``, counted like ``os.walk``: links to files count their
    target, broken links and unreadable entries are skipped, linked folders are not entered.
    """
    total, pending = 0, [root]
    while pending:
        try:
            entries = list(os.scandir(pending.pop()))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir():
                    if not entry.is_symlink():
                        pending.append(entry.path)
                    continue
                total += entry.stat().st_size
            except OSError:
                continue
    return total


@settings_bp.route("/settings/info", methods=["GET"])
def settings_info():
    total_size = _directory_size(DATA_DIR)

    disk = shutil.disk_usage(DATA_DIR)
    from easyml.utils.storage import uploads_usage

    return jsonify(
        {
            "data_dir": DATA_DIR,
            "storage_used_mb": round(total_size / (1024 * 1024), 2),
            "disk_free_mb": round(disk.free / (1024 * 1024), 2),
            "version": __version__,
            "uploads": uploads_usage(folder=UPLOADS_DIR),
        }
    )


@settings_bp.route("/settings/clean-uploads", methods=["POST"])
def settings_clean_uploads():
    """Delete uploads older than ``older_than_hours`` (default 24); every run keeps its own copy of its data."""
    from easyml.utils.storage import clean_uploads

    body = request.get_json(silent=True) or {}
    try:
        hours = float(body.get("older_than_hours", 24))
    except (TypeError, ValueError):
        hours = -1
    if hours < 1:
        return jsonify({"status": "error", "status_code": 400,
                        "error_message": "older_than_hours must be a number of at least 1."}), 400
    result = clean_uploads(older_than_hours=hours, folder=UPLOADS_DIR)
    write_log(f"Deleted {result['deleted']} old upload(s), {result['freed_bytes']} bytes", "info", os.path.basename(__file__))
    return jsonify(result)


@settings_bp.route("/settings/clear-data", methods=["DELETE"])
def settings_clear_data():
    try:
        close_all_file_handlers()
        from easyml.core.model_cache import clear as clear_model_cache

        clear_model_cache()

        for directory in [DB_DIR, RUNS_DIR, UPLOADS_DIR, LOGS_DIR, EDA_DIR]:
            if os.path.exists(directory):
                shutil.rmtree(directory, ignore_errors=True)
                os.makedirs(directory, exist_ok=True)

        reinit_file_handlers()
        create_database_table("runs")
        write_log("All data cleared by user request", "info", os.path.basename(__file__))
        return jsonify({"message": "All data cleared successfully.", "data_dir": DATA_DIR}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@settings_bp.route("/settings/capabilities", methods=["GET"])
def settings_capabilities():
    from easyml.core.column_weights import WEIGHT_EFFECT_BY_KEY, weight_effects_by_class

    # How much each tabular model follows input column weights (0.14): by key (Configure) and by class (Results).
    return jsonify({**get_capabilities(), "weight_effects": WEIGHT_EFFECT_BY_KEY,
                    "weight_effects_by_class": weight_effects_by_class()})


@settings_bp.route("/settings/device", methods=["GET"])
def settings_device():
    cuda_available = False
    cuda_name = ""
    torch_version = "Not installed"
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_name = torch.cuda.get_device_name(0)
        elif "+cpu" in torch_version:
            cuda_name = "CPU-only PyTorch installed"
        else:
            cuda_name = "CUDA not found by PyTorch"
    except (ImportError, OSError):
        cuda_name = "PyTorch not installed"

    return jsonify(
        {
            "cuda_available": cuda_available,
            "cuda_device_name": cuda_name,
            "torch_version": torch_version,
            "device": "cuda" if cuda_available else "cpu",
            "cuda_install_hint": project_info.CUDA_INSTALL_HINT,
            "help_url": project_info.CUDA_HELP_URL,
        }
    )


def _environment():
    """Short facts for a bug report: version, operating system, Python and optional libraries (no paths)."""
    import platform

    from easyml.capabilities import installed_optional_libraries

    extras = installed_optional_libraries()
    system, release = platform.system(), platform.release()
    if system == "Windows" and release == "10" and getattr(sys, "getwindowsversion", None)             and sys.getwindowsversion().build >= 22000:
        release = "11"  # Python before 3.12 reports Windows 11 as "10"
    return {
        "version": __version__,
        "os": f"{system} {release}".strip(),
        "python": platform.python_version(),
        "extras": ", ".join(extras) if extras else "none (basic install)",
    }


@settings_bp.route("/settings/credits", methods=["GET"])
def settings_credits():
    return jsonify(
        {
            "project": project_info.PROJECT_NAME,
            "version": __version__,
            "summary": project_info.SUMMARY,
            "license": project_info.LICENSE,
            "contact_email": project_info.CONTACT_EMAIL,
            "contact": project_info.contact_links(),
            "links": project_info.visible_links(_environment()),
            "python_guide": project_info.python_guide(),
            "show_repo_links": project_info.REPO_PUBLIC,
            "contributing_url": project_info.contributing_url(),
            "developers": project_info.DEVELOPERS,
            "third_party": project_info.THIRD_PARTY,
        }
    )


@settings_bp.route("/exit", methods=["POST"])
def shutdown():
    write_log("Forced shutdown initiated via /exit", "info", os.path.basename(__file__))
    threading.Thread(target=lambda: os._exit(0)).start()
    return jsonify({"message": "Force shutdown triggered."}), 200
