import os
import shutil
import threading

from flask import Blueprint, jsonify

from easyml import __version__, project_info
from easyml.capabilities import get_capabilities
from easyml.config.paths import DATA_DIR, DB_DIR, EDA_DIR, LOGS_DIR, RUNS_DIR, UPLOADS_DIR
from easyml.utils.database_operations import create_database_table
from easyml.utils.logger import close_all_file_handlers, reinit_file_handlers, write_log


settings_bp = Blueprint("easyml_settings", __name__)


@settings_bp.route("/settings/info", methods=["GET"])
def settings_info():
    total_size = 0
    for dirpath, _, filenames in os.walk(DATA_DIR):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)

    disk = shutil.disk_usage(DATA_DIR)
    return jsonify(
        {
            "data_dir": DATA_DIR,
            "storage_used_mb": round(total_size / (1024 * 1024), 2),
            "disk_free_mb": round(disk.free / (1024 * 1024), 2),
            "version": __version__,
        }
    )


@settings_bp.route("/settings/clear-data", methods=["DELETE"])
def settings_clear_data():
    try:
        close_all_file_handlers()

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
    return jsonify(get_capabilities())


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
            "links": project_info.visible_links(),
            "python_guide": project_info.python_guide(),
            "show_repo_links": project_info.REPO_PUBLIC,
            "developers": project_info.DEVELOPERS,
            "third_party": project_info.THIRD_PARTY,
        }
    )


@settings_bp.route("/exit", methods=["POST"])
def shutdown():
    write_log("Forced shutdown initiated via /exit", "info", os.path.basename(__file__))
    threading.Thread(target=lambda: os._exit(0)).start()
    return jsonify({"message": "Force shutdown triggered."}), 200
