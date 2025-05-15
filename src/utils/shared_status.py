import json
import threading
import os
from datetime import datetime

STATUS_FILE = "progress_status.json"
_status_lock = threading.Lock()

# Append progress messages to a list and persist
progress_log = []


# def update_progress(msg: str):
#     global progress_log
#     with _status_lock:
#         entry = {
#             "message": msg,
#             "timestamp": datetime.now().isoformat()
#         }
#         progress_log.append(entry)
#         with open(STATUS_FILE, "w") as f:
#             json.dump(progress_log, f)
#
def update_progress(msg: str, final: bool = False):
    global progress_log
    with _status_lock:
        entry = {
            "message": msg,
            "timestamp": datetime.now().isoformat()
        }
        if final:
            entry["done"] = True
        progress_log.append(entry)
        with open(STATUS_FILE, "w") as f:
            json.dump(progress_log, f)


def get_progress():
    try:
        with _status_lock:
            if not os.path.exists(STATUS_FILE):
                return []
            with open(STATUS_FILE, "r") as f:
                content = f.read().strip()
                if not content:
                    return []
                return json.loads(content)
    except Exception as e:
        return [{"message": f"[Error reading progress] {e}", "timestamp": None}]
