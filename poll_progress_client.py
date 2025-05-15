import time
import requests
import threading
import os
import json
from datetime import datetime

FLASK_API_URL = "http://localhost:5001/ml_predict"
PROGRESS_API_URL = "http://localhost:5002/get_progress"
PROGRESS_LOG_FILE = "progress_status.json"

# Update to match the 'name' used in your request
RUN_NAME = "cli_run"
RUN_TIMESTAMP = datetime.now().strftime('%Y_%m_%d %H%M%S')
RUN_DIR = os.path.join("runs", f"{RUN_NAME}_{RUN_TIMESTAMP}")
FINAL_OUTPUT_FILE = os.path.join(RUN_DIR, "final_output.json")

# Clear progress log
if os.path.exists(PROGRESS_LOG_FILE):
    with open(PROGRESS_LOG_FILE, "w") as f:
        f.write("[]")

# 1. Function to trigger model training
def send_training_request():
    payload = {
        "task": "MLClassification",
        "name": RUN_NAME,
        "dataset_path": r"D:\PycharmProjects\easy-ml\test_folder\diabetes.csv",
        "target_column": "Outcome",
        "scaling": True,
        "models": ["random_forest", "xg_boost", "knn", "logistic_regression", "ada_boost", "cat_boost"],
        "hyper_parameter_tuning": 1,
        "categorical_columns": [],
        "reduce_features": True,
        "ignore_columns": []
    }

    print("[HTTP] Sending training request to /ml_predict...")
    try:
        res = requests.post(FLASK_API_URL, json=payload)
        print(f"[HTTP] Status: {res.status_code}")
        print("[HTTP] Initial Response:")
        print(json.dumps(res.json(), indent=2))
    except Exception as e:
        print(f"[ERROR] Could not send training request: {e}")

# 2. Polling progress every 1 second
def poll_progress():
    seen_messages = set()
    while True:
        try:
            res = requests.get(PROGRESS_API_URL)
            if res.headers.get("Content-Type", "").startswith("application/json"):
                logs = res.json()
                if isinstance(logs, list):
                    for entry in logs:
                        if isinstance(entry, dict):
                            msg = entry.get("message")
                            ts = entry.get("timestamp")
                            key = f"{ts}:{msg}"
                            if key not in seen_messages:
                                print(f"[Progress] {ts} - {msg}")
                                seen_messages.add(key)
                            if entry.get("done") is True:
                                print("[Client] Training marked as complete. Stopping polling.")
                                return
                else:
                    print(f"[ERROR] Unexpected JSON structure: {logs}")
            else:
                print(f"[ERROR] Non-JSON response received: {res.text.strip()}")
        except Exception as e:
            print(f"[ERROR] Could not fetch progress: {e}")
        time.sleep(1) # ----> every N second sending update request (every 1 second here)

# 3. Load and print final result JSON
def print_final_result():
    if os.path.exists(FINAL_OUTPUT_FILE):
        print("\n[Client] Final Training Result:")
        with open(FINAL_OUTPUT_FILE, "r") as f:
            result = json.load(f)
            print(json.dumps(result, indent=2))
    else:
        print(f"\n[Client] No final output file found at {FINAL_OUTPUT_FILE}")

# Main execution
if __name__ == '__main__':
    poll_thread = threading.Thread(target=poll_progress)
    poll_thread.start()

    send_training_request()

    poll_thread.join()
    print("[Client] Training process ended.")

    # Print final result from disk
    print_final_result()

