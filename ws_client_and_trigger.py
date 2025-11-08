import requests
import json
import threading
import time
from socketio import Client

FLASK_HOST = "http://localhost:5010"
LOAD_DATA_ENDPOINT = f"{FLASK_HOST}/load_data"
ML_PREDICT_ENDPOINT = f"{FLASK_HOST}/ml_predict"
WS_URL = FLASK_HOST  # Socket.IO default path is `/socket.io`

sio = Client()

# ======== WebSocket Events ========
@sio.event
def connect():
    print("[WS] ✅ Connected to WebSocket server")

@sio.event
def disconnect():
    print("[WS] ❌ Disconnected from WebSocket server")

@sio.on("progress")
def on_progress(data):
    ts = data.get("timestamp", "")
    msg = data.get("message", "")
    print(f"[Progress] {ts} - {msg}")
    if data.get("done"):
        print("[WS] 🎉 Training marked as complete.")


# ======== Upload CSV File First ========
def upload_dataset(file_path):
    print("[HTTP] Uploading dataset to /load_data...")
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            res = requests.post(LOAD_DATA_ENDPOINT, files=files)
            res.raise_for_status()
            print("[HTTP] ✅ Dataset uploaded successfully.")
            # print("[HTTP] Data Summary:", json.dumps(res.json(), indent=2))
    except Exception as e:
        print(f"[HTTP] ❌ Dataset upload failed: {e}")
        exit(1)


# ======== Trigger /ml_predict ========
def trigger_training(task):
    regression_payload = {
    "task" : "MLRegression",
    "name": "regression_run_1",
    "dataset_path": "D:\\PycharmProjects\\easy-ml\\test_folder\\Housing.csv",
    "target_column": "price",
    "scaling" : True,
    "models": ["linear_regression", "ada_boost", "random_forest", "cat_boost"],
    "hyper_parameter_tuning": 1,
    "categorical_columns" : [],
    "reduce_features": True,
    "ignore_columns": []
}

    classification_payload =  {
    "task" : "MLClassification",
    "name": "clf_run_1",
    "dataset_path": "D:\\PycharmProjects\\easy-ml\\test_folder\\diabetes.csv",
    "target_column": "Outcome",
    "scaling" : True,
    "models": ["linear_regression", "ada_boost", "random_forest", "cat_boost"],
    "hyper_parameter_tuning": 1,
    "categorical_columns" : [],
    "reduce_features": True,
    "ignore_columns": []
}



    try:
        print("[HTTP] Sending training request to /ml_predict...")
        if task == "r":
            res = requests.post(ML_PREDICT_ENDPOINT, json=regression_payload)
        elif task == "c":
            res = requests.post(ML_PREDICT_ENDPOINT, json=classification_payload)


        print(f"[HTTP] Status: {res.status_code}")
        print("[HTTP] Initial Response:", json.dumps(res.json(), indent=2))
    except Exception as e:
        print(f"[HTTP] ❌ Training request failed: {e}")
        exit(1)


# ======== MAIN ENTRY ========
if __name__ == "__main__":

    task = "c"

    r_dataset_path = r"D:\PycharmProjects\easy-ml\test_folder\Housing.csv"  # 🔁 change if needed
    c_dataset_path = r"D:\PycharmProjects\easy-ml\test_folder\diabetes.csv"

    # Step 1: Upload dataset
    if task=="r":
        upload_dataset(r_dataset_path)
    elif task=="c":
        upload_dataset(c_dataset_path)


    # Step 2: Connect to WebSocket
    sio.connect(WS_URL)

    # Step 3: Trigger training (non-blocking)
    trigger_thread = threading.Thread(target=trigger_training(task))
    trigger_thread.start()

    # Step 4: Keep alive until training completes
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        sio.disconnect()
