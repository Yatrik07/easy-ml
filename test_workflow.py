import requests
import socketio
import time
import os

BASE_URL = "http://127.0.0.1:5010"
sio = socketio.Client()

progress_events = []

@sio.event
def connect():
    print("[WS] Connected to server")

@sio.event
def disconnect():
    print("[WS] Disconnected from server")

@sio.on('progress')
def on_progress(data):
    print(f"[WS Progress] {data['pct']}% - {data['message']}")
    progress_events.append(data)

def run_test():
    try:
        sio.connect(BASE_URL)
    except Exception as e:
        print("Failed to connect to WS:", e)
        return

    # First load some data. The app requires dataset_path for tabular ML.
    # We will use sample data if available. Let's create a dummy dataset.
    import pandas as pd
    import numpy as np

    df = pd.DataFrame({
        "feature1": np.random.rand(100),
        "feature2": np.random.rand(100),
        "target": np.random.randint(0, 2, 100)
    })
    
    os.makedirs("test_folder", exist_ok=True)
    dataset_path = os.path.abspath("test_folder/dummy_data.csv")
    df.to_csv(dataset_path, index=False)
    
    payload = {
        "task": "MLClassification",
        "name": "test_script_run",
        "dataset_path": dataset_path,
        "target_column": "target",
        "models": ["Random Forest", "Logistic Regression"],
        "hyper_parameter_tuning": 0,
        "categorical_columns": [],
        "ignore_columns": [],
        "scaling": "No",
        "reduce_features": "No"
    }

    print(f"Sending POST request to {BASE_URL}/ml_predict...")
    response = requests.post(f"{BASE_URL}/ml_predict", json=payload)
    print("API Response:", response.status_code, response.json())

    # Wait for the background task to complete and emit progress
    print("Waiting for training to complete via WS...")
    timeout = 30
    start = time.time()
    while time.time() - start < timeout:
        if len(progress_events) > 0 and progress_events[-1].get("done"):
            print("Training finished successfully.")
            break
        time.sleep(0.5)
    else:
        print("Timeout waiting for final WS event.")

    sio.disconnect()

if __name__ == "__main__":
    run_test()
