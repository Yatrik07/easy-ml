from flask import Flask, jsonify, request, send_from_directory, Response, send_from_directory
import logging
from easyml.config.GLOBAL_VARS import *
from easyml import MLTrainer
from easyml.utils.data_loader import get_data_info
from datetime import datetime
import warnings, os
from easyml.utils.logger import write_log
from easyml.utils.utils import NumpyJSONProvider
from easyml.utils.database_operations import get_data_from_database, delete_run_by_id
from easyml.core.inference import MLInferenceController
from easyml.utils.database_operations import get_record_by_id
import json
warnings.filterwarnings('ignore') # , category=UserWarning -> lgbm
os.environ["LIGHTGBM_VERBOSE"] = "0"
from flask_cors import CORS
from io import StringIO
from easyml.utils.socket_emitter import init_socketio, emit_progress
import threading
import shutil

app = Flask(__name__, static_folder="frontend/dist", static_url_path="/")
app.json = NumpyJSONProvider(app)
CORS(app)
socketio = init_socketio(app)

import sys
if "--no-console" in sys.argv:
    os.environ["EASYML_NO_CONSOLE_LOGS"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # Log to console
        logging.StreamHandler(),
        # Log to file
        logging.FileHandler(os.path.join(PROJECT_ROOT, "artifacts", "logs", "main_logs.log"), encoding='utf-8')
    ]
)


@app.route('/load_data', methods=['POST'])
def load_data():
    """
    Loads a dataset either from an uploaded file or from a path in JSON.
    Returns data preview and metadata.

    - If 'file' is present in the request: processes the uploaded CSV file.
    - If 'path' is present in the JSON body: copies and loads the file from that path.

    Returns:
        JSON: Data preview and metadata, or an error response.
    """

    try:
        if 'file' in request.files:
            # Handle uploaded CSV file
            file = request.files['file']
            stream = StringIO(file.stream.read().decode("UTF8"), newline=None)
            data = pd.read_csv(stream)
            data.to_csv(save_data_path, index=False)

        elif request.is_json and 'path' in request.json:
            # Handle file loading from local file system
            source_path = request.json['path']
            destination_path = save_data_path
            print(f"source_path: {source_path}\ndestination path: {destination_path}")

            # Copy large file efficiently
            with open(source_path, 'rb') as src_file, open(destination_path, 'wb') as dst_file:
                shutil.copyfileobj(src_file, dst_file)

            data = pd.read_csv(destination_path)

        else:
            return jsonify({
                "status": "error",
                "message": "Invalid request: must include either a file upload or a JSON body with 'path'."
            }), 5001

        data_info = get_data_info(data)
        return jsonify(data_info)

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to load dataset: {str(e)}"
        }), 5001


@app.route('/get_models', methods=['GET'])
def get_models():
    """
    Returns available classification and regression model names.

    Returns:
        JSON: Dictionary of model types.
    """

    model_names = { "MLClassification": list(ui2back_mappings_classification_models.keys()),
                    "MLRegression": list(ui2back_mappings_regression_models.keys())}

    return jsonify(model_names)


@app.route('/ml_predict', methods=['POST'])
def ml_predict():
    """
    Endpoint to trigger an ML training run with preprocessing and evaluation.

    Returns:
        JSON: Model performance and metadata or error details.
    """

    print("Entered")
    write_log("new incoming request to /ml_predict", "info", os.path.basename(__file__))
    # Get data from the POST request
    data = request.get_json()
    write_log("received json data", "info", os.path.basename(__file__))

    task = data['task']
    name = data['name']
    dataset_path = save_data_path
    target_column = data['target_column']
    # preprocessing = data['preprocessing']
    preprocessing = True # assinging as this is being used in some places, but internally this has no affect, we always do preprocessing
    # imputation = data['imputation']
    imputation = True # assinging as this is being used in some places, but internally this has no affect,we alwasy do imputation.
    scaling = data['scaling']
    models = data['models']
    hyper_parameter_tuning = data['hyper_parameter_tuning']
    categorical_columns = data['categorical_columns']
    os.makedirs(os.path.join(PROJECT_ROOT, "artifacts", 'runs'), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    path_to_save = os.path.join(PROJECT_ROOT, "artifacts", 'runs', f"{name}_{timestamp}")
    print("path_to_save:", path_to_save)
    print("path_to_save:", os.path.abspath(path_to_save))
    os.makedirs(path_to_save, exist_ok=False)
    reduce_features = data['reduce_features']
    ignore_columns = data['ignore_columns']

    write_log("creating a object of MLTrainer", "info", os.path.basename(__file__))
    trainer = MLTrainer(name=name, task=task, dataset_path=dataset_path, target_column=target_column, preprocessing=preprocessing,
                        imputation=imputation, scaling=scaling, models=models, hyper_parameter_tuning=hyper_parameter_tuning,
                        categorical_columns=categorical_columns, path_to_save=path_to_save,
                        reduce_features=reduce_features, ignore_columns= ignore_columns, timestamp=timestamp )

    write_log("going for trainer.run()", "info", os.path.basename(__file__))

    try:
        output = trainer.run()

        # Return response as JSON
        return output.jsonify()

    except Exception as e:

        return {
            "status_code": 500,
            "error_message": trainer.controller.request_body.error_message,
        }


@app.route('/runs', methods=['GET', 'DELETE'])
def manage_runs():
    """
    Handles fetching or deleting training run records.

    Methods:
        GET: Returns all saved run metadata.
        DELETE: Deletes a run and its associated artifacts.

    Returns:
        JSON: Data or status message.
    """

    if request.method == 'GET':
        runs_data = get_data_from_database()
        return runs_data

    elif request.method == 'DELETE':
        data = request.get_json()
        run_id = data.get('run_id')

        if run_id is None:
            return jsonify({"error": "Missing run_id"}), 400

        success = delete_run_by_id(run_id)

        if success:
            return jsonify({"message": f"Run with ID {run_id} deleted successfully."}), 200
        else:
            return jsonify({"error": f"Run with ID {run_id} not found or could not be deleted."}), 404

    # ✅ fallback in case another method sneaks in
    return jsonify({"error": "Unsupported request method"}), 405


@app.route("/get_run_data", methods=['GET'])
def get_run_data():
    """
    Returns metadata and column info for a specific run ID.

    Returns:
        JSON: Run configuration, column data, and list of trained models.
    """

    run_id = request.args['run_id']
    result = get_record_by_id(record_id=run_id)
    if "status_code" not in result:
        result['status_code'] = 200
    status_code = result.pop('status_code')
    if status_code == 200:
        data_path = result['folder_path']
        column_data = json.load(open(os.path.join(data_path, "column_data.json"), 'r'))
        config_data = json.load(open(os.path.join(data_path, "config_json.json"), 'r'))
        result['column_data'] = column_data
        result['models'] = list(config_data['performance'].keys())
        return result, status_code
    else:
        return jsonify(result), status_code


@app.route('/inference', methods=['POST'])
def inference():
    """
    Performs inference on a single data point using a trained model.

    Returns:
        JSON: Prediction and probabilities or error.
    """

    try:
        write_log("new incoming request to /inference", "info", os.path.basename(__file__))
        # Get data from the POST request
        json_data = request.get_json(force=True)
        write_log("received json data", "info", os.path.basename(__file__))

        # Extract top-level fields
        run_id = int(json_data['run_id'])
        model_name = json_data['model_name']
        input_data = json_data['data']

        # Create controller and get inference
        controller = MLInferenceController(run_id=run_id, data=input_data, model_name=model_name)
        predictions = controller.get_inference()

        return predictions

    except Exception as e:

        return {
            "status_code": 500,
            "error_message": controller.inference_body.error_message,
        }


@app.route('/debug/logs', methods=['GET'])
def debug_logs():
    """
    Returns contents of the main log file for debugging.

    Returns:
        Response: Text/plain log content.
    """

    try:
        log_file_path = os.path.join(PROJECT_ROOT, "artifacts", "logs", "main_logs.log")

        if not os.path.exists(log_file_path):
            return Response("Log file not found.", status=404, mimetype='text/plain')

        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()

        return Response(log_content, mimetype='text/plain')

    except Exception as e:
        error_msg = f"Error reading log file: {str(e)}"
        write_log(error_msg, "error", os.path.basename(__file__))
        return Response(error_msg, status=500, mimetype='text/plain')


@app.route('/exit', methods=['POST'])
def shutdown():
    """
    Shuts down the Flask development server (for local use only).

    Returns:
        JSON: Confirmation message.
    """

    write_log("Forced shutdown initiated via /exit", "info", os.path.basename(__file__))
    threading.Thread(target=lambda: os._exit(0)).start()
    return jsonify({"message": "Force shutdown triggered."}), 200

# Serve Vue frontend
@app.route("/")
def serve_index():
    """
    Serves the Vue app's index.html.
    """

    print("in /")
    return send_from_directory(app.static_folder, "index.html")


# Catch-all route for Vue.js routing
@app.route("/<path:path>")
def serve_vue_routes(path):
    """
    Serves static assets or falls back to index.html for Vue routing.
    """

    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")


def main():
    """
    Starts the Flask app with SocketIO.
    """

    # from easyml.app import app
    # write_log("🚀 easy-ml server started successfully.", "info", "app.py")
    # app.run(port=5010, threaded=True)


    from easyml.utils.socket_emitter import init_socketio
    socketio = init_socketio(app)
    write_log("🚀 easy-ml server (with WebSocket) started successfully.", "info", "app.py")
    socketio.run(app, port=5010, debug=True)


if __name__ == '__main__':
    # from easyml.app import app
    # app.run(port=5010)

    from easyml.utils.socket_emitter import init_socketio
    socketio = init_socketio(app)
    write_log("🚀 easy-ml server (with WebSocket) started successfully.", "info", "app.py")
    socketio.run(app, port=5010, debug=True)