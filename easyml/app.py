from flask import Flask, jsonify, request
import logging
from easyml.config.GLOBAL_VARS import *
from easyml import MLTrainer
from easyml.utils.data_loader import get_data_info
from datetime import datetime
import warnings, os
from easyml.utils.logger import write_log
from easyml.utils.database_operations import get_data_from_database, delete_run_by_id
from easyml.core.inference import MLInferenceController
warnings.filterwarnings('ignore') # , category=UserWarning -> lgbm
os.environ["LIGHTGBM_VERBOSE"] = "0"
from flask_cors import CORS
from io import StringIO

app = Flask(__name__)
CORS(app)

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
    This endpoint loads the data from the given path and returns a
    data sample, head and tail along with columns, shape and size.
    """
    file = request.files['file']
    stream = StringIO(file.stream.read().decode("UTF8"), newline=None)
    data = pd.read_csv(stream)
    data.to_csv(save_data_path, index=False)
    data_info = get_data_info(data)
    return jsonify(data_info)


@app.route('/get_models', methods=['GET'])
def get_models():
    model_names = { "MLClassification": list(ui2back_mappings_classification_models.keys()),
                    "MLRegression": list(ui2back_mappings_regression_models.keys())}

    return jsonify(model_names)

@app.route('/ml_predict', methods=['POST'])
def ml_predict():
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

@app.route('/inference', methods=['POST'])
def inference():
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

# Run the Flask app

def main():
    from easyml.app import app
    write_log("🚀 easy-ml server started successfully.", "info", "app.py")
    app.run(port=5001, threaded=True)

if __name__ == '__main__':
    from easyml.app import app
    app.run(port=5001)
