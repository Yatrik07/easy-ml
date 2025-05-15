from flask import Flask, jsonify, request
import logging
from src.config.GLOBAL_VARS import *
from src import MLTrainer
from src.utils.data_loader import get_data_info
from datetime import datetime
import warnings, os
from src.utils.logger import write_log
from src.utils.shared_status import update_progress
warnings.filterwarnings('ignore') # , category=UserWarning -> lgbm
os.environ["LIGHTGBM_VERBOSE"] = "0"
app = Flask(__name__)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # Log to console
        logging.StreamHandler(),
        # Log to file
        logging.FileHandler(r'logs\main_logs.log', encoding='utf-8')
    ]
)

@app.route('/load_data', methods=['POST'])
def load_data():
    """
    This endpoint loads the data from the given path and returns a
    data sample, head and tail along with columns, shape and size.
    """
    req = request.get_json()
    data_info = get_data_info(req["data_path"])
    return jsonify(data_info)


@app.route('/get_models', methods=['GET'])
def get_models():
    model_names = { "MLClassification": list(ui2back_mappings_classification_models.keys()),
                    "MLRegression": list(ui2back_mappings_regression_models.keys())}

    return jsonify(model_names)

# @app.route('/ml_predict', methods=['POST'])
# def ml_predict():
#     print("Entered")
#     write_log("new incoming request to /ml_predict", "info", os.path.basename(__file__))
#     update_progress("new incoming request to /ml_predict")
#     # Get data from the POST request
#     data = request.get_json()
#     write_log("received json data", "info", os.path.basename(__file__))
#
#     task = data['task']
#     name = data['name']
#     dataset_path = data['dataset_path']
#     target_column = data['target_column']
#     # preprocessing = data['preprocessing']
#     preprocessing = True # assinging as this is being used in some places, but internally this has no affect, we always do preprocessing
#     # imputation = data['imputation']
#     imputation = True # assinging as this is being used in some places, but internally this has no affect,we alwasy do imputation.
#     scaling = data['scaling']
#     models = data['models']
#     hyper_parameter_tuning = data['hyper_parameter_tuning']
#     categorical_columns = data['categorical_columns']
#     os.makedirs('runs', exist_ok=True)
#     path_to_save = os.path.join('runs', f"{name}_{datetime.now().strftime('%Y_%m_%d %H%M%S')}")
#     os.makedirs(path_to_save, exist_ok=False)
#     reduce_features = data['reduce_features']
#     ignore_columns = data['ignore_columns']
#
#     write_log("creating a object of MLTrainer", "info", os.path.basename(__file__))
#     trainer = MLTrainer(task=task, dataset_path=dataset_path, target_column=target_column, preprocessing=preprocessing,
#                         imputation=imputation, scaling=scaling, models=models, hyper_parameter_tuning=hyper_parameter_tuning,
#                         categorical_columns=categorical_columns, path_to_save=path_to_save,
#                         reduce_features=reduce_features, ignore_columns= ignore_columns)
#
#     write_log("going for trainer.run()", "info", os.path.basename(__file__))
#
#     try:
#         output = trainer.run()
#
#         # Return response as JSON
#         return output.jsonify()
#
#     except Exception as e:
#
#         return {
#             "status_code": 500,
#             "error_message": trainer.controller.request_body.error_message,
#         }

from multiprocessing import Process, Queue

def background_training(trainer_data, result_queue):
    from src import MLTrainer
    try:
        trainer = MLTrainer(**trainer_data)
        output = trainer.run()
        result_queue.put(("success", output.jsonify()))
    except Exception as e:
        error_msg = "[Unhandled Exception]"
        try:
            error_msg = trainer.controller.request_body.error_message
        except:
            import traceback
            error_msg = traceback.format_exc()
        result_queue.put(("error", {"status_code": 500, "error_message": error_msg}))

@app.route('/ml_predict', methods=['POST'])
def ml_predict():
    write_log("new incoming request to /ml_predict", "info", os.path.basename(__file__))
    update_progress("new incoming request to /ml_predict")

    data = request.get_json()

    task = data['task']
    name = data['name']
    dataset_path = data['dataset_path']
    target_column = data['target_column']
    preprocessing = True
    imputation = True
    scaling = data['scaling']
    models = data['models']
    hyper_parameter_tuning = data['hyper_parameter_tuning']
    categorical_columns = data['categorical_columns']
    os.makedirs('runs', exist_ok=True)
    path_to_save = os.path.join('runs', f"{name}_{datetime.now().strftime('%Y_%m_%d %H%M%S')}")
    os.makedirs(path_to_save, exist_ok=False)
    reduce_features = data['reduce_features']
    ignore_columns = data['ignore_columns']

    trainer_data = dict(
        task=task,
        dataset_path=dataset_path,
        target_column=target_column,
        preprocessing=preprocessing,
        imputation=imputation,
        scaling=scaling,
        models=models,
        hyper_parameter_tuning=hyper_parameter_tuning,
        categorical_columns=categorical_columns,
        path_to_save=path_to_save,
        reduce_features=reduce_features,
        ignore_columns=ignore_columns
    )

    result_queue = Queue()

    try:
        process = Process(target=background_training, args=(trainer_data, result_queue))
        process.start()

        # BLOCKING WAIT for training result
        status, payload = result_queue.get()  # <-- waits until the background process puts result

        if status == "success":
            return jsonify(payload)  # already in jsonify() form
        else:
            return jsonify(payload)

    except Exception as e:
        return jsonify({
            "status_code": 500,
            "error_message": str(e)
        })


# Run the Flask app
if __name__ == '__main__':
    app.run(port=5001, debug=False, use_reloader=False)
