from flask import Flask, jsonify, request
from src.config.GLOBAL_VARS import *
from src import MLTrainer
from src.utils.data_loader import get_data_info
from flask_cors import CORS
from io import StringIO

app = Flask(__name__)
CORS(app)

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
    # Get data from the POST request
    data = request.get_json()

    task = data['task']
    dataset_path = save_data_path
    target_column = data['target_column']
    preprocessing = data['preprocessing']
    imputation = data['imputation']
    scaling = data['scaling']
    models = data['models']
    hyper_parameter_tuning = data['hyper_parameter_tuning']
    categorical_columns = data['categorical_columns']
    path_to_save = data['path_to_save']
    reduce_features = data['reduce_features']
    handle_imbalanced_data = data['handle_imbalanced_data']
    ignore_columns = data['ignore_columns']

    trainer = MLTrainer(task=task, dataset_path=dataset_path, target_column=target_column, preprocessing=preprocessing,
                        imputation=imputation, scaling=scaling, models=models, hyper_parameter_tuning=hyper_parameter_tuning,
                        categorical_columns=categorical_columns, path_to_save=path_to_save,
                        reduce_features=reduce_features, ignore_columns= ignore_columns, handle_imbalanced_data=handle_imbalanced_data)

    output = trainer.run()

    # Return response as JSON
    return output.jsonify()


# Run the Flask app
if __name__ == '__main__':
    app.run(port=5001)
