from src import MLTrainer
from flask import Flask, request

app = Flask(__name__)


@app.route('/ml_predict', methods=['POST'])
def ml_predict():
    # Get data from the POST request
    data = request.get_json()

    task = data['task']
    dataset_path = data['dataset_path']
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

    trainer = MLTrainer(task=task, dataset_path=dataset_path, target_column=target_column, preprocessing=preprocessing,
                        imputation=imputation, scaling=scaling, models=models, hyper_parameter_tuning=hyper_parameter_tuning,
                        categorical_columns=categorical_columns, path_to_save=path_to_save,
                        reduce_features=reduce_features, handle_imbalanced_data=handle_imbalanced_data)

    output = trainer.run()

    # Return response as JSON
    return output.jsonify()


# Run the Flask app
if __name__ == '__main__':
    app.run()
