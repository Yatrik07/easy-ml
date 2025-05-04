import sklearn
import os
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, r2_score
import tempfile
import shutil
from sklearn.preprocessing import LabelEncoder
from src.config.GLOBAL_VARS import *
import tempfile
import shutil


#new
def ml_model_builder_trainer_controller(request_body):
    task = "classification" if "classification" in request_body.task.lower() else "regression" if "regression" in request_body.task.lower() else None

    mappings = ui2back_mappings_regression_models if task == "regression" else ui2back_mappings_classification_models if task else None

    models_to_apply = [mappings[model] for model in request_body.models if model in mappings] if request_body.models else \
        DEFAULT_ML_MODELS_LIST_FOR_REGRESSION if task == "regression" else \
        DEFAULT_ML_MODELS_LIST_FOR_CLASSIFICATION if task == "classification" else \
        None

    grid = {} if request_body.hyper_parameter_tuning == 0 else \
        globals().get(
            f"PARAM_GRID_LEVEL{str(request_body.hyper_parameter_tuning)}_FOR_"
            f"{'REGRESSION' if task == 'regression' else 'CLASSIFICATION'}",
            {}
        )

    scoring = 'accuracy' if 'classification' in request_body.task.lower() else 'r2'

    for single_model in models_to_apply:

        print(type(single_model).__name__)

        if isinstance(single_model, (CatBoostClassifier, CatBoostRegressor)):
            temp_dir = tempfile.mkdtemp()
            print(temp_dir)
            single_model.set_params(train_dir=temp_dir)
            single_model.set_params(thread_count=1)
            single_model.set_params(allow_writing_files=False)
            grid_search = GridSearchCV(estimator=single_model,
                                       param_grid=grid[type(single_model).__name__] if grid and type(
                                           single_model).__name__ in grid else {},
                                       cv=4,
                                       scoring=scoring, n_jobs=-1)
            grid_search.fit(request_body.X_train, request_body.y_train)
            shutil.rmtree(temp_dir, ignore_errors=True)

        else:
            grid_search = GridSearchCV(estimator=single_model,
                                       param_grid=grid[type(single_model).__name__] if grid and type(
                                           single_model).__name__ in grid else {},
                                       cv=4,
                                       scoring=scoring, n_jobs=-1)

            grid_search.fit(request_body.X_train, request_body.y_train)

        if task == "classification":
            accuracy_train, accuracy_test = accuracy_score( request_body.y_train, grid_search.predict(request_body.X_train) ), \
                accuracy_score( request_body.y_test, grid_search.predict( request_body.X_test ) )
        elif task == "regression":
            accuracy_train, accuracy_test = r2_score( request_body.y_train, grid_search.predict(request_body.X_train) ), \
                r2_score( request_body.y_test, grid_search.predict( request_body.X_test ) )

        # model_path = os.path.join(request_body.path_to_save, f"{single_model._class.name_}.pkl")
        model_path = os.path.join(request_body.path_to_save, f"{type(single_model).__name__}.pkl")

        if isinstance(single_model, (CatBoostClassifier, CatBoostRegressor)):
            grid_search.best_estimator_.save_model(f"{model_path}.cbm")
        else:
            joblib.dump(grid_search.best_estimator_, model_path, protocol=5)

        request_body.performance[type(single_model).__name__] = {"model_path": model_path,
                                                                     "accuracy_train": accuracy_train,
                                                                     "accuracy_test": accuracy_test
                                                                     }

