import sklearn
import os
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, r2_score
from src.config.GLOBAL_VARS import *
from src.utils.logger import get_logger

logger = get_logger()


def ml_model_builder_trainer_controller(request_body):

    models_to_apply = request_body.models if request_body.models else \
        DEFAULT_ML_MODELS_LIST_FOR_REGRESSION if "regression" in request_body.task.lower() else \
        DEFAULT_ML_MODELS_LIST_FOR_CLASSIFICATION if "classification" in request_body.task.lower() else \
        None

    for single_model in models_to_apply:

        grid = {} if request_body.hyper_parameter_tuning == 0 else \
            globals().get(
                f"PARAM_GRID_LEVEL{str(request_body.hyper_parameter_tuning)}_FOR_{'REGRESSION' if 'regression' in request_body.task.lower() else 'CLASSIFICATION'}",
                None)

        # grid_search = GridSearchCV(estimator=single_model, param_grid=grid[single_model.__class__.__name__]
        #   if grid else grid, cv=4, scoring='accuracy', n_jobs=-1)
        scoring = 'accuracy' if 'classification' in request_body.task.lower() else 'r2'
        grid_search = GridSearchCV(estimator=single_model,
                                   param_grid=grid[single_model.__class__.__name__] if grid else grid, cv=4,
                                   scoring=scoring, n_jobs=-1)

        grid_search.fit(request_body.X_train, request_body.y_train)

        if "classification" in request_body.task.lower():
            accuracy_train, accuracy_test = accuracy_score( request_body.y_train, grid_search.predict(request_body.X_train) ), \
                accuracy_score( request_body.y_test, grid_search.predict( request_body.X_test ) )
        elif "regression" in request_body.task.lower():
            accuracy_train, accuracy_test = r2_score( request_body.y_train, grid_search.predict(request_body.X_train) ), \
                r2_score( request_body.y_test, grid_search.predict( request_body.X_test ) )

        model_path = os.path.join(request_body.path_to_save, f"{single_model.__class__.__name__}.pkl")
        joblib.dump(single_model, model_path )

        request_body.performance[single_model.__class__.__name__] = {"model_path": model_path,
                                                                     "accuracy_train": accuracy_train,
                                                                     "accuracy_test": accuracy_test
                                                                     }
        logger.info(f'Model {single_model.__class__.__name__} done')

