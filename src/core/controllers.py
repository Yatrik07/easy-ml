# entry point
from src.utils.data_loader import *
from src.config.GLOBAL_VARS import ML_RequestBody
from src.config.GLOBAL_VARS import  *
from src.core.model_builder import ml_model_builder_trainer_controller
from src.core.pre_processing import preprocessing_controller
from sklearn.model_selection import train_test_split

class MLRegressionController:
    def __init__(self, dataset_path, target_column, preprocessing=False, imputation=False, scaling=False,
                 models=None, hyper_parameter_tuning=0, verbose=False, categorical_columns=None, path_to_save=None,
                 reduce_features=False, handle_imbalanced_data=False, task=None):

        self.request_body = ML_RequestBody(
            dataset_path=dataset_path,
            target_column=target_column,
            preprocessing=preprocessing,
            scaling=scaling,
            models=models,
            categorical_columns=categorical_columns,
            path_to_save=path_to_save,
            hyper_parameter_tuning=hyper_parameter_tuning,
            verbose=verbose,
            imputation=imputation,
            reduce_features=reduce_features,
            handle_imbalanced_data=handle_imbalanced_data,
            task=task
        )

    def run(self):
        # if self.request_body.dataset_path and type(self.request_body.dataset_path) == str:
        data_loader(self.request_body)

        if self.request_body.preprocessing:
            preprocessing_controller(self.request_body)

        else:
            self.request_body.X_train, self.request_body.X_test, self.request_body.y_train, self.request_body.y_test = \
                train_test_split(self.request_body.X, self.request_body.y, test_size=0.2)

        ml_model_builder_trainer_controller(self.request_body)

        # print("request_body.performance: ", self.request_body)
        print("request_body.performance: ", {key: value for key, value in self.request_body.dict().items() if
               key not in ['X', 'y', 'df', 'X_train', 'X_test', 'y_train', 'y_test']})

        return self.request_body


class MLClassificationController:
    def __init__(self, dataset_path, target_column, preprocessing=False, imputation=False, scaling=False,
                 models=None, hyper_parameter_tuning=0, verbose=False, categorical_columns=None, path_to_save=None,
                 reduce_features=False, handle_imbalanced_data=None, task=None):

        self.request_body = ML_RequestBody(
            dataset_path=dataset_path,
            target_column=target_column,
            preprocessing=preprocessing,
            scaling=scaling,
            models=models,
            categorical_columns=categorical_columns,
            path_to_save=path_to_save,
            hyper_parameter_tuning=hyper_parameter_tuning,
            verbose=verbose,
            imputation=imputation,
            reduce_features=reduce_features,
            handle_imbalanced_data=handle_imbalanced_data,
            task=task
        )

    def run(self):
        print("Here")
        data_loader(self.request_body)

        if self.request_body.preprocessing:
            preprocessing_controller(self.request_body)

        ml_model_builder_trainer_controller(self.request_body)

        # print("request_body.performance: ", self.request_body.performance)
        print("request_body.performance: ", {key: value for key, value in self.request_body.dict().items() if
               key not in ['X', 'y', 'df', 'X_train', 'X_test', 'y_train', 'y_test']})

        return self.request_body