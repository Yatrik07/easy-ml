from src.utils.data_loader import *
from src.config.GLOBAL_VARS import ML_RequestBody
from src.config.GLOBAL_VARS import *
from src.core.model_builder import ml_model_builder_trainer_controller
from src.core.pre_processing import preprocessing_controller
from sklearn.model_selection import train_test_split
from src.utils.logger import write_log
ERROR_CODE=5005

class MLRegressionController:
    def __init__(self, dataset_path, target_column, preprocessing=False, imputation=False, scaling=False,
                 models=None, hyper_parameter_tuning=0, categorical_columns=None, path_to_save=None,
                 reduce_features=False, task=None, ignore_columns=None):

        self.request_body = ML_RequestBody(
            dataset_path=dataset_path,
            target_column=target_column,
            preprocessing=preprocessing,
            scaling=scaling,
            models=models,
            categorical_columns=categorical_columns,
            path_to_save=path_to_save,
            hyper_parameter_tuning=hyper_parameter_tuning,
            imputation=imputation,
            reduce_features=reduce_features,
            ignore_columns=ignore_columns,
            task=task
        )

    def run(self):
        try:
            write_log(f"entered {self.__class__.__name__} -  run()", "info", os.path.basename(__file__))
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


        except Exception as e:
            if self.request_body.status_code == 200:
                self.request_body.status_code = ERROR_CODE
                self.request_body.error_message = str(e)
                self.request_body.error_traceback = traceback.format_exc()

                write_log("error occurred- {} - {}".format(
                    self.request_body.error_message, self.request_body.error_traceback),
                    "error", os.path.basename(__file__))

            raise e


class MLClassificationController:
    def __init__(self, dataset_path, target_column, preprocessing=False, imputation=False, scaling=False,
                 models=None, hyper_parameter_tuning=0, categorical_columns=None, path_to_save=None,
                 reduce_features=False, ignore_columns=None, task=None):

        self.request_body = ML_RequestBody(
            dataset_path=dataset_path,
            target_column=target_column,
            preprocessing=preprocessing,
            scaling=scaling,
            models=models,
            categorical_columns=categorical_columns,
            path_to_save=path_to_save,
            hyper_parameter_tuning=hyper_parameter_tuning,
            imputation=imputation,
            reduce_features=reduce_features,
            ignore_columns=ignore_columns,
            task=task
        )

    def run(self):

        try:

            data_loader(self.request_body)

            preprocessing_controller(self.request_body)

            ml_model_builder_trainer_controller(self.request_body)

            # print("request_body.performance: ", self.request_body.performance)
            print("request_body.performance: ", {key: value for key, value in self.request_body.dict().items() if
                   key not in ['X', 'y', 'df', 'X_train', 'X_test', 'y_train', 'y_test']})

            return self.request_body

        except Exception as e:
            if self.request_body.status_code == 200:
                self.request_body.status_code = ERROR_CODE
                self.request_body.error_message = str(e)
                self.request_body.error_traceback = traceback.format_exc()

                write_log("error occurred- {} - {}".format(
                    self.request_body.error_message, self.request_body.error_traceback),
                    "error", os.path.basename(__file__))

            raise e