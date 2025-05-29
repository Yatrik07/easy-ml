from src.utils.data_loader import *
from easyml.core.model_builder import ml_model_builder_trainer_controller
from easyml.core.pre_processing import preprocessing_controller
from sklearn.model_selection import train_test_split
from src.utils.database_operations import *
from datetime import datetime

class MLRegressionController:
    def __init__(self, name, dataset_path, target_column,timestamp,  preprocessing=False, imputation=False, scaling=False,
                 models=None, hyper_parameter_tuning=0, categorical_columns=None, path_to_save=None,
                 reduce_features=False, task=None, ignore_columns=None):

        self.request_body = ML_RequestBody(
            name=name,
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
            task=task,
            timestamp=timestamp
        )

    def run(self):
        try:

            create_database_table("runs")

            write_log(f"entered {self.__class__.__name__} -  run()", "info", os.path.basename(__file__))
            data_loader(self.request_body)

            if self.request_body.preprocessing:
                preprocessing_controller(self.request_body)

            else:
                self.request_body.X_train, self.request_body.X_test, self.request_body.y_train, self.request_body.y_test = \
                    train_test_split(self.request_body.X, self.request_body.y, test_size=0.2)

            ml_model_builder_trainer_controller(self.request_body)

            enter_record_to_table("runs", **{"run_name": self.request_body.name,
                                                    "folder_path": self.request_body.path_to_save,
                                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                    "task": self.request_body.task
                                             }
                                  )

            # print("request_body.performance: ", self.request_body)
            print("request_body.performance: ", {key: value for key, value in self.request_body.dict().items() if
                   key not in ['X', 'y', 'df', 'X_train', 'X_test', 'y_train', 'y_test']})

            return self.request_body


        except Exception as e:
            if self.request_body.status_code == 200:
                self.request_body.error_message = "OtherError"
                self.request_body.status_code = error_code_mappings[self.request_body.error_message]
                self.request_body.error_traceback = traceback.format_exc()

                write_log("error occurred- {} - {}".format(
                    self.request_body.error_message, self.request_body.error_traceback),
                    "error", os.path.basename(__file__))

            raise e


class MLClassificationController:
    def __init__(self, name, dataset_path, target_column, timestamp, preprocessing=False, imputation=False, scaling=False,
                 models=None, hyper_parameter_tuning=0, categorical_columns=None, path_to_save=None,
                 reduce_features=False, ignore_columns=None, task=None):

        self.request_body = ML_RequestBody(
            name=name,
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
            task=task,
            timestamp=timestamp
        )

    def run(self):

        try:

            create_database_table("runs")

            data_loader(self.request_body)

            preprocessing_controller(self.request_body)

            ml_model_builder_trainer_controller(self.request_body)

            enter_record_to_table("runs", **{"run_name": self.request_body.name,
                                                    "folder_path": self.request_body.path_to_save,
                                                    "timestamp": self.request_body.timestamp,
                                                    "task": self.request_body.task
                                             }
                                  )

            # print("request_body.performance: ", self.request_body.performance)
            print("request_body.performance: ", {key: value for key, value in self.request_body.dict().items() if
                   key not in ['X', 'y', 'df', 'X_train', 'X_test', 'y_train', 'y_test']})

            return self.request_body

        except Exception as e:
            if self.request_body.status_code == 200:
                self.request_body.error_message = "otherError"
                self.request_body.status_code = error_code_mappings[self.request_body.error_message]
                self.request_body.error_traceback = traceback.format_exc()

                write_log("error occurred- {} - {}".format(
                    self.request_body.error_message, self.request_body.error_traceback),
                    "error", os.path.basename(__file__))

            raise e