from easyml.utils.data_loader import *
from easyml.core.model_builder import ml_model_builder_trainer_controller
from easyml.core.pre_processing import preprocessing_controller
from sklearn.model_selection import train_test_split
from easyml.utils.database_operations import *
from datetime import datetime
from easyml.utils.socket_emitter import emit_progress
from easyml.utils.utils import NumpyEncoder


class MLRegressionController:
    """
    Controller class for managing regression ML pipelines.

    This includes loading data, optional preprocessing, training models,
    and storing run metadata in a database.
    """

    def __init__(self, name, dataset_path, target_column,timestamp,  preprocessing=False, imputation=False, scaling=False,
                 models=None, hyper_parameter_tuning=0, categorical_columns=None, path_to_save=None,
                 reduce_features=False, task=None, ignore_columns=None):
        """
        Initializes the MLRegressionController with user-defined configuration.

        Args:
            name (str): Name of the run.
            dataset_path (str): Path to the dataset file.
            target_column (str): Name of the target column.
            timestamp (str): Timestamp of the run.
            preprocessing (bool): Whether to perform preprocessing.
            imputation (bool): Whether to apply missing value imputation.
            scaling (bool): Whether to scale numerical features.
            models (list): List of model names to train.
            hyper_parameter_tuning (int): Whether to perform HPT (0/1).
            categorical_columns (list): List of categorical feature names.
            path_to_save (str): Output directory path to save results.
            reduce_features (bool): Whether to apply feature reduction.
            task (str): Task type (e.g., 'Regression').
            ignore_columns (list): Columns to ignore during training.
        """

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
            write_log(f"entered {self.__class__.__name__} -  run()", "info", os.path.basename(__file__))

            create_database_table("runs")
            emit_progress("⚙️ Initializing pipeline...", step=1, total_steps=5)

            data_loader(self.request_body)
            emit_progress("📂 Data loaded and validated", step=2, total_steps=5)

            if self.request_body.preprocessing:
                preprocessing_controller(self.request_body)
            else:
                self.request_body.X_train, self.request_body.X_test, self.request_body.y_train, self.request_body.y_test = \
                    train_test_split(self.request_body.X, self.request_body.y, test_size=0.2)
            emit_progress("🔧 Preprocessing & feature engineering complete", step=3, total_steps=5)

            emit_progress("🤖 Training models (this may take a while)...", step=4, total_steps=5)
            ml_model_builder_trainer_controller(self.request_body)

            enter_record_to_table("runs", **{"run_name": self.request_body.name,
                                                    "folder_path": self.request_body.path_to_save,
                                                    "timestamp": self.request_body.timestamp.replace(" ", "T"),
                                                    "task": self.request_body.task,
                                                    "dataset_name": getattr(self.request_body, 'original_dataset_name', os.path.basename(self.request_body.dataset_path or "")),
                                                    "metrics_json": json.dumps(self.request_body.performance, cls=NumpyEncoder)
                                             }
                                  )
            emit_progress("💾 Results saved to database", step=4, total_steps=5)

            runs_data = get_data_from_database("runs")
            if runs_data.get("status_code") == 200 and len(runs_data.get("data", [])) > 0:
                self.request_body.run_id = runs_data["data"][-1]["id"]
            else:
                self.request_body.run_id = -1

            response_payload = {
                "status_code": 200,
                "message": "Regression pipeline completed successfully",
                "run_id": self.request_body.run_id,
                "performance": self.request_body.performance
            }

            emit_progress("✅ Regression pipeline complete!", step=5, total_steps=5, final=True, result=response_payload)
            return self.request_body

        except Exception as e:
            if self.request_body.status_code == 200:
                self.request_body.error_message = "OtherError"
                self.request_body.status_code = error_code_mappings[self.request_body.error_message]
                self.request_body.error_traceback = traceback.format_exc()
                write_log("error occurred- {} - {}".format(
                    self.request_body.error_message, self.request_body.error_traceback),
                    "error", os.path.basename(__file__))
            emit_progress(f"❌ Error: {str(e)}", pct=100, final=True,
                          result={"status": "error", "error": str(e), "message": str(e)})
            raise e


class MLClassificationController:
    """
    Controller class for managing classification ML pipelines.

    This includes loading data, preprocessing, model training,
    and saving results to a database.
    """

    def __init__(self, name, dataset_path, target_column, timestamp, preprocessing=False, imputation=False, scaling=False,
                 models=None, hyper_parameter_tuning=0, categorical_columns=None, path_to_save=None,
                 reduce_features=False, ignore_columns=None, task=None):
        """
        Initializes the MLClassificationController with the required configuration.

        Args:
            name (str): Name of the run.
            dataset_path (str): Path to the dataset file.
            target_column (str): Name of the target column.
            timestamp (str): Timestamp of the run.
            preprocessing (bool): Whether to perform preprocessing.
            imputation (bool): Whether to apply missing value imputation.
            scaling (bool): Whether to scale numerical features.
            models (list): List of model names to train.
            hyper_parameter_tuning (int): Whether to perform HPT (0/1).
            categorical_columns (list): List of categorical feature names.
            path_to_save (str): Output directory path to save results.
            reduce_features (bool): Whether to apply feature reduction.
            ignore_columns (list): Columns to ignore during training.
            task (str): Task type (e.g., 'Classification').
        """

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
            write_log(f"entered {self.__class__.__name__} -  run()", "info", os.path.basename(__file__))

            create_database_table("runs")
            emit_progress("⚙️ Initializing pipeline...", step=1, total_steps=5)

            data_loader(self.request_body)
            emit_progress("📂 Data loaded and validated", step=2, total_steps=5)

            preprocessing_controller(self.request_body)
            emit_progress("🔧 Preprocessing & feature engineering complete", step=3, total_steps=5)

            emit_progress("🤖 Training models (this may take a while)...", step=4, total_steps=5)
            ml_model_builder_trainer_controller(self.request_body)

            enter_record_to_table("runs", **{"run_name": self.request_body.name,
                                                    "folder_path": self.request_body.path_to_save,
                                                    "timestamp": self.request_body.timestamp.replace(" ", "T"),
                                                    "task": self.request_body.task,
                                                    "dataset_name": getattr(self.request_body, 'original_dataset_name', os.path.basename(self.request_body.dataset_path or "")),
                                                    "metrics_json": json.dumps(self.request_body.performance, cls=NumpyEncoder)
                                             }
                                  )
            emit_progress("💾 Results saved to database", step=4, total_steps=5)

            runs_data = get_data_from_database("runs")
            if runs_data.get("status_code") == 200 and len(runs_data.get("data", [])) > 0:
                self.request_body.run_id = runs_data["data"][-1]["id"]
            else:
                self.request_body.run_id = -1

            response_payload = {
                "status_code": 200,
                "message": "Classification pipeline completed successfully",
                "run_id": self.request_body.run_id,
                "performance": self.request_body.performance
            }

            emit_progress("✅ Classification pipeline complete!", step=5, total_steps=5, final=True, result=response_payload)
            return self.request_body

        except Exception as e:
            if self.request_body.status_code == 200:
                self.request_body.error_message = "OtherError"
                self.request_body.status_code = error_code_mappings[self.request_body.error_message]
                self.request_body.error_traceback = traceback.format_exc()
                write_log("error occurred- {} - {}".format(
                    self.request_body.error_message, self.request_body.error_traceback),
                    "error", os.path.basename(__file__))
            emit_progress(f"❌ Error: {str(e)}", pct=100, final=True,
                          result={"status": "error", "error": str(e), "message": str(e)})
            raise e