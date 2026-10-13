import json
import os
import traceback

from easyml.config.GLOBAL_VARS import ML_RequestBody, error_code_mappings
from easyml.core.model_builder import ml_model_builder_trainer_controller
from easyml.core.pre_processing import preprocessing_controller
from easyml.utils.data_loader import data_loader
from easyml.utils.database_operations import create_database_table, enter_record_to_table
from easyml.utils.logger import write_log
from easyml.utils.socket_emitter import emit_progress
from easyml.utils.utils import NumpyEncoder


class _TabularController:
    """
    Shared tabular pipeline: load data, preprocess, train models, and store run metadata in the database.

    On success the final progress event is emitted here. Failures (including user cancellation)
    are re-raised so the caller emits exactly one final error/cancelled event.
    """

    label = ""

    def __init__(self, name, dataset_path, target_column, timestamp, preprocessing=True, imputation=False, scaling=False,
                 models=None, hyper_parameter_tuning=0, categorical_columns=None, path_to_save=None,
                 reduce_features=False, ignore_columns=None, task=None, imbalance_handling="none", **kwargs):
        """
        Args:
            name (str): Name of the run.
            dataset_path (str): Path to the dataset file.
            target_column (str): Name of the target column.
            timestamp (str): Timestamp of the run.
            preprocessing (bool): Whether to perform preprocessing.
            imputation (bool): Whether to apply missing value imputation.
            scaling (bool): Whether to scale numerical features.
            models (list): List of model names to train.
            hyper_parameter_tuning (int): Grid search level (0 = none).
            categorical_columns (list): List of categorical feature names.
            path_to_save (str): Output directory path to save results.
            reduce_features (bool): Whether to apply feature reduction.
            ignore_columns (list): Columns to ignore during training.
            task (str): Task type (e.g., 'MLClassification').
            imbalance_handling (str): Resampling method for classification ("none", "smote", ...).
        """

        self.request_body = ML_RequestBody(
            name=name,
            dataset_path=dataset_path,
            target_column=target_column,
            # Inference always loads preprocessing_pipeline.pkl, so the pipeline is always built.
            preprocessing=True,
            scaling=scaling,
            models=models or [],
            categorical_columns=categorical_columns or [],
            path_to_save=path_to_save,
            hyper_parameter_tuning=hyper_parameter_tuning,
            imputation=imputation,
            reduce_features=reduce_features,
            ignore_columns=ignore_columns,
            task=task,
            timestamp=timestamp,
            imbalance_handling=imbalance_handling if self.supports_imbalance_handling else "none",
        )

    supports_imbalance_handling = False

    def run(self):
        try:
            write_log(f"entered {self.__class__.__name__} -  run()", "info", os.path.basename(__file__))

            create_database_table("runs")
            emit_progress("⚙️ Initializing pipeline and loading data...", pct=5)

            data_loader(self.request_body)
            emit_progress("🧹 Validating and splitting dataset into train/test...", pct=15)

            preprocessing_controller(self.request_body)
            emit_progress("🔧 Applying preprocessing and feature engineering...", pct=35)

            emit_progress(f"🤖 Training {self.label.lower()} models...", pct=45)
            ml_model_builder_trainer_controller(self.request_body)

            dataset_name = self.request_body.original_dataset_name or os.path.basename(self.request_body.dataset_path or "")
            self.request_body.run_id = enter_record_to_table(
                "runs",
                run_name=self.request_body.name,
                folder_path=self.request_body.path_to_save,
                timestamp=self.request_body.timestamp.replace(" ", "T"),
                task=self.request_body.task,
                dataset_name=dataset_name,
                metrics_json=json.dumps(self.request_body.performance, cls=NumpyEncoder),
            )
            emit_progress("💾 Results saved to database", step=4, total_steps=5)

            df = self.request_body.df
            pipeline_summary = {
                "dataset": {
                    "name": dataset_name,
                    "rows": len(df) if df is not None else 0,
                    "cols": len(df.columns) if df is not None else 0,
                },
                "preprocessing": {
                    "imputation": self.request_body.imputation,
                    "scaling": self.request_body.scaling,
                    "reduce_features": self.request_body.reduce_features,
                    "categorical_columns": self.request_body.categorical_columns or [],
                    "ignored_columns": self.request_body.ignore_columns or [],
                },
                "imbalance_handling": self.request_body.imbalance_handling,
                "split": "80/20",
                "task": self.request_body.task,
                "models_trained": list(self.request_body.performance.keys()),
                "hyper_parameter_tuning": bool(self.request_body.hyper_parameter_tuning),
            }

            response_payload = {
                "status_code": 200,
                "message": f"{self.label} pipeline completed successfully",
                "run_id": self.request_body.run_id,
                "performance": self.request_body.performance,
                "pipeline_summary": pipeline_summary
            }

            emit_progress(f"✅ {self.label} pipeline complete!", pct=100, final=True, result=response_payload)
            return self.request_body

        except InterruptedError:
            raise
        except Exception:
            if self.request_body.status_code == 200:
                self.request_body.error_message = "OtherError"
                self.request_body.status_code = error_code_mappings[self.request_body.error_message]
                self.request_body.error_traceback = traceback.format_exc()
                write_log("error occurred- {} - {}".format(
                    self.request_body.error_message, self.request_body.error_traceback),
                    "error", os.path.basename(__file__))
            raise


class MLRegressionController(_TabularController):
    """Controller class for managing regression ML pipelines."""

    label = "Regression"


class MLClassificationController(_TabularController):
    """Controller class for managing classification ML pipelines."""

    label = "Classification"
    supports_imbalance_handling = True
