import json
import os
import traceback

from easyml.core.multi_target import clean_targets, clean_weights
from easyml.core.seeding import validate_cv_folds, validate_seed
from easyml.utils.run_inputs import inherit_from_evaluated_run, record_inputs, recipe_from_sidecars
from easyml.utils.run_meta import clean_tags, tags_json
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

    def __init__(self, name, dataset_path, target_column=None, timestamp="", preprocessing=True, imputation=False,
                 scaling=False,
                 models=None, hyper_parameter_tuning=0, categorical_columns=None, path_to_save=None,
                 reduce_features=False, ignore_columns=None, task=None, imbalance_handling="none",
                 original_dataset_name=None, parent_run_id=None, recipe=None, random_seed=42, cv_folds=0, tags=None,
                 final_of_run=None, fixed_params=None, estimates=None, target_columns=None, target_weights=None,
                 **kwargs):
        """
        Args:
            name (str): Name of the run.
            dataset_path (str): Path to the dataset file.
            target_column (str or list): Name of the target column, or a list of several (0.13; also
                ``target_columns``). Each target gets its own models; ``target_weights`` ({column: 0.1-10})
                weigh the targets' scores in the run's score.
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
            parent_run_id (int): The run this one retrains, if any.
            final_of_run (int): Train the final model of this evaluated run: every row, its best parameters
                (``fixed_params``), its test scores as estimates (``estimates``); see ``retraining.plan_final``.
        """

        target_column, target_columns = clean_targets(target_column, target_columns)
        if target_columns:
            target_weights = clean_weights(target_weights, target_columns)
            left_out = [t for t in target_columns if t in (ignore_columns or [])]
            if left_out:
                raise ValueError(f"'{left_out[0]}' is a column to predict, so it cannot also be left out.")
            if self.supports_imbalance_handling and (imbalance_handling or "none").lower() != "none":
                raise ValueError("Class balance (resampling) is not available with several columns to predict.")
        elif target_weights:
            raise ValueError("Target weights need several columns to predict.")

        # A file cleaned by Data prep carries its frozen recipe next to it (see easyml.dataprep.frozen).
        self.source_path = dataset_path
        recipe = recipe or recipe_from_sidecars(dataset_path)
        self.request_body = ML_RequestBody(
            name=name,
            dataset_path=dataset_path,
            target_column=target_column,
            target_columns=target_columns,
            target_weights=target_weights or {},
            # Inference always loads preprocessing_pipeline.pkl, so the pipeline is always built.
            preprocessing=True,
            scaling=scaling,
            models=models or [],
            categorical_columns=categorical_columns or [],
            requested_categorical_columns=list(categorical_columns or []),
            path_to_save=path_to_save,
            hyper_parameter_tuning=hyper_parameter_tuning,
            imputation=imputation,
            reduce_features=reduce_features,
            ignore_columns=ignore_columns,
            task=task,
            timestamp=timestamp,
            imbalance_handling=imbalance_handling if self.supports_imbalance_handling else "none",
            original_dataset_name=original_dataset_name,
            parent_run_id=parent_run_id,
            recipe=recipe or None,
            random_seed=validate_seed(random_seed),
            cv_folds=validate_cv_folds(cv_folds),
            tags=clean_tags(tags) or None,
            final_of_run=final_of_run,
            fixed_params=fixed_params or {},
            estimates=estimates or {},
        )

    supports_imbalance_handling = False

    def run(self):
        try:
            write_log(f"entered {self.__class__.__name__} -  run()", "debug", os.path.basename(__file__))

            create_database_table("runs")
            emit_progress("⚙️ Initializing pipeline and loading data...", pct=5)

            data_loader(self.request_body)
            emit_progress("🧹 Validating the dataset (final model: every row is used for training)..."
                          if self.request_body.final_of_run else "🧹 Validating and splitting dataset into train/test...",
                          pct=15)

            preprocessing_controller(self.request_body)
            emit_progress("🔧 Applying preprocessing and feature engineering...", pct=35)

            emit_progress(f"🤖 Training {self.label.lower()} models...", pct=45)
            ml_model_builder_trainer_controller(self.request_body)
            record_inputs(self.request_body.path_to_save, self.source_path, self.request_body.df,
                          list(self.request_body.X.columns),
                          self.request_body.target_columns or self.request_body.target_column,
                          recipe=self.request_body.recipe)
            if self.request_body.final_of_run:
                inherit_from_evaluated_run(self.request_body.path_to_save, self.request_body.final_of_run,
                                           list(self.request_body.performance))

            dataset_name = self.request_body.original_dataset_name or os.path.basename(self.request_body.dataset_path or "")
            self.request_body.run_id = enter_record_to_table(
                "runs",
                run_name=self.request_body.name,
                folder_path=self.request_body.path_to_save,
                timestamp=self.request_body.timestamp.replace(" ", "T"),
                task=self.request_body.task,
                dataset_name=dataset_name,
                metrics_json=json.dumps(self.request_body.performance, cls=NumpyEncoder),
                parent_run_id=self.request_body.parent_run_id,
                origin=("final" if self.request_body.final_of_run
                        else "retrain" if self.request_body.parent_run_id else None),
                tags=tags_json(self.request_body.tags),
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
                "split": "all rows (final model)" if self.request_body.final_of_run else "80/20",
                "final_of_run": self.request_body.final_of_run,
                "task": self.request_body.task,
                "models_trained": list(self.request_body.performance.keys()),
                "hyper_parameter_tuning": bool(self.request_body.hyper_parameter_tuning),
                "parent_run_id": self.request_body.parent_run_id,
            }
            if self.request_body.target_columns:
                pipeline_summary["targets"] = self.request_body.target_columns
                pipeline_summary["target_weights"] = self.request_body.target_weights

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
