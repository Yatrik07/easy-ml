import os
import datetime
import traceback
import json
import shutil
import torch
from easyml.cv.cv_config import CV_RequestBody
from easyml.config.GLOBAL_VARS import error_code_mappings
from easyml.utils.logger import write_log
from easyml.utils.database_operations import create_database_table, enter_record_to_table
from easyml.cv.cv_preprocessing import cv_preprocessing_controller
from easyml.cv.cv_model_builder import classification_metrics, predict_loader, train_cv_model
from easyml.config.paths import RUNS_DIR
from easyml.utils.socket_emitter import emit_progress
from easyml.utils.utils import NumpyEncoder

class CVController:
    """
    Orchestrates the Computer Vision training pipeline for image classification tasks.
    """
    def __init__(self, kwargs: dict):
        self.request_body = CV_RequestBody(**kwargs)
        # Only a folder created here is removed again when the run fails; a caller's path_to_save is left alone.
        self._created_folder = None
        
    def execute_pipeline(self) -> dict:
        write_log("entered cv execute_pipeline", "info", os.path.basename(__file__))
        
        try:
            emit_progress("⚙️ Initializing CV Pipeline...", pct=5)
            create_database_table("runs")
            if not self.request_body.path_to_save:
                self._create_run_folder()

            emit_progress("🖼️ Loading image datasets and creating iterators...", pct=15)
            loaders = cv_preprocessing_controller(self.request_body)

            self._save_config()

            models_trained = []
            performance_dict = {}

            total_models = len(self.request_body.models) if self.request_body.models else 1
            for idx, model_name in enumerate(self.request_body.models or []):
                base_pct = int(20 + (idx / max(1, total_models)) * 70)
                pct_range = int(70 / max(1, total_models))
                emit_progress(f"🤖 Training model {idx+1}/{total_models}: {model_name}...", pct=base_pct)
                trained_model, best_val_acc = train_cv_model(self.request_body, model_name, loaders.train, loaders.val,
                                                             base_pct=base_pct, pct_range=pct_range)

                model_path = os.path.join(self.request_body.path_to_save, f"{model_name}.pth")
                torch.save(trained_model.state_dict(), model_path)

                emit_progress(f"📏 Evaluating {model_name} on the held-out test images...", pct=base_pct + pct_range)
                performance_dict[model_name] = self._evaluate(trained_model, loaders, model_path, best_val_acc)
                models_trained.append(model_name)

                write_log(f"Model {model_name} trained and saved successfully", "info", os.path.basename(__file__))

            emit_progress("💾 Finalizing results in database...", pct=95)
            dataset_name = ", ".join(
                os.path.basename(p.strip().rstrip("/\\")) for p in (self.request_body.dataset_path or "").split(",") if p.strip()
            )
            self.request_body.run_id = enter_record_to_table(
                "runs",
                run_name=self.request_body.name,
                folder_path=self.request_body.path_to_save,
                timestamp=datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"),
                task=self.request_body.task,
                dataset_name=dataset_name,
                metrics_json=json.dumps(performance_dict, cls=NumpyEncoder),
            )

            pipeline_summary = {
                "dataset": {
                    "name": dataset_name,
                    "path": self.request_body.dataset_path,
                    "classes": self.request_body.classes,
                    "num_classes": self.request_body.num_classes,
                    "num_images": sum(len(loader.dataset) for loader in (loaders.train, loaders.val, loaders.test)),
                },
                "preprocessing": {
                    "image_size": list(self.request_body.image_size),
                    "batch_size": self.request_body.batch_size,
                    "augmentation": self.request_body.augment,
                    "validation_split": self.request_body.validation_split,
                    "test_split": self.request_body.test_split,
                },
                "training": {
                    "epochs": self.request_body.epochs,
                    "learning_rate": self.request_body.learning_rate,
                    "use_gpu": self.request_body.use_gpu,
                },
                "split": "{}/{}/{} (train/validation/test)".format(
                    *(len(loader.dataset) for loader in (loaders.train, loaders.val, loaders.test))),
                "task": self.request_body.task,
                "models_trained": list(performance_dict.keys()),
            }

            response_payload = {
                "status_code": 200,
                "message": "Computer Vision Training completed successfully",
                "run_id": self.request_body.run_id,
                "performance": performance_dict,
                "pipeline_summary": pipeline_summary
            }
            emit_progress("✅ CV Training pipeline complete!", pct=100, final=True, result=response_payload)
            return response_payload

        except InterruptedError:
            # User-initiated cancellation — the caller reports it to the UI
            write_log("CV pipeline cancelled by user", "info", os.path.basename(__file__))
            self._remove_created_folder()
            raise
        except Exception as e:
            if self.request_body.status_code == 200:
                self.request_body.error_message = "CV_PipelineExecutionError"
                self.request_body.status_code = error_code_mappings.get(self.request_body.error_message, 500)
                self.request_body.error_traceback = traceback.format_exc()
            write_log("CV error occurred- {} - {}".format(
                self.request_body.error_message, self.request_body.error_traceback),
                "error", os.path.basename(__file__))
            self._remove_created_folder()
            raise e

    def _remove_created_folder(self):
        if self.request_body.run_id is None and self._created_folder and os.path.isdir(self._created_folder):
            shutil.rmtree(self._created_folder, ignore_errors=True)

    def _evaluate(self, model, loaders, model_path, best_val_acc):
        """Train and held-out test metrics for one model, shaped like tabular/NLP ``metrics_json`` entries."""
        class_names = loaders.class_names
        train_metrics = classification_metrics(*predict_loader(model, loaders.train_eval), class_names)
        test_metrics = classification_metrics(*predict_loader(model, loaders.test), class_names)

        def rounded(metrics, key):
            value = metrics.get(key)
            return round(float(value), 5) if value is not None else None

        return {
            "model_path": model_path,
            "accuracy_train": rounded(train_metrics, "accuracy"),
            "accuracy_test": rounded(test_metrics, "accuracy"),
            "accuracy_val": round(float(best_val_acc), 5),
            "f1_train": rounded(train_metrics, "f1_weighted"),
            "f1_test": rounded(test_metrics, "f1_weighted"),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "best_params": {},
        }

    def _create_run_folder(self):
        import re
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        safe_name = re.sub(r"[^A-Za-z0-9 _.-]+", "_", self.request_body.name).strip(" ._") or "cv_run"
        self.request_body.path_to_save = os.path.join(RUNS_DIR, f"{safe_name}_{timestamp}")
        os.makedirs(self.request_body.path_to_save, exist_ok=True)
        self._created_folder = self.request_body.path_to_save
        write_log(f"Created run folder: {self.request_body.path_to_save}", "info", os.path.basename(__file__))
        
    def _save_config(self):
        config_path = os.path.join(self.request_body.path_to_save, "config_json.json")
        conf = {
            "task": self.request_body.task,
            "classes": self.request_body.classes,
            "num_classes": self.request_body.num_classes,
            "image_size": self.request_body.image_size,
            "models": self.request_body.models,
            "custom_layers": getattr(self.request_body, 'custom_layers', None)
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(conf, f, indent=4)
        write_log("Saved configuration to config_json.json", "info", os.path.basename(__file__))
