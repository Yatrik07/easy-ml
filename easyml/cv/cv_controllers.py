import os
import datetime
import traceback
import json
import torch
from easyml.cv.cv_config import CV_RequestBody
from easyml.config.GLOBAL_VARS import error_code_mappings
from easyml.utils.logger import write_log
from easyml.utils.database_operations import enter_record_to_table, get_data_from_database
from easyml.cv.cv_preprocessing import cv_preprocessing_controller
from easyml.cv.cv_model_builder import train_cv_model, get_cv_model
from easyml.config.paths import RUNS_DIR
from easyml.utils.socket_emitter import emit_progress
from easyml.utils.utils import NumpyEncoder

class CVController:
    """
    Orchestrates the Computer Vision training pipeline for image classification tasks.
    """
    def __init__(self, kwargs: dict):
        self.request_body = CV_RequestBody(**kwargs)
        
    def execute_pipeline(self) -> dict:
        write_log("entered cv execute_pipeline", "info", os.path.basename(__file__))
        total_models = len(self.request_body.models) if self.request_body.models else 1
        total_steps = 3 + total_models  # init + load + (N models) + finalize
        step = 0
        try:
            step += 1
            emit_progress("⚙️ Initializing CV Pipeline...", step=step, total_steps=total_steps)
            self._create_run_folder()

            step += 1
            emit_progress("🖼️ Loading image datasets and creating data iterators...", step=step, total_steps=total_steps)
            train_loader, val_loader, class_names = cv_preprocessing_controller(self.request_body)

            self._save_config()

            models_trained = []
            performance_dict = {}

            for model_name in self.request_body.models:
                step += 1
                emit_progress(f"🤖 Training {model_name}... (this may take a while)", step=step, total_steps=total_steps)
                trained_model, best_acc = train_cv_model(self.request_body, model_name, train_loader, val_loader)

                model_path = os.path.join(self.request_body.path_to_save, f"{model_name}.pth")
                torch.save(trained_model.state_dict(), model_path)

                models_trained.append(model_name)
                performance_dict[model_name] = {"accuracy_test": best_acc}

                write_log(f"Model {model_name} trained and saved successfully", "info", os.path.basename(__file__))

            emit_progress("💾 Finalizing results in database...", step=step, total_steps=total_steps)
            enter_record_to_table("runs", **{
                "run_name": self.request_body.name,
                "folder_path": self.request_body.path_to_save,
                "timestamp": datetime.datetime.now().isoformat(),
                "task": self.request_body.task,
                "dataset_name": os.path.basename(self.request_body.dataset_path or ""),
                "metrics_json": json.dumps(performance_dict, cls=NumpyEncoder)
            })

            runs_data = get_data_from_database("runs")
            if runs_data.get("status_code") == 200 and len(runs_data.get("data", [])) > 0:
                self.request_body.run_id = runs_data["data"][-1]["id"]
            else:
                self.request_body.run_id = -1

            response_payload = {
                "status_code": 200,
                "message": "Computer Vision Training completed successfully",
                "run_id": self.request_body.run_id,
                "performance": performance_dict
            }
            emit_progress("✅ CV Training pipeline complete!", pct=100, final=True, result=response_payload)
            return response_payload

        except InterruptedError:
            # User-initiated cancellation — don't log as error
            write_log("CV pipeline cancelled by user", "info", os.path.basename(__file__))
            return {"status_code": 200, "status": "cancelled", "message": "Training was cancelled by the user."}
        except Exception as e:
            if self.request_body.status_code == 200:
                self.request_body.error_message = "CV_PipelineExecutionError"
                self.request_body.status_code = error_code_mappings.get(self.request_body.error_message, 500)
                self.request_body.error_traceback = traceback.format_exc()
            write_log("CV error occurred- {} - {}".format(
                self.request_body.error_message, self.request_body.error_traceback),
                "error", os.path.basename(__file__))
            emit_progress(f"❌ Error: {str(e)}", pct=100, final=True,
                          result={"status": "error", "error": str(e), "message": str(e)})
            raise e

    def _create_run_folder(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{self.request_body.name}_{timestamp}"
        self.request_body.path_to_save = os.path.join(RUNS_DIR, folder_name)
        os.makedirs(self.request_body.path_to_save, exist_ok=True)
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
        with open(config_path, "w") as f:
            json.dump(conf, f, indent=4)
        write_log("Saved configuration to config_json.json", "info", os.path.basename(__file__))
