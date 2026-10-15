import os
import re
import shutil
import traceback
import uuid
from datetime import datetime

from easyml.config.constants import error_code_mappings
from easyml.config.paths import RUNS_DIR
from easyml.utils.logger import write_log


def _build_controller_registry():
    from easyml.core.controllers import MLClassificationController, MLRegressionController

    registry = {
        "MLClassification": MLClassificationController,
        "MLRegression": MLRegressionController,
    }

    try:
        from easyml.nlp.nlp_controllers import NLPTextClassificationController

        registry["NLPTextClassification"] = NLPTextClassificationController
    except ImportError:
        pass

    return registry


class MLTrainer:
    """
    Resolve the requested task to the matching controller and execute it.

    Example::

        from easyml import MLTrainer

        run = MLTrainer(
            task="MLClassification",          # or "MLRegression", "NLPTextClassification"
            name="churn",
            dataset_path="churn.csv",
            target_column="churned",
            models=["logistic_regression", "random_forest"],
        ).run()
        print(run.run_id, run.performance)

    ``timestamp`` and ``path_to_save`` are optional; a run folder is created in the
    Easy-ML data directory when they are omitted.
    """

    def __init__(self, **args):
        self.controller = None
        # Only a folder created here is removed again when the run fails; a caller's path_to_save is left alone.
        self._created_folder = None

        try:
            task = args["task"]
            if not args.get("timestamp"):
                args["timestamp"] = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            if not args.get("path_to_save"):
                safe_name = re.sub(r"[^A-Za-z0-9 _.-]+", "_", str(args.get("name") or "run")).strip(" ._") or "run"
                args["path_to_save"] = os.path.join(RUNS_DIR, f"{safe_name}_{args['timestamp']}_{uuid.uuid4().hex[:4]}")
                os.makedirs(args["path_to_save"], exist_ok=True)
                self._created_folder = args["path_to_save"]
            controller_class_name = f"{task}Controller"
            write_log(
                f"entered MLTrainer __init__ for controller - {controller_class_name}",
                "info",
                os.path.basename(__file__),
            )

            registry = _build_controller_registry()
            controller_class = registry.get(task)
            if controller_class is None:
                raise ValueError(f"Unsupported task: {task}")

            self.controller = controller_class(**args)

        except Exception as exc:
            self._remove_created_folder()
            if (
                self.controller is not None
                and hasattr(self.controller, "request_body")
                and self.controller.request_body.status_code == 200
            ):
                self.controller.request_body.error_message = "OtherError"
                self.controller.request_body.status_code = error_code_mappings["OtherError"]
                self.controller.request_body.error_traceback = traceback.format_exc()

                write_log(
                    "error occurred- {} - {}".format(
                        self.controller.request_body.error_message,
                        self.controller.request_body.error_traceback,
                    ),
                    "error",
                    os.path.basename(__file__),
                )

            raise exc

    def _remove_created_folder(self):
        request_body = getattr(self.controller, "request_body", None)
        if getattr(request_body, "run_id", None) is not None:
            return  # the run reached the database, so its folder is not an orphan
        if self._created_folder and os.path.isdir(self._created_folder):
            shutil.rmtree(self._created_folder, ignore_errors=True)

    def run(self):
        try:
            return self.controller.run()

        except InterruptedError:
            self._remove_created_folder()
            raise
        except Exception as exc:
            self._remove_created_folder()
            if self.controller.request_body.status_code == 200:
                self.controller.request_body.error_message = "OtherError"
                self.controller.request_body.status_code = error_code_mappings["OtherError"]
                self.controller.request_body.error_traceback = traceback.format_exc()

                write_log(
                    "error occurred- {} - {}".format(
                        self.controller.request_body.error_message,
                        self.controller.request_body.error_traceback,
                    ),
                    "error",
                    os.path.basename(__file__),
                )

            raise exc
