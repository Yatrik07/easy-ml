"""
Use a model trained in Easy-ML Studio from any Python program.

    import easyml

    model = easyml.load_model(5)                 # best model of run #5
    model.predict({"age": 45, "city": "NY"})     # -> ["high"]
    model.predict("new_customers.csv")           # one prediction per row

Runs are read from the Easy-ML data directory (``EASYML_DATA_DIR`` or the platform default),
so this works on the machine where the model was trained, or wherever that folder is copied.
"""

import os

from easyml.core.artifacts import validate_model_name
from easyml.utils.database_operations import get_record_by_id
from easyml.utils.run_summary import model_label, parse_metrics, rank_models, task_label

TABULAR_TASKS = ("MLClassification", "MLRegression")
TEXT_TASK = "NLPTextClassification"
IMAGE_TASK = "ImageClassification"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")


class Predictor:
    """A trained model from one Easy-ML run, ready to make predictions."""

    def __init__(self, run_id, model_name=None):
        record = get_record_by_id(int(run_id))
        self.run_id = int(run_id)
        self.task = record["task"]
        self.run_name = record.get("run_name")
        self._folder = record["folder_path"]

        ranked = rank_models(self.task, parse_metrics(record.get("metrics_json")))
        self.available_models = [m["key"] for m in ranked]
        if model_name is None:
            if not self.available_models:
                raise ValueError(f"Run {self.run_id} has no trained models.")
            model_name = self.available_models[0]
        validate_model_name(model_name)
        if self.available_models and model_name not in self.available_models:
            raise ValueError(
                f"Model {model_name!r} is not part of run {self.run_id}. "
                f"Available: {', '.join(self.available_models)}"
            )
        self.model_name = model_name
        self._controller = None

    def __repr__(self):
        return (f"Predictor(run_id={self.run_id}, task={task_label(self.task)!r}, "
                f"model={model_label(self.model_name)!r})")

    # ------------------------------------------------------------------ public API

    def predict(self, data):
        """
        Predict and return a list with one prediction per input.

        * Tabular runs: a dict (one row), a list of dicts, a pandas DataFrame, or a .csv/.xlsx path.
        * Text runs: a string, a list of strings, or a .csv path (uses the run's text column).
        * Image runs: an image path, a list of image paths, or a folder of images.
        """
        return self.predict_with_confidence(data)[0]

    def predict_with_confidence(self, data):
        """Like :meth:`predict`, but returns ``(predictions, confidences)``; confidences may be None."""
        if self.task in TABULAR_TASKS:
            return self._predict_tabular(data)
        if self.task == TEXT_TASK:
            return self._predict_text(data)
        if self.task == IMAGE_TASK:
            return self._predict_images(data)
        raise ValueError(f"Unsupported task {self.task!r}")

    # ------------------------------------------------------------------ task specific

    def _predict_tabular(self, data):
        import pandas as pd

        from easyml.core.inference import MLInferenceController
        from easyml.utils.data_loader import load_data_from_path

        if isinstance(data, (str, os.PathLike)):
            frame = load_data_from_path(str(data))
        elif isinstance(data, pd.DataFrame):
            frame = data
        elif isinstance(data, dict):
            frame = pd.DataFrame([data])
        elif isinstance(data, (list, tuple)):
            frame = pd.DataFrame(list(data))
        else:
            raise TypeError("Tabular input must be a dict, a list of dicts, a DataFrame or a CSV/Excel path.")

        if self._controller is None:
            self._controller = MLInferenceController(run_id=self.run_id, data=None, model_name=self.model_name)
        return self._controller.predict_frame(frame)

    def text_column(self):
        import json

        try:
            with open(os.path.join(self._folder, "config_json.json"), "r", encoding="utf-8") as handle:
                return json.load(handle).get("text_column")
        except OSError:
            return None

    def _predict_text(self, data, text_column=None, frame=None):
        from easyml.nlp.nlp_inference import NLPInferenceController

        if isinstance(data, (str, os.PathLike)) and str(data).lower().endswith((".csv", ".xlsx", ".xls")) \
                and os.path.isfile(str(data)):
            if frame is None:  # the CLI passes the frame it already loaded
                from easyml.utils.data_loader import load_data_from_path

                frame = load_data_from_path(str(data))
            column = text_column or self.text_column()
            if column not in frame.columns:
                raise ValueError(f"Column {column!r} not found in {data}. Pass text_column=... to choose one.")
            texts = frame[column].astype(str).tolist()
        elif isinstance(data, str):
            texts = [data]
        elif isinstance(data, (list, tuple)):
            texts = [str(t) for t in data]
        else:
            import pandas as pd

            if isinstance(data, pd.Series):
                texts = data.astype(str).tolist()
            else:
                raise TypeError("Text input must be a string, a list of strings, or a CSV path.")

        if self._controller is None:
            self._controller = NLPInferenceController(run_id=self.run_id, data=None, model_name=self.model_name)
        return self._controller.predict_texts(texts)

    def _predict_images(self, data):
        try:
            from easyml.cv.cv_inference import CVInferenceController
        except (ImportError, OSError) as exc:
            raise ImportError('Image models need PyTorch: pip install "easy-ml-studio[cv]"') from exc

        paths = image_paths(data)
        if not paths:
            raise ValueError(f"No images found in {data!r}.")
        if self._controller is None:
            self._controller = CVInferenceController(run_id=self.run_id, model_name=self.model_name, image_base64="")
        return self._controller.predict_images(paths)


def image_paths(data):
    """Expand an image path, a folder, or a list of paths into a sorted list of image files."""
    if isinstance(data, (list, tuple)):
        return [str(p) for p in data]
    path = str(data)
    if os.path.isdir(path):
        found = []
        for folder, _, files in os.walk(path):
            found.extend(os.path.join(folder, f) for f in files if f.lower().endswith(IMAGE_EXTENSIONS))
        return sorted(found)
    return [path]


def load_model(run_id, model_name=None):
    """
    Load a trained model from an Easy-ML run.

    Args:
        run_id: The run number shown in the studio's Run History (or by ``easy-ml runs``).
        model_name: A model class name such as ``"RandomForestClassifier"``. Defaults to the run's best model.
    """
    return Predictor(run_id, model_name=model_name)
