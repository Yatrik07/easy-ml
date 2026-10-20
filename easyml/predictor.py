"""
Use a model trained in Easy-ML Studio from any Python program.

    import easyml

    model = easyml.load_model(5)                 # best model of run #5
    model.predict({"age": 45, "city": "NY"})     # -> ["high"]
    model.predict("new_customers.csv")           # one prediction per row

Runs are read from the Easy-ML data directory (``EASYML_DATA_DIR`` or the platform default),
so this works on the machine where the model was trained, or wherever that folder is copied.
"""

import io
import os
import threading

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
        # One Predictor may serve many threads (easy-ml serve): the model is loaded once, under this
        # lock, and predictions afterwards only read it.
        self._lock = threading.Lock()

    def __repr__(self):
        return (f"Predictor(run_id={self.run_id}, task={task_label(self.task)!r}, "
                f"model={model_label(self.model_name)!r})")

    # ------------------------------------------------------------------ public API

    def predict(self, data, cleaned=None):
        """
        Predict and return a list with one prediction per input.

        * Tabular runs: a dict (one row), a list of dicts, a pandas DataFrame, or a .csv/.xlsx path.
        * Text runs: a string, a list of strings, or a .csv path (uses the run's text column).
        * Image runs: an image path, a list of image paths, or a folder of images.

        A run trained on cleaned data (Data prep) first cleans raw input with the same steps and the
        numbers learned from its training data. Data that already has the cleaned columns is used as it
        is; ``cleaned=True`` or ``cleaned=False`` decides this explicitly.
        """
        return self.predict_with_confidence(data, cleaned=cleaned)[0]

    def predict_with_confidence(self, data, cleaned=None):
        """Like :meth:`predict`, but returns ``(predictions, confidences)``; confidences may be None."""
        predictions, confidences, _ = self.predict_with_notes(data, cleaned=cleaned)
        return predictions, confidences

    def predict_with_notes(self, data, cleaned=None):
        """Like :meth:`predict_with_confidence`, plus notes about cleaning the input: ``(predictions, confidences, notes)``."""
        if self.task in TABULAR_TASKS:
            return self._predict_tabular(data, cleaned)
        if self.task == TEXT_TASK:
            return (*self._predict_text(data, cleaned=cleaned), [])
        if self.task == IMAGE_TASK:
            return (*self._predict_images(data), [])
        raise ValueError(f"Unsupported task {self.task!r}")

    @property
    def cleans_input(self):
        """True when the run was trained on cleaned data and cleans raw input itself (Easy-ML 0.11+)."""
        return self.task != IMAGE_TASK and getattr(self._ready_controller(), "prep", None) is not None

    @property
    def input_columns(self):
        """The columns to give :meth:`predict` (for runs that clean their input: the original, raw columns)."""
        from easyml.core.artifacts import read_json

        if self.task == IMAGE_TASK:
            return []
        prep = getattr(self._ready_controller(), "prep", None)
        if self.task == TEXT_TASK:
            column = self.text_column()
            return list(prep.input_columns) if prep is not None and column not in prep.input_columns else [column]
        if prep is not None:
            return list(prep.input_columns)
        return list((read_json(self._folder, "column_data.json").get("column_dtypes") or {}).keys())

    def check(self, data):
        """
        Does ``data`` look like the data this model was trained on? Returns a
        :class:`~easyml.monitoring.datacheck.DataCheckReport` (``.ok``, ``.summary``, ``.to_frame()``).
        Accepts the same inputs as :meth:`predict` (tables and text, not images).
        """
        import pandas as pd

        from easyml.monitoring.datacheck import DataCheckReport, check_frame, reference_profile

        if self.task == IMAGE_TASK:
            raise ValueError("The data check works on tables and text, not images.")
        is_file = isinstance(data, (str, os.PathLike)) and str(data).lower().endswith((".csv", ".xlsx", ".xls"))
        if self.task == TEXT_TASK and not is_file and not isinstance(data, pd.DataFrame):
            texts = [data] if isinstance(data, str) else [str(t) for t in data]
            frame = pd.DataFrame({self.text_column(): texts})
        else:
            frame = self._frame(data)
        return DataCheckReport(check_frame(reference_profile(self._folder), frame))
    def warm_up(self):
        """Load the model now instead of on the first prediction. Returns self."""
        self._ready_controller()
        return self

    @property
    def is_image(self):
        return self.task == IMAGE_TASK

    @property
    def is_text(self):
        return self.task == TEXT_TASK

    def _ready_controller(self):
        """The task's inference controller, created and fully loaded exactly once."""
        controller = self._controller
        if controller is not None:
            return controller
        with self._lock:
            if self._controller is None:
                self._controller = self._load_controller()
            return self._controller

    def _load_controller(self):
        if self.task in TABULAR_TASKS:
            from easyml.core.inference import MLInferenceController

            controller = MLInferenceController(run_id=self.run_id, data=None, model_name=self.model_name)
            controller.load_objects()
        elif self.task == TEXT_TASK:
            from easyml.nlp.nlp_inference import NLPInferenceController

            controller = NLPInferenceController(run_id=self.run_id, data=None, model_name=self.model_name)
            controller._load_artifacts()
        elif self.task == IMAGE_TASK:
            try:
                from easyml.cv.cv_inference import CVInferenceController
            except (ImportError, OSError) as exc:
                raise ImportError('Image models need PyTorch: pip install "easy-ml-studio[cv]"') from exc

            controller = CVInferenceController(run_id=self.run_id, model_name=self.model_name, image_base64="")
            controller.load_model()
        else:
            raise ValueError(f"Unsupported task {self.task!r}")
        return controller

    # ------------------------------------------------------------------ task specific

    def data_warnings(self, data, limit=10):
        """Short warnings when table input does not look like the training data (empty list otherwise; never raises)."""
        if self.task not in TABULAR_TASKS:
            return []
        try:
            from easyml.monitoring.datacheck import reference_profile, warnings_for

            return warnings_for(reference_profile(self._folder), self._frame(data), limit=limit)
        except Exception:
            return []

    @staticmethod
    def _frame(data):
        import pandas as pd

        from easyml.utils.data_loader import load_data_from_path

        if isinstance(data, (str, os.PathLike)):
            return load_data_from_path(str(data))
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, dict):
            return pd.DataFrame([data])
        if isinstance(data, (list, tuple)):
            return pd.DataFrame(list(data))
        raise TypeError("Tabular input must be a dict, a list of dicts, a DataFrame or a CSV/Excel path.")

    def _predict_tabular(self, data, cleaned=None):
        return self._ready_controller().predict_frame_with_notes(self._frame(data), cleaned=cleaned)

    def text_column(self):
        import json

        try:
            with open(os.path.join(self._folder, "config_json.json"), "r", encoding="utf-8") as handle:
                return json.load(handle).get("text_column")
        except OSError:
            return None

    def _predict_text(self, data, text_column=None, frame=None, cleaned=None):
        from easyml.utils.frames import as_texts

        if isinstance(data, (str, os.PathLike)) and str(data).lower().endswith((".csv", ".xlsx", ".xls")) \
                and os.path.isfile(str(data)):
            if frame is None:  # the CLI passes the frame it already loaded
                from easyml.utils.data_loader import load_data_from_path

                frame = load_data_from_path(str(data))
            column = text_column or self.text_column()
            controller = self._ready_controller()
            if controller.prep is not None and column == self.text_column() and cleaned is not True:
                return controller.predict_frame_texts(frame, cleaned=cleaned)  # cleaned with the run's recipe
            if column not in frame.columns:
                raise ValueError(f"Column {column!r} not found in {data}. Pass text_column=... to choose one.")
            texts = as_texts(frame[column])
        elif isinstance(data, str):
            texts = [data]
        elif isinstance(data, (list, tuple)):
            texts = [str(t) for t in data]
        else:
            import pandas as pd

            if isinstance(data, pd.Series):
                texts = as_texts(data)
            else:
                raise TypeError("Text input must be a string, a list of strings, or a CSV path.")

        return self._ready_controller().predict_texts(texts, cleaned_input=cleaned)

    def _predict_images(self, data):
        paths = image_paths(data)
        if not paths:
            raise ValueError(f"No images found in {data!r}.")
        return self._ready_controller().predict_images(paths)


def _image_source(item):
    """A path stays a path string; raw bytes become a readable file object; file objects pass through."""
    if isinstance(item, (bytes, bytearray)):
        return io.BytesIO(item)
    if hasattr(item, "read"):
        return item
    return str(item)


def image_paths(data):
    """
    Expand an image path, a folder, or a list of paths into a sorted list of image files.
    Image bytes and file objects (alone or in a list) are passed through as readable sources.
    """
    if isinstance(data, (list, tuple)):
        return [_image_source(p) for p in data]
    if isinstance(data, (bytes, bytearray)) or hasattr(data, "read"):
        return [_image_source(data)]
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
