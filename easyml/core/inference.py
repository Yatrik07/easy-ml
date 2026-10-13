import os
import traceback

import numpy as np
import pandas as pd

from easyml.config.GLOBAL_VARS import ML_InferenceBody, error_code_mappings
from easyml.core.artifacts import (
    coerce_feature_frame,
    flatten_predictions,
    load_pipeline,
    load_tabular_model,
    load_target_decoder,
    read_json,
)
from easyml.utils.database_operations import get_record_by_id
from easyml.utils.logger import write_log
from easyml.utils.utils import convert_numpy


class MLInferenceController:
    """
    Controller for performing inference on a trained ML model using saved artifacts.

    Attributes:
        inference_body (ML_InferenceBody): Holds model config, input, and outputs.
    """

    def __init__(self, run_id, data, model_name):
        """
        Initializes the inference controller with the required metadata.

        Args:
            run_id (int): Unique ID corresponding to the training run.
            data (dict): Input data for which prediction is to be made.
            model_name (str): Name of the model to be loaded for inference.
        """

        self.inference_body = ML_InferenceBody(
            run_id=run_id,
            data=data,
            model_name=model_name
        )
        self.column_data = {}

    def _record_error(self):
        if self.inference_body.status_code == 200:
            self.inference_body.error_message = "InferenceError"
            self.inference_body.status_code = error_code_mappings[self.inference_body.error_message]
            self.inference_body.error_traceback = traceback.format_exc()
            write_log("error occurred- {} - {}".format(
                self.inference_body.error_message, self.inference_body.error_traceback),
                "error", os.path.basename(__file__))

    def get_inference(self):
        """
        Loads the model, pipeline, and target encodings, then performs prediction.

        Returns:
            dict: Contains the predicted value and associated probability (if available).
        """

        write_log("entered get_inference", "info", os.path.basename(__file__))

        try:
            self.load_objects()
            return self.predict_single_point()
        except Exception:
            self._record_error()
            raise

    def _transform(self, df):
        df = coerce_feature_frame(df, self.column_data)
        return self.inference_body.pipeline.transform(df)

    def _decode(self, value):
        value = convert_numpy(value)
        if self.inference_body.target_encodings is None:
            return value
        return self.inference_body.target_encodings.get(int(value), value)

    def predict_single_point(self):
        """
        Applies the pipeline and model to predict a single input record.

        Returns:
            dict: Dictionary containing prediction and probability (if available).
        """

        write_log("entered predict_single_point", "info", os.path.basename(__file__))

        transformed_input = self._transform(pd.DataFrame([self.inference_body.data]))
        model = self.inference_body.model
        prediction = flatten_predictions(model.predict(transformed_input))[0]

        is_classification = "classification" in self.inference_body.task.lower()
        probability = None
        if is_classification and hasattr(model, "predict_proba"):
            probability = float(np.max(model.predict_proba(transformed_input)[0]))

        output = {
            "prediction": self._decode(prediction) if is_classification else convert_numpy(prediction),
            "probability": probability,
        }
        write_log(f"prediction complete: {output['prediction']}", "info", os.path.basename(__file__))
        return output

    def predict_batch(self, df: pd.DataFrame):
        """
        Applies the pipeline and model to predict multiple records from a DataFrame.
        Returns the original DataFrame with an appended 'Prediction' column.
        """
        write_log("entered predict_batch", "info", os.path.basename(__file__))
        try:
            self.load_objects()
            predictions = flatten_predictions(self.inference_body.model.predict(self._transform(df)))

            output_df = df.copy()
            if "classification" in self.inference_body.task.lower():
                output_df['Prediction'] = [self._decode(p) for p in predictions]
            else:
                output_df['Prediction'] = [convert_numpy(p) for p in predictions]
            return output_df
        except Exception:
            self._record_error()
            raise

    def load_objects(self):
        """
        Loads the saved model, preprocessing pipeline, column metadata and target encodings from disk.
        """

        write_log("entered load_objects", "info", os.path.basename(__file__))

        record = get_record_by_id(self.inference_body.run_id)
        folder_path = record["folder_path"]
        self.inference_body.task = record["task"]

        self.inference_body.model = load_tabular_model(folder_path, self.inference_body.model_name, record["task"])
        self.inference_body.pipeline = load_pipeline(folder_path)
        self.column_data = read_json(folder_path, "column_data.json")
        if "classification" in record["task"].lower():
            self.inference_body.target_encodings = load_target_decoder(folder_path)

        write_log("Loaded model and pipeline objects successfully", "info", os.path.basename(__file__))
