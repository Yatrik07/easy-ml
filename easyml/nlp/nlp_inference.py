"""
NLP Inference Module
====================
Handles loading trained NLP artifacts and performing prediction
on new text input.

Mirrors easyml/core/inference.py for NLP tasks.
"""

import os
import json
import traceback
import joblib
import scipy.sparse

from easyml.nlp.nlp_config import NLP_InferenceBody, NLP_ERROR_CODES
from easyml.nlp.nlp_preprocessing import apply_preprocessing_for_inference
from easyml.nlp.nlp_feature_engineering import (
    load_fe_config,
    transform_text_for_inference,
    transform_texts_for_inference,
)
from easyml.core.artifacts import load_frozen_recipe, validate_model_name
from easyml.utils.database_operations import get_record_by_id
from easyml.utils.labels import decoder_from_config
from easyml.utils.logger import write_log
from easyml.utils.utils import convert_numpy


class NLPInferenceController:
    """
    Controller for NLP inference.

    Loads a trained model, vectorizer/embedding config, and target encodings
    from a previous run, then applies identical preprocessing + feature
    engineering to produce a prediction.
    """

    def __init__(self, run_id, data, model_name):
        """
        Initialize the NLP inference controller.

        Args:
            run_id (int): Database ID of the training run.
            data (str): Raw text input for prediction.
            model_name (str): Class name of the model to load (e.g., 'LogisticRegression').
        """
        self.inference_body = NLP_InferenceBody(
            run_id=run_id,
            data=data,
            model_name=validate_model_name(model_name),
        )
        self.prep = None  # the run's frozen data-prep recipe, when it was trained on cleaned data
        self.text_column = None

    def get_inference(self):
        """
        Perform full inference pipeline.

        Returns:
            dict: Contains 'prediction' (label) and 'probability' (confidence).

        Raises:
            Exception: Logs and re-raises with NLPInferenceError code.
        """
        try:
            write_log("entered NLPInferenceController.get_inference", "debug", os.path.basename(__file__))

            # Load all artifacts
            self._load_artifacts()
            write_log("loaded NLP inference artifacts successfully", "info", os.path.basename(__file__))

            # Perform prediction
            result = self._predict()
            write_log("NLP inference prediction completed", "info", os.path.basename(__file__))

            return result

        except Exception as e:
            if self.inference_body.status_code == 200:
                self.inference_body.error_message = "NLPInferenceError"
                self.inference_body.status_code = NLP_ERROR_CODES["NLPInferenceError"]
                self.inference_body.error_traceback = traceback.format_exc()
                write_log(f"error occurred - {self.inference_body.error_message} - "
                          f"{self.inference_body.error_traceback}",
                          "error", os.path.basename(__file__))
            raise e

    def clean_texts(self, texts, cleaned=None):
        """
        Texts as the model saw them in training: runs trained on cleaned data apply their recipe to the text
        column first (e.g. trimming or a fixed letter case). Other runs get the texts unchanged.
        """
        if self.prep is None or cleaned is True:
            return list(texts)
        import pandas as pd

        frame, _ = self.prep.prepare(pd.DataFrame({self.text_column: list(texts)}), cleaned=cleaned)
        return [("" if pd.isna(t) else str(t)) for t in frame[self.text_column]]

    def predict_frame_texts(self, frame, cleaned=None):
        """Classify the rows of a table: runs trained on cleaned data clean the table first (any recipe step)."""
        if self.inference_body.model is None:
            self._load_artifacts()
        if self.prep is not None and cleaned is not True:
            prepared, _ = self.prep.prepare(frame, cleaned=cleaned)
            texts = ["" if t != t else str(t) for t in prepared[self.text_column]]
            return self.predict_texts(texts, cleaned_input=True)
        if self.text_column not in frame.columns:
            raise ValueError(f"Column {self.text_column!r} not found.")
        return self.predict_texts(frame[self.text_column].astype(str).tolist(), cleaned_input=True)

    def predict_texts(self, texts, cleaned_input=None):
        """
        Classify several texts at once. ``cleaned_input`` works like ``cleaned`` in
        :meth:`FrozenRecipe.prepare` for runs trained on cleaned data.

        Returns:
            tuple: (labels, confidences); confidences is None for models without predict_proba.
        """
        if self.inference_body.model is None:
            self._load_artifacts()
        texts = self.clean_texts(texts, cleaned=cleaned_input)
        config = self.inference_body.preprocessing_config
        cleaned = [apply_preprocessing_for_inference(str(t), config) for t in texts]
        features = transform_texts_for_inference(cleaned, self._folder_path, self.inference_body.embedding_config)
        if any(dm in self.inference_body.model_name.lower() for dm in ['mlp', 'lgbm']) and scipy.sparse.issparse(features):
            features = features.toarray()
        model = self.inference_body.model
        predictions = model.predict(features).reshape(-1)
        decoder = self.inference_body.target_encodings or {}
        labels = [decoder.get(convert_numpy(p), str(convert_numpy(p))) for p in predictions]
        confidences = None
        if hasattr(model, "predict_proba"):
            try:
                confidences = [float(row.max()) for row in model.predict_proba(features)]
            except Exception:
                confidences = None
        return labels, confidences

    def _load_artifacts(self):
        """Load model, config, vectorizer/embedding config, and target encodings."""

        write_log("loading NLP inference artifacts", "info", os.path.basename(__file__))

        # Get run record from database
        record = get_record_by_id(self.inference_body.run_id)

        folder_path = record["folder_path"]
        self.inference_body.task = record["task"]

        # Load model
        model_path = os.path.join(folder_path, f"{self.inference_body.model_name}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model '{self.inference_body.model_name}' was not found for this run.")

        from easyml.core.model_cache import cached_load

        self.inference_body.model = cached_load("nlp_model", model_path, lambda: joblib.load(model_path))
        write_log(f"loaded model from {model_path}", "info", os.path.basename(__file__))

        # Load config_json.json for preprocessing config and target encodings
        config_path = os.path.join(folder_path, "config_json.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        self.inference_body.preprocessing_config = config
        self.text_column = config.get("text_column")
        self.prep = load_frozen_recipe(folder_path)

        # Load target encodings (reverse: encoded_value -> label)
        decoder = decoder_from_config(config)
        if decoder:
            self.inference_body.target_encodings = decoder
        write_log("loaded preprocessing config and target encodings", "info", os.path.basename(__file__))

        # Load feature engineering config
        fe_config = load_fe_config(folder_path)
        self.inference_body.embedding_config = fe_config
        self._folder_path = folder_path
        write_log(f"loaded FE config: mode={fe_config['feature_mode']}", "info", os.path.basename(__file__))

    def _predict(self):
        """
        Apply preprocessing, feature engineering, and model prediction.

        Returns:
            dict: {'prediction': <label>, 'probability': <confidence>}
        """
        raw_text = self.clean_texts([str(self.inference_body.data)])[0]

        # Step 1: Apply same preprocessing
        cleaned_text = apply_preprocessing_for_inference(
            raw_text,
            self.inference_body.preprocessing_config
        )
        write_log(f"preprocessed inference text: '{cleaned_text[:100]}...'", "info", os.path.basename(__file__))

        # Step 2: Feature engineering (vectorize or embed)
        fe_config = self.inference_body.embedding_config
        features = transform_text_for_inference(cleaned_text, self._folder_path, fe_config)

        # Handle sparse/dense compatibility
        model_name_lower = self.inference_body.model_name.lower()
        # Check if model needs dense input
        needs_dense = any(dm in model_name_lower for dm in ['mlp', 'lgbm'])
        if needs_dense and scipy.sparse.issparse(features):
            features = features.toarray()

        # Step 3: Predict
        prediction = self.inference_body.model.predict(features)

        # Step 4: Get probability if available (LinearSVC has none; decision scores are not probabilities)
        probability = None
        if hasattr(self.inference_body.model, "predict_proba"):
            try:
                probability = float(self.inference_body.model.predict_proba(features)[0].max())
            except Exception:
                probability = None

        # Decode prediction label
        pred_value = convert_numpy(prediction.reshape(-1)[0])
        if self.inference_body.target_encodings:
            pred_label = self.inference_body.target_encodings.get(pred_value, str(pred_value))
        else:
            pred_label = str(pred_value)

        output = {
            "prediction": pred_label,
            "probability": probability,
        }

        write_log(f"NLP inference result: {output}", "info", os.path.basename(__file__))
        return output
