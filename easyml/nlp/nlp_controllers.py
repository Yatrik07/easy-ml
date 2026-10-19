"""
NLP Controllers Module
=======================
NLP pipeline orchestrators, following the same pattern as
easyml/core/controllers.py (MLClassificationController / MLRegressionController).

Each controller wires together:
    data loading → preprocessing → feature engineering → model training → database save
"""

import os
import traceback
import shutil
import pandas as pd
import json 

from easyml.core.seeding import validate_cv_folds, validate_seed
from easyml.utils.run_meta import clean_tags, tags_json
from easyml.nlp.nlp_config import NLP_RequestBody, NLP_ERROR_CODES
from easyml.nlp.nlp_preprocessing import nlp_preprocessing_controller
from easyml.nlp.nlp_feature_engineering import vectorizer_controller, embedding_controller
from easyml.nlp.nlp_model_builder import nlp_model_builder_controller
from easyml.utils.data_loader import load_data_from_path, encode_categorical_list
from easyml.utils.cancellation import is_cancelled
from easyml.utils.database_operations import create_database_table, enter_record_to_table
from easyml.utils.holdout import kept_rows
from easyml.utils.logger import write_log
from easyml.utils.socket_emitter import emit_progress
from easyml.utils.utils import NumpyEncoder


def nlp_data_loader(request_body: NLP_RequestBody):
    """
    Loads a text dataset and prepares text (X) and target (y) columns.

    Steps:
    1. Moves data.csv into the run folder
    2. Reads the CSV
    3. Drops rows with null text or target
    4. Separates text column (X) and target column (y)
    5. Encodes target labels to integers for classification

    Args:
        request_body (NLP_RequestBody): Data carrier.

    Raises:
        Exception: Re-raises with NLPDataIngestionError code.
    """
    try:
        write_log("entered nlp_data_loader", "debug", os.path.basename(__file__))

        # Move data to run folder
        run_data_path = os.path.join(request_body.path_to_save, "data.csv")
        original_path = request_body.dataset_path
        if not original_path or not os.path.exists(original_path):
            raise FileNotFoundError(f"Dataset '{original_path}' was not found. Please upload it again.")
        shutil.copyfile(original_path, run_data_path)

        # Keep the user's own file name when the caller provided it (uploads are stored under internal names).
        request_body.original_dataset_name = request_body.original_dataset_name or os.path.basename(original_path)
        request_body.dataset_path = run_data_path

        # Load data
        request_body.df = load_data_from_path(request_body.dataset_path)

        if request_body.text_column not in request_body.df.columns:
            raise ValueError(f"Text column '{request_body.text_column}' not found in dataset. "
                             f"Available columns: {request_body.df.columns.tolist()}")

        if request_body.target_column not in request_body.df.columns:
            raise ValueError(f"Target column '{request_body.target_column}' not found in dataset. "
                             f"Available columns: {request_body.df.columns.tolist()}")

        # Drop nulls in text and target columns
        request_body.df = request_body.df.dropna(subset=[request_body.text_column, request_body.target_column])
        request_body.source_rows = kept_rows(None, request_body.df.index)
        request_body.df.reset_index(drop=True, inplace=True)

        # (imputation=False would drop rows with missing text or target, which the step above already did)

# Ensure text column is string
        request_body.df[request_body.text_column] = request_body.df[request_body.text_column].astype(str)

        # Separate X (text) and y (target)
        request_body.X = request_body.df[request_body.text_column]
        request_body.y = request_body.df[request_body.target_column]

        # Encode target labels
        request_body.target_dtype = str(request_body.y.dtype)
        request_body.target_encodings, encoded_y = encode_categorical_list(request_body.y.tolist())
        request_body.y = pd.Series(encoded_y)

        write_log(
            f"NLP data loaded: {len(request_body.df)} rows, "
            f"text_col='{request_body.text_column}', target_col='{request_body.target_column}', "
            f"labels={list(request_body.target_encodings.keys())}",
            "info", os.path.basename(__file__)
        )

    except Exception as e:
        if request_body.status_code == 200:
            request_body.error_message = "NLPDataIngestionError"
            request_body.status_code = NLP_ERROR_CODES[request_body.error_message]
            request_body.error_traceback = traceback.format_exc()
            write_log(f"error occurred - {request_body.error_message} - {request_body.error_traceback}",
                      "error", os.path.basename(__file__))
        raise e


class NLPTextClassificationController:
    """
    Controller for NLP text classification pipeline.

    Orchestrates the full NLP workflow:
        Step 1: Create/verify database table
        Step 2: Load text data
        Step 3: Preprocess text (clean, normalize)
        Step 4: Feature engineering (vectorize or embed)
        Step 5: Train and evaluate models
        Step 6: Save run metadata to database

    This class is dynamically dispatched by MLTrainer when task='NLPTextClassification'.
    """

    def __init__(self, name, dataset_path, target_column, timestamp, text_column="",
                 lowercase=True, remove_stopwords=True, lemmatize=False,
                 regex_clean=True, regex_pattern=r'[^a-zA-Z0-9\s]',
                 feature_mode="vectorizer",
                 vectorizer_type="tfidf", max_features=10000, ngram_range=None,
                 min_df=2, max_df=0.95,
                 embedding_model="all-MiniLM-L6-v2", embedding_batch_size=32,
                 embedding_max_length=256,
                 models=None, hyper_parameter_tuning=0, imputation=True,
                 path_to_save=None, task=None, imbalance_handling="none", original_dataset_name=None, parent_run_id=None, recipe=None,
                 random_seed=42, cv_folds=0, tags=None, final_of_run=None, fixed_params=None, estimates=None, **kwargs):
        """
        Initialize the NLP text classification controller.

        Args:
            name: Run name.
            dataset_path: Path to the uploaded dataset.
            target_column: Name of the target/label column.
            timestamp: Run timestamp.
            text_column: Name of the text column.
            lowercase: Whether to lowercase text.
            remove_stopwords: Whether to remove stopwords.
            lemmatize: Whether to apply lemmatization.
            regex_clean: Whether to apply regex cleaning.
            regex_pattern: Regex pattern for cleaning.
            feature_mode: "vectorizer" or "embedding".
            vectorizer_type: "count", "tfidf", or "hashing".
            max_features: Max number of features for vectorizer.
            ngram_range: N-gram range as [min, max].
            min_df: Min document frequency for vectorizer.
            max_df: Max document frequency for vectorizer.
            embedding_model: Sentence-transformer model name.
            embedding_batch_size: Batch size for embedding.
            embedding_max_length: Max sequence length for embedding.
            models: List of model names to train.
            hyper_parameter_tuning (int): HPT level (0, 1, 2).
            path_to_save: Directory to save run artifacts.
            task: Task type string.
            imbalance_handling: SMOTE, Tomek, etc.
        """
        if ngram_range is None:
            ngram_range = [1, 2]
        if feature_mode not in ("vectorizer", "embedding"):
            raise ValueError(f"Unknown feature_mode '{feature_mode}'. Use 'vectorizer' or 'embedding'.")

        from easyml.utils.run_inputs import recipe_from_sidecars

        # A file cleaned by Data prep carries its frozen recipe next to it (see easyml.dataprep.frozen).
        self.source_path = dataset_path
        recipe = recipe or recipe_from_sidecars(dataset_path)
        self.request_body = NLP_RequestBody(
            name=name,
            task=task or "NLPTextClassification",
            dataset_path=dataset_path,
            target_column=target_column,
            text_column=text_column,
            timestamp=timestamp,
            lowercase=lowercase,
            remove_stopwords=remove_stopwords,
            lemmatize=lemmatize,
            regex_clean=regex_clean,
            regex_pattern=regex_pattern,
            feature_mode=feature_mode,
            vectorizer_type=vectorizer_type,
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            embedding_model=embedding_model,
            embedding_batch_size=embedding_batch_size,
            embedding_max_length=embedding_max_length,
            models=models or [],
            hyper_parameter_tuning=hyper_parameter_tuning,
            imbalance_handling=imbalance_handling,
            imputation=imputation,
            path_to_save=path_to_save,
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

    @staticmethod
    def _check_cancelled():
        if is_cancelled():
            raise InterruptedError("Training cancelled by user")

    def run(self):
        try:
            write_log(f"entered {self.__class__.__name__} - run()", "debug", os.path.basename(__file__))

            create_database_table("runs")
            emit_progress("⚙️ Initializing NLP pipeline...", pct=5)

            nlp_data_loader(self.request_body)
            emit_progress(f"📂 Text data loaded ({len(self.request_body.df)} rows)", pct=10)
            self._check_cancelled()

            nlp_preprocessing_controller(self.request_body)
            emit_progress("🔤 Text preprocessing complete (clean, tokenize, stopwords)", pct=20)
            self._check_cancelled()

            if self.request_body.feature_mode == "embedding":
                emit_progress("🧠 Generating sentence embeddings (this may take a while)...", pct=25)
                embedding_controller(self.request_body)
            else:
                vectorizer_controller(self.request_body)
            emit_progress("📊 Feature engineering complete", pct=35)
            self._check_cancelled()

            emit_progress("🤖 Training NLP models...", pct=45)
            nlp_model_builder_controller(self.request_body)
            from easyml.utils.run_inputs import record_inputs

            body = self.request_body
            record_inputs(body.path_to_save, self.source_path, body.df, [body.text_column], body.target_column,
                          recipe=body.recipe, text_column=body.text_column)
            if body.final_of_run:
                from easyml.utils.run_inputs import inherit_from_evaluated_run

                inherit_from_evaluated_run(body.path_to_save, body.final_of_run, list(body.performance))

            self.request_body.run_id = enter_record_to_table(
                "runs",
                run_name=self.request_body.name,
                folder_path=self.request_body.path_to_save,
                timestamp=self.request_body.timestamp.replace(" ", "T"),
                task=self.request_body.task,
                dataset_name=self.request_body.original_dataset_name or os.path.basename(self.request_body.dataset_path or ""),
                metrics_json=json.dumps(self.request_body.performance, cls=NumpyEncoder),
                parent_run_id=self.request_body.parent_run_id,
                origin=("final" if self.request_body.final_of_run
                        else "retrain" if self.request_body.parent_run_id else None),
                tags=tags_json(self.request_body.tags),
            )
            emit_progress("💾 Results saved to database", pct=95)

            pipeline_summary = {
                "dataset": {
                    "name": self.request_body.original_dataset_name or os.path.basename(self.request_body.dataset_path or ''),
                    "rows": len(self.request_body.df) if self.request_body.df is not None else 0,
                    "cols": len(self.request_body.df.columns) if self.request_body.df is not None else 0,
                },
                "preprocessing": {
                    "lowercase": getattr(self.request_body, 'lowercase', True),
                    "remove_stopwords": getattr(self.request_body, 'remove_stopwords', True),
                    "lemmatize": getattr(self.request_body, 'lemmatize', False),
                    "regex_clean": getattr(self.request_body, 'regex_clean', True),
                },
                "feature_engineering": {
                    "mode": getattr(self.request_body, 'feature_mode', 'vectorizer'),
                    "vectorizer_type": getattr(self.request_body, 'vectorizer_type', 'tfidf'),
                    "embedding_model": getattr(self.request_body, 'embedding_model', ''),
                },
                "imbalance_handling": getattr(self.request_body, 'imbalance_handling', 'none'),
                "split": "all rows (final model)" if self.request_body.final_of_run else "80/20",
                "final_of_run": self.request_body.final_of_run,
                "task": self.request_body.task,
                "models_trained": list(self.request_body.performance.keys()),
                "hyper_parameter_tuning": bool(getattr(self.request_body, 'hyper_parameter_tuning', False)),
            }

            response_payload = {
                "status_code": 200,
                "message": "NLP Classification pipeline completed successfully",
                "run_id": self.request_body.run_id,
                "performance": self.request_body.performance,
                "pipeline_summary": pipeline_summary
            }
            emit_progress("✅ NLP pipeline complete!", pct=100, final=True, result=response_payload)
            return self.request_body

        except InterruptedError:
            raise
        except Exception as e:
            if self.request_body.status_code == 200:
                self.request_body.error_message = "NLPModelTrainingError"
                self.request_body.status_code = NLP_ERROR_CODES.get(
                    self.request_body.error_message, NLP_ERROR_CODES["NLPModelTrainingError"]
                )
                self.request_body.error_traceback = traceback.format_exc()
                write_log(f"error occurred - {self.request_body.error_message} - "
                          f"{self.request_body.error_traceback}",
                          "error", os.path.basename(__file__))
            raise e
