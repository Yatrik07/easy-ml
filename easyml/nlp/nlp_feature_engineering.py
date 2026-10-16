"""
NLP Feature Engineering Module
===============================
Converts cleaned text into numeric features for model training.

Two modes:
1. Vectorizer mode: CountVectorizer / TF-IDF / HashingVectorizer (sparse output)
2. Embedding mode: Sentence-BERT / HuggingFace models (dense output)

Fitted vectorizers and embedding configs are saved for inference reuse.
"""

import os
from collections import OrderedDict
import json
import traceback
import joblib

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split

from easyml.utils.logger import write_log
from easyml.nlp.nlp_config import NLP_RequestBody, NLP_ERROR_CODES


def _safe_stratify(y):
    """Returns y for stratify if all classes have >= 2 samples, else None."""
    return y if y.value_counts().min() >= 2 else None

# ---------------------------------------- Vectorizer Mode ----------------------------------------

def _create_vectorizer(request_body: NLP_RequestBody):
    """Create a vectorizer instance based on request_body configuration."""

    ngram = tuple(request_body.ngram_range)
    vtype = request_body.vectorizer_type.lower()

    if vtype == "count":
        return CountVectorizer(
            max_features=request_body.max_features,
            ngram_range=ngram,
            min_df=request_body.min_df,
            max_df=request_body.max_df,
        )
    elif vtype == "tfidf":
        return TfidfVectorizer(
            max_features=request_body.max_features,
            ngram_range=ngram,
            min_df=request_body.min_df,
            max_df=request_body.max_df,
        )
    elif vtype == "hashing":
        return HashingVectorizer(
            n_features=request_body.max_features,
            ngram_range=ngram,
            alternate_sign=False,
        )
    else:
        raise ValueError(f"Unknown vectorizer type: {vtype}. Use 'count', 'tfidf', or 'hashing'.")


def vectorizer_controller(request_body: NLP_RequestBody):
    """
    Vectorize text using the configured vectorizer.

    1. Splits text into train/test
    2. Fits vectorizer on training text
    3. Transforms both train and test
    4. Saves vectorizer to disk

    Args:
        request_body (NLP_RequestBody): Must have X populated with cleaned text.

    Raises:
        Exception: Re-raises with NLPFeatureEngineeringError code.
    """
    try:
        write_log("entered vectorizer_controller", "debug", os.path.basename(__file__))

        # Split text before vectorizing to prevent data leakage
        text_train, text_test, y_train, y_test = train_test_split(
            request_body.X, request_body.y,
            test_size=0.2, random_state=request_body.random_seed,
            stratify=_safe_stratify(request_body.y)
        )

        # Preserve text copies for reference
        request_body.test_rows = text_test.index.to_numpy()
        request_body.X_text_train = text_train.reset_index(drop=True)
        request_body.X_text_test = text_test.reset_index(drop=True)

        vectorizer = _create_vectorizer(request_body)

        write_log(f"fitting {request_body.vectorizer_type} vectorizer on {len(text_train)} texts",
                  "info", os.path.basename(__file__))

        request_body.X_train = vectorizer.fit_transform(text_train)
        request_body.X_test = vectorizer.transform(text_test)
        request_body.vectorizer = vectorizer

        request_body.y_train = y_train.reset_index(drop=True)
        request_body.y_test = y_test.reset_index(drop=True)

        write_log(f"vectorizer output: X_train={request_body.X_train.shape}, X_test={request_body.X_test.shape}",
                  "info", os.path.basename(__file__))

        # Save vectorizer (HashingVectorizer is stateless but we save config anyway)
        if request_body.vectorizer_type.lower() != "hashing":
            vectorizer_path = os.path.join(request_body.path_to_save, "vectorizer.pkl")
            joblib.dump(vectorizer, vectorizer_path, protocol=5)
            write_log(f"saved vectorizer to {vectorizer_path}", "info", os.path.basename(__file__))
        else:
            # Save hashing config for reconstruction
            hashing_config = {
                "type": "hashing",
                "n_features": request_body.max_features,
                "ngram_range": request_body.ngram_range,
            }
            config_path = os.path.join(request_body.path_to_save, "vectorizer_config.json")
            json.dump(hashing_config, open(config_path, "w"))
            write_log(f"saved hashing vectorizer config to {config_path}", "info", os.path.basename(__file__))

        # Save feature engineering metadata
        fe_config = {
            "feature_mode": "vectorizer",
            "vectorizer_type": request_body.vectorizer_type,
            "max_features": request_body.max_features,
            "ngram_range": request_body.ngram_range,
            "min_df": request_body.min_df,
            "max_df": request_body.max_df,
        }
        json.dump(fe_config, open(os.path.join(request_body.path_to_save, "fe_config.json"), "w"))

    except Exception as e:
        if request_body.status_code == 200:
            request_body.error_message = "NLPFeatureEngineeringError"
            request_body.status_code = NLP_ERROR_CODES[request_body.error_message]
            request_body.error_traceback = traceback.format_exc()
            write_log(f"error occurred - {request_body.error_message} - {request_body.error_traceback}",
                      "error", os.path.basename(__file__))
        raise e


# ---------------------------------------- Embedding Mode ----------------------------------------

def embedding_controller(request_body: NLP_RequestBody):
    """
    Generate dense embeddings using sentence-transformers.

    1. Splits text into train/test
    2. Encodes text using the specified embedding model
    3. Stores dense arrays in request_body
    4. Saves embedding config for inference

    Args:
        request_body (NLP_RequestBody): Must have X populated with cleaned text.

    Raises:
        Exception: Re-raises with NLPFeatureEngineeringError code.
    """
    try:
        write_log("entered embedding_controller", "debug", os.path.basename(__file__))

        from easyml.capabilities import optional_import

        SentenceTransformer = optional_import("sentence_transformers", "Embedding mode", "nlp").SentenceTransformer


        # Split text before embedding
        text_train, text_test, y_train, y_test = train_test_split(
            request_body.X, request_body.y,
            test_size=0.2, random_state=request_body.random_seed,
            stratify=_safe_stratify(request_body.y)
        )

        request_body.test_rows = text_test.index.to_numpy()
        request_body.X_text_train = text_train.reset_index(drop=True)
        request_body.X_text_test = text_test.reset_index(drop=True)

        model_name = request_body.embedding_model
        write_log(f"loading sentence-transformer model: {model_name}", "info", os.path.basename(__file__))

        model = SentenceTransformer(model_name)
        model.max_seq_length = request_body.embedding_max_length


        write_log(f"encoding {len(text_train)} train texts", "info", os.path.basename(__file__))
        try:
            from easyml.utils.socket_emitter import emit_progress
            emit_progress(f"🔠 Encoding {len(text_train)} train texts (this may take a while)...", pct=28)
        except Exception:
            pass

        request_body.X_train = model.encode(
            text_train.tolist(),
            batch_size=request_body.embedding_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )


        write_log(f"encoding {len(text_test)} test texts", "info", os.path.basename(__file__))
        try:
            emit_progress(f"🔠 Encoding {len(text_test)} test texts...", pct=33)
        except Exception:
            pass

        request_body.X_test = model.encode(
            text_test.tolist(),
            batch_size=request_body.embedding_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )


        request_body.y_train = y_train.reset_index(drop=True)
        request_body.y_test = y_test.reset_index(drop=True)

        write_log(f"embedding output: X_train={request_body.X_train.shape}, X_test={request_body.X_test.shape}",
                  "info", os.path.basename(__file__))

        # Save embedding config
        emb_config = {
            "feature_mode": "embedding",
            "model_name": model_name,
            "batch_size": request_body.embedding_batch_size,
            "max_length": request_body.embedding_max_length,
            "embedding_dim": request_body.X_train.shape[1],
        }
        json.dump(emb_config, open(os.path.join(request_body.path_to_save, "fe_config.json"), "w"))
        write_log(f"saved embedding config", "info", os.path.basename(__file__))

    except Exception as e:
        if request_body.status_code == 200:
            request_body.error_message = "NLPFeatureEngineeringError"
            request_body.status_code = NLP_ERROR_CODES[request_body.error_message]
            request_body.error_traceback = traceback.format_exc()
            write_log(f"error occurred - {request_body.error_message} - {request_body.error_traceback}",
                      "error", os.path.basename(__file__))
        raise e


# ---------------------------------------- Inference Helpers ----------------------------------------

def load_vectorizer_for_inference(run_folder: str):
    """Load a fitted vectorizer from a run folder."""
    vectorizer_path = os.path.join(run_folder, "vectorizer.pkl")
    if os.path.exists(vectorizer_path):
        return joblib.load(vectorizer_path)

    # Check for hashing vectorizer config
    config_path = os.path.join(run_folder, "vectorizer_config.json")
    if os.path.exists(config_path):
        config = json.load(open(config_path, "r"))
        return HashingVectorizer(
            n_features=config["n_features"],
            ngram_range=tuple(config["ngram_range"]),
            alternate_sign=False,
        )

    raise FileNotFoundError(f"No vectorizer found in {run_folder}")


def load_fe_config(run_folder: str) -> dict:
    """Load feature engineering config from a run folder."""
    config_path = os.path.join(run_folder, "fe_config.json")
    if os.path.exists(config_path):
        return json.load(open(config_path, "r"))
    raise FileNotFoundError(f"No fe_config.json found in {run_folder}")


_INFERENCE_CACHE = OrderedDict()
_INFERENCE_CACHE_SIZE = 8


def _cached(key, loader):
    """Keep loaded embedding models in memory between inference requests (least recently used goes first)."""
    if key in _INFERENCE_CACHE:
        _INFERENCE_CACHE.move_to_end(key)
        return _INFERENCE_CACHE[key]
    value = loader()
    _INFERENCE_CACHE[key] = value
    while len(_INFERENCE_CACHE) > _INFERENCE_CACHE_SIZE:
        _INFERENCE_CACHE.popitem(last=False)
    return value


def _load_vectorizer_cached(run_folder):
    from easyml.core.model_cache import cached_load

    path = os.path.join(run_folder, "vectorizer.pkl")
    if not os.path.exists(path):
        path = os.path.join(run_folder, "vectorizer_config.json")
    if not os.path.exists(path):
        return load_vectorizer_for_inference(run_folder)  # raises the usual "not found" error
    return cached_load("vectorizer", path, lambda: load_vectorizer_for_inference(run_folder))


def transform_text_for_inference(text: str, run_folder: str, fe_config: dict):
    """Transform a single preprocessed text string into a one-row feature matrix."""
    return transform_texts_for_inference([text], run_folder, fe_config)


def transform_texts_for_inference(texts, run_folder: str, fe_config: dict):
    """
    Transform preprocessed text strings into features for inference.

    Args:
        texts: Preprocessed text strings.
        run_folder: Path to the run artifacts folder.
        fe_config: Feature engineering configuration dict.

    Returns:
        Feature array suitable for model.predict().
    """
    if fe_config["feature_mode"] == "vectorizer":
        vectorizer = _load_vectorizer_cached(run_folder)
        return vectorizer.transform(list(texts))

    elif fe_config["feature_mode"] == "embedding":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError('Embedding inference needs sentence-transformers: pip install "easy-ml-studio[nlp]"')

        def _load():
            model = SentenceTransformer(fe_config["model_name"])
            if fe_config.get("max_length"):
                model.max_seq_length = fe_config["max_length"]
            return model

        model = _cached(("embedding", fe_config["model_name"], fe_config.get("max_length")), _load)
        return model.encode(list(texts), show_progress_bar=False, normalize_embeddings=True)

    else:
        raise ValueError(f"Unknown feature_mode: {fe_config['feature_mode']}")
