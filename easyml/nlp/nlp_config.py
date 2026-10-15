"""
NLP Configuration Module
========================
Contains all NLP-specific model definitions, hyperparameter grids,
Pydantic data carriers, and UI-to-backend mappings.

Mirrors the structure of easyml/config/GLOBAL_VARS.py for NLP tasks.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Any
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report)
from functools import partial

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError):
    LIGHTGBM_AVAILABLE = False

# ---------------------------------------- NLP Metrics ----------------------------------------

NLP_CLASSIFICATION_METRICS = ['accuracy', 'f1_weighted', 'confusion_matrix', 'classification_report']

NLP_METRICS_MAPPING = {
    'accuracy': accuracy_score,
    'f1_weighted': partial(f1_score, average='weighted'),
    'confusion_matrix': confusion_matrix,
    'classification_report': partial(classification_report, output_dict=True),
}

# ---------------------------------------- NLP Models ----------------------------------------

AVAILABLE_NLP_MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, solver='lbfgs'),
    "naive_bayes": MultinomialNB(),
    "complement_nb": ComplementNB(),
    "svm": LinearSVC(max_iter=2000),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "mlp": MLPClassifier(max_iter=300, early_stopping=True),
}

if LIGHTGBM_AVAILABLE:
    AVAILABLE_NLP_MODELS["light_gbm"] = LGBMClassifier(verbose=-1)

# Models compatible with sparse matrices (vectorizer output)
SPARSE_COMPATIBLE_MODELS = {"logistic_regression", "naive_bayes", "complement_nb", "svm", "random_forest"}

# Models that need dense input (embeddings or converted sparse)
DENSE_ONLY_MODELS = {"mlp", "light_gbm"}

DEFAULT_NLP_MODELS = ["logistic_regression", "naive_bayes", "svm"]

# ---------------------------------------- Hyperparameter Grids ----------------------------------------

NLP_PARAM_GRID_LEVEL1 = {
    "LogisticRegression": {
        "C": [0.1, 1.0],
        "penalty": ["l2"],
    },
    "MultinomialNB": {
        "alpha": [0.1, 1.0],
    },
    "ComplementNB": {
        "alpha": [0.1, 1.0],
    },
    "LinearSVC": {
        "C": [0.1, 1.0],
    },
    "RandomForestClassifier": {
        "n_estimators": [50, 100],
        "max_depth": [None, 20],
    },
    "MLPClassifier": {
        "hidden_layer_sizes": [(100,), (64, 32)],
        "learning_rate_init": [0.001, 0.01],
    },
}

if LIGHTGBM_AVAILABLE:
    NLP_PARAM_GRID_LEVEL1["LGBMClassifier"] = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
    }

NLP_PARAM_GRID_LEVEL2 = {
    "LogisticRegression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l2"],
        "solver": ["lbfgs", "newton-cg"],
    },
    "MultinomialNB": {
        "alpha": [0.01, 0.1, 0.5, 1.0, 2.0],
    },
    "ComplementNB": {
        "alpha": [0.01, 0.1, 0.5, 1.0, 2.0],
    },
    "LinearSVC": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "loss": ["hinge", "squared_hinge"],
    },
    "RandomForestClassifier": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 30],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    },
    "MLPClassifier": {
        "hidden_layer_sizes": [(100,), (128, 64), (64, 32)],
        "learning_rate_init": [0.001, 0.005, 0.01],
        "alpha": [0.0001, 0.001],
    },
}

if LIGHTGBM_AVAILABLE:
    NLP_PARAM_GRID_LEVEL2["LGBMClassifier"] = {
        "n_estimators": [100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [-1, 10, 20],
        "num_leaves": [31, 50],
    }

# ---------------------------------------- UI to Backend Mappings ----------------------------------------

def get_nlp_model_instance(model_name: str):
    """Returns a fresh copy of the named model to avoid state leakage."""
    mapping = {
        "logistic_regression": lambda: LogisticRegression(max_iter=1000, solver='lbfgs'),
        "naive_bayes": lambda: MultinomialNB(),
        "complement_nb": lambda: ComplementNB(),
        "svm": lambda: LinearSVC(max_iter=2000),
        "random_forest": lambda: RandomForestClassifier(n_estimators=100),
        "mlp": lambda: MLPClassifier(max_iter=300, early_stopping=True),
    }
    if LIGHTGBM_AVAILABLE:
        mapping["light_gbm"] = lambda: LGBMClassifier(verbose=-1)

    factory = mapping.get(model_name)
    if factory is None:
        raise ValueError(f"Unknown NLP model: {model_name}")
    return factory()


# ---------------------------------------- NLP Error Codes ----------------------------------------

NLP_ERROR_CODES = {
    "NLPDataIngestionError": 6000,
    "NLPPreProcessingError": 6001,
    "NLPFeatureEngineeringError": 6002,
    "NLPModelTrainingError": 6003,
    "NLPInferenceError": 6004,
}

# ---------------------------------------- Vectorizer Defaults ----------------------------------------

VECTORIZER_DEFAULTS = {
    "type": "tfidf",          # "count", "tfidf", "hashing"
    "max_features": 10000,
    "ngram_range": [1, 2],    # unigrams + bigrams
    "min_df": 2,
    "max_df": 0.95,
}

EMBEDDING_DEFAULTS = {
    "model_name": "all-MiniLM-L6-v2",
    "batch_size": 32,
    "max_length": 256,
}

# ---------------------------------------- Pydantic Data Carriers ----------------------------------------

class NLP_RequestBody(BaseModel):
    """Central data carrier for NLP training pipeline. Accumulates state as it flows through stages."""

    # Run metadata
    task: str = ""
    name: str = ""
    timestamp: str = ""

    # Data
    dataset_path: Optional[str] = None
    df: Optional[pd.DataFrame] = None
    text_column: str = ""
    target_column: str = ""
    X: Optional[Any] = None            # Raw text series, then transformed features
    y: Optional[Any] = None            # Target labels
    X_train: Optional[Any] = None
    X_test: Optional[Any] = None
    y_train: Optional[Any] = None
    y_test: Optional[Any] = None

    target_encodings: Optional[dict] = None

    # Preprocessing flags
    lowercase: bool = True
    remove_stopwords: bool = True
    stopwords_source: str = "sklearn"  # runs created before 0.4.0 used NLTK's list
    lemmatize: bool = False
    regex_clean: bool = True
    regex_pattern: str = r'[^a-zA-Z0-9\s]'
    imputation: bool = True

    # Feature engineering
    feature_mode: str = "vectorizer"   # "vectorizer" or "embedding"

    # Vectorizer config
    vectorizer_type: str = "tfidf"     # "count", "tfidf", "hashing"
    max_features: int = 10000
    ngram_range: List[int] = Field(default_factory=lambda: [1, 2])
    min_df: int = 2
    max_df: float = 0.95

    # Embedding config
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    embedding_max_length: int = 256

    # Model training
    models: Optional[List[str]] = Field(default_factory=list)
    hyper_parameter_tuning: int = 0
    imbalance_handling: str = "none"  # "none", "smote", "tomek", "smotetomek"
    performance: Optional[dict] = Field(default_factory=dict)

    # Paths
    path_to_save: Optional[str] = None

    # Cleaned text (intermediate)
    X_text_train: Optional[Any] = None  # cleaned text for train (preserved for inference reference)
    X_text_test: Optional[Any] = None   # cleaned text for test

    # Status
    status_code: int = 200
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    run_id: Optional[int] = None
    original_dataset_name: Optional[str] = None
    target_dtype: Optional[str] = None
    parent_run_id: Optional[int] = None
    recipe: Optional[dict] = None  # the data-prep recipe that produced the training data, if any
    source_rows: Optional[Any] = None  # data.csv row of each kept row (holdout.json)
    test_rows: Optional[Any] = None    # positions of the test texts in the training frame
    # Kept in memory for the result charts (evaluation.json); never serialized.
    vectorizer: Optional[Any] = None
    charts: dict = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def jsonify(self):
        """Serialize key fields for config_json.json storage."""
        from easyml import __version__
        from easyml.config.constants import CONFIG_VERSION
        from easyml.utils.labels import labels_in_code_order

        extra = {"recipe": self.recipe} if self.recipe else {}
        return {
            **extra,
            "config_version": CONFIG_VERSION,
            "easyml_version": __version__,
            "task": self.task,
            "original_dataset_name": self.original_dataset_name,
            "parent_run_id": self.parent_run_id,
            "imputation": self.imputation,
            "name": self.name,
            "text_column": self.text_column,
            "target_column": self.target_column,
            "target_encodings": self.target_encodings,
            "target_labels": labels_in_code_order(self.target_encodings),
            "target_dtype": self.target_dtype,
            "lowercase": self.lowercase,
            "remove_stopwords": self.remove_stopwords,
            "stopwords_source": self.stopwords_source,
            "lemmatize": self.lemmatize,
            "regex_clean": self.regex_clean,
            "regex_pattern": self.regex_pattern,
            "feature_mode": self.feature_mode,
            "vectorizer_type": self.vectorizer_type,
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "embedding_model": self.embedding_model,
            "embedding_batch_size": self.embedding_batch_size,
            "embedding_max_length": self.embedding_max_length,
            "models": self.models,
            "hyper_parameter_tuning": self.hyper_parameter_tuning,
            "imbalance_handling": self.imbalance_handling,
            "performance": self.performance,
            "status_code": self.status_code,
            "run_id": getattr(self, "run_id", None)
        }


class NLP_InferenceBody(BaseModel):
    """Data carrier for NLP inference requests."""

    task: str = ""
    run_id: int = 0
    data: Any = None        # raw text input (string)
    model_name: str = ""

    target_encodings: Optional[dict] = None
    model: Any = None
    vectorizer: Any = None
    embedding_config: Optional[dict] = None

    # Preprocessing config loaded from run
    preprocessing_config: Optional[dict] = None

    status_code: int = 200
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
