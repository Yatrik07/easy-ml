"""
Report which optional features this installation supports.

Checks use ``importlib.util.find_spec`` so heavyweight packages (torch, shap, spaCy) are
never imported just to answer the question.
"""

import importlib.util
from functools import lru_cache

INSTALL_HINTS = {
    "boost": 'pip install "easy-ml-studio[boost]"',
    "nlp": 'pip install "easy-ml-studio[nlp]"',
    "cv": 'pip install "easy-ml-studio[cv]"',
    "eda": 'pip install "easy-ml-studio[eda]"',
    "spacy_model": "easy-ml download-models",
}


def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@lru_cache(maxsize=1)
def get_capabilities():
    from easyml.config.GLOBAL_VARS import (
        ui2back_mappings_classification_models,
        ui2back_mappings_regression_models,
    )
    from easyml.nlp.nlp_config import AVAILABLE_NLP_MODELS

    cv_available = has_module("torch") and has_module("torchvision") and has_module("PIL")
    spacy_available = has_module("spacy")
    features = {
        "cv": cv_available,
        "embeddings": has_module("sentence_transformers"),
        "lemmatize": spacy_available and has_module("en_core_web_sm"),
        "explain": has_module("shap"),
        "eda_report": has_module("ydata_profiling"),
        "excel": has_module("openpyxl"),
    }

    cv_models = []
    if cv_available:
        from easyml.cv.cv_config import AVAILABLE_CV_MODELS
        cv_models = list(AVAILABLE_CV_MODELS)

    return {
        "features": features,
        "models": {
            "MLClassification": list(ui2back_mappings_classification_models.keys()),
            "MLRegression": list(ui2back_mappings_regression_models.keys()),
            "NLPTextClassification": list(AVAILABLE_NLP_MODELS.keys()),
            "ImageClassification": cv_models,
        },
        "install_hints": {
            "cv": INSTALL_HINTS["cv"],
            "embeddings": INSTALL_HINTS["nlp"],
            "lemmatize": INSTALL_HINTS["nlp"] if not spacy_available else INSTALL_HINTS["spacy_model"],
            "explain": INSTALL_HINTS["eda"],
            "eda_report": INSTALL_HINTS["eda"],
            "boost": INSTALL_HINTS["boost"],
        },
    }
