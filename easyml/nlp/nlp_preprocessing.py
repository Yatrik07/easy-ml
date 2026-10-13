"""
NLP Preprocessing Module
=========================
Text cleaning pipeline that operates on pandas Series of raw text.
Each step is conditionally applied based on flags in NLP_RequestBody.

Pipeline: lowercase → regex_clean → stopword_removal → lemmatization
"""

import re
import os
import traceback

import pandas as pd
from easyml.utils.logger import write_log
from easyml.nlp.nlp_config import NLP_RequestBody, NLP_ERROR_CODES

# Lazy-loaded globals to avoid import overhead when not used
_STOPWORDS = {}
_NLP_SPACY = None


class LemmatizerUnavailableError(RuntimeError):
    """spaCy or its English model is not installed."""


def _get_stopwords(source="sklearn"):
    """
    English stopword list.

    New runs use scikit-learn's built-in list (no download needed). Runs created before 0.4.0
    recorded NLTK's list; it is used for them when NLTK and its corpus are available.
    """
    if source not in _STOPWORDS:
        words = None
        if source == "nltk":
            try:
                from nltk.corpus import stopwords
                words = set(stopwords.words('english'))
            except (ImportError, LookupError):
                words = None
        if words is None:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            words = set(ENGLISH_STOP_WORDS)
        _STOPWORDS[source] = words
    return _STOPWORDS[source]


def _get_spacy_nlp():
    """Lazy-load spaCy English model for lemmatization."""
    global _NLP_SPACY
    if _NLP_SPACY is None:
        try:
            import spacy
        except Exception as exc:  # ImportError, or a broken install (e.g. incompatible typer/click)
            raise LemmatizerUnavailableError(
                'Lemmatization needs a working spaCy install: pip install "easy-ml-studio[nlp]"'
            ) from exc
        try:
            _NLP_SPACY = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except OSError as exc:
            raise LemmatizerUnavailableError(
                "Lemmatization needs the spaCy English model. Install it with: easy-ml download-models"
            ) from exc
    return _NLP_SPACY


# ---------------------------------------- Individual Steps ----------------------------------------

def lowercase_text(series: pd.Series) -> pd.Series:
    """Convert all text to lowercase."""
    return series.str.lower()


def apply_regex_clean(series: pd.Series, pattern: str = r'[^a-zA-Z0-9\s]') -> pd.Series:
    """Remove characters matching the regex pattern."""
    return series.apply(lambda text: re.sub(pattern, ' ', str(text)))


def remove_stopwords_from_text(series: pd.Series, source: str = "sklearn") -> pd.Series:
    """Remove English stopwords from text."""
    stops = _get_stopwords(source)
    return series.apply(
        lambda text: ' '.join(word for word in str(text).split() if word.lower() not in stops)
    )


def lemmatize_series(series: pd.Series) -> pd.Series:
    """Lemmatize text using spaCy."""
    nlp = _get_spacy_nlp()

    lemmas = [' '.join(token.lemma_ for token in doc) for doc in nlp.pipe(series.astype(str), batch_size=256)]
    return pd.Series(lemmas, index=series.index)


def strip_extra_whitespace(series: pd.Series) -> pd.Series:
    """Collapse multiple whitespace characters into single spaces and strip."""
    return series.str.replace(r'\s+', ' ', regex=True).str.strip()


# ---------------------------------------- Controller ----------------------------------------

def nlp_preprocessing_controller(request_body: NLP_RequestBody):
    """
    Orchestrates the text preprocessing pipeline.

    Applies cleaning steps in order based on the flags in request_body:
    1. Lowercase
    2. Regex cleaning
    3. Stopword removal
    4. Lemmatization
    5. Whitespace normalization

    Operates on request_body.X (pandas Series of raw text strings).
    The cleaned text overwrites request_body.X in place.

    Args:
        request_body (NLP_RequestBody): Data carrier with text data and preprocessing flags.

    Raises:
        Exception: Logs error and re-raises with NLPPreProcessingError code.
    """
    try:
        write_log("entered nlp_preprocessing_controller", "info", os.path.basename(__file__))

        if request_body.X is None:
            raise ValueError("request_body.X is None — text data must be loaded before preprocessing.")

        text_series = request_body.X.copy()
        steps_applied = []
        from easyml.utils.socket_emitter import emit_progress
        emit_progress("🔤 Applying NLP preprocessing rules...", pct=15)

        # Step 1: Lowercase
        if request_body.lowercase:
            text_series = lowercase_text(text_series)
            steps_applied.append("lowercase")
            write_log("applied lowercase", "info", os.path.basename(__file__))

        # Step 2: Regex cleaning
        if request_body.regex_clean:
            text_series = apply_regex_clean(text_series, request_body.regex_pattern)
            steps_applied.append(f"regex_clean({request_body.regex_pattern})")
            write_log("applied regex cleaning", "info", os.path.basename(__file__))

        # Step 3: Stopword removal
        if request_body.remove_stopwords:
            text_series = remove_stopwords_from_text(text_series, request_body.stopwords_source)
            steps_applied.append("remove_stopwords")
            write_log("applied stopword removal", "info", os.path.basename(__file__))

        # Step 4: Lemmatization
        if request_body.lemmatize:
            try:
                text_series = lemmatize_series(text_series)
                steps_applied.append("lemmatize")
                write_log("applied lemmatization", "info", os.path.basename(__file__))
            except LemmatizerUnavailableError as exc:
                # Skip the step and record it, so inference applies exactly the same cleaning.
                request_body.lemmatize = False
                write_log(f"skipping lemmatization: {exc}", "warning", os.path.basename(__file__))
                emit_progress(f"⚠️ Skipped lemmatization: {exc}", pct=18)

        # Step 5: Whitespace normalization (always applied)
        text_series = strip_extra_whitespace(text_series)

        # Handle any NaN that might have crept in
        text_series = text_series.fillna('')

        request_body.X = text_series
        write_log(f"NLP preprocessing complete. Steps applied: {steps_applied}", "info", os.path.basename(__file__))

    except Exception as e:
        if request_body.status_code == 200:
            request_body.error_message = "NLPPreProcessingError"
            request_body.status_code = NLP_ERROR_CODES[request_body.error_message]
            request_body.error_traceback = traceback.format_exc()

            write_log(f"error occurred- {request_body.error_message} - {request_body.error_traceback}",
                      "error", os.path.basename(__file__))
        raise e


def apply_preprocessing_for_inference(text: str, config: dict) -> str:
    """
    Apply the same preprocessing steps to a single text string during inference.

    Args:
        text (str): Raw text input.
        config (dict): Preprocessing config loaded from config_json.json.

    Returns:
        str: Cleaned text.
    """
    result = str(text)

    if config.get("lowercase", True):
        result = result.lower()

    if config.get("regex_clean", True):
        pattern = config.get("regex_pattern", r'[^a-zA-Z0-9\s]')
        result = re.sub(pattern, ' ', result)

    if config.get("remove_stopwords", True):
        stops = _get_stopwords(config.get("stopwords_source", "nltk"))
        result = ' '.join(word for word in result.split() if word.lower() not in stops)

    if config.get("lemmatize", False):
        nlp = _get_spacy_nlp()
        doc = nlp(result)
        result = ' '.join(token.lemma_ for token in doc)

    # Whitespace normalization
    result = re.sub(r'\s+', ' ', result).strip()

    return result
