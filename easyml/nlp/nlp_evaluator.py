"""
NLP Evaluator Module
====================
Computes classification metrics for NLP models on both train and test sets.
Mirrors easyml/core/evaluator.py for NLP tasks.
"""

import os
import numpy as np
from easyml.nlp.nlp_config import NLP_CLASSIFICATION_METRICS, NLP_METRICS_MAPPING, NLP_RequestBody
from easyml.utils.logger import write_log


def get_nlp_metrics(request_body: NLP_RequestBody, y_real, y_pred, **kwargs):
    """
    Computes all NLP classification metrics.

    Args:
        request_body (NLP_RequestBody): Request context with target encodings.
        y_real (array-like): True labels.
        y_pred (array-like): Predicted labels.
        **kwargs: Additional kwargs (e.g., labels for confusion_matrix ordering).

    Returns:
        dict: Computed metrics.
    """
    metrics = {}

    # Reverse-encode labels for human-readable metrics
    if request_body.target_encodings:
        rev_encodings = {val: key for key, val in request_body.target_encodings.items()}
        y_real = np.array([rev_encodings.get(v, v) for v in y_real])
        y_pred = np.array([rev_encodings.get(v, v) for v in y_pred])

    for metric_name in NLP_CLASSIFICATION_METRICS:
        try:
            metric_fn = NLP_METRICS_MAPPING[metric_name]

            if metric_name == 'confusion_matrix' and 'labels' in kwargs:
                metrics[metric_name] = metric_fn(y_real, y_pred, labels=kwargs['labels'])
                metrics['confusion_matrix_order'] = kwargs['labels']
            else:
                metrics[metric_name] = metric_fn(y_real, y_pred)

        except Exception as e:
            write_log(
                f"Error computing NLP metric '{metric_name}': {e}",
                level='error', module_name=os.path.basename(__file__)
            )

    return metrics


def get_nlp_train_test_metrics(request_body: NLP_RequestBody,
                               y_real_train, y_pred_train,
                               y_real_test, y_pred_test, **kwargs):
    """
    Computes metrics for both training and test sets.

    Args:
        request_body (NLP_RequestBody): Request context.
        y_real_train: True labels for training set.
        y_pred_train: Predictions for training set.
        y_real_test: True labels for test set.
        y_pred_test: Predictions for test set.
        **kwargs: Extra args like label ordering.

    Returns:
        dict: {'train_metrics': {...}, 'test_metrics': {...}}
    """
    train_metrics = get_nlp_metrics(request_body, y_real_train, y_pred_train, **kwargs)
    test_metrics = get_nlp_metrics(request_body, y_real_test, y_pred_test, **kwargs)

    return {'train_metrics': train_metrics, 'test_metrics': test_metrics}
