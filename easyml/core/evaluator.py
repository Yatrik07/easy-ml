import os
import numpy as np
from easyml.config.GLOBAL_VARS import REGRESSION_METRICS, CLASSIFICATION_METRICS, METRICS_MAPPING, ML_RequestBody
from easyml.utils.logger import write_log

def get_metrics(request_body : ML_RequestBody, task, y_real, y_pred, **kwargs):
    """
    Computes metrics for a given task using real and predicted values.

    Args:
        request_body (ML_RequestBody): The request context including target encodings.
        task (str): Either 'classification' or 'regression'.
        y_real (array-like): True labels or values.
        y_pred (array-like): Predicted labels or values.

    Returns:
        dict: Computed metrics for the specified task.
    """

    metrics_to_include = REGRESSION_METRICS if task == 'regression' else CLASSIFICATION_METRICS
    metrics = {}
    if task == 'classification':
        target_encodings = request_body.target_encodings
        rev_target_encodings = {val: key for key, val in target_encodings.items()}
        y_real = np.array([rev_target_encodings[tmp] for tmp in y_real])
        y_pred = np.array([rev_target_encodings[tmp] for tmp in y_pred])

    for metric in metrics_to_include:
        if metric == 'confusion_matrix':
            kwargs_ = {'labels' : kwargs['labels']}
        else:
            kwargs_ = {}
        try:
            metrics[metric] = METRICS_MAPPING[metric](*[y_real, y_pred], **kwargs_)
            if metric == 'confusion_matrix':
                metrics['confusion_matrix_order'] = kwargs['labels']
        except Exception as e:
            write_log(f"Error in get_metrics for task {task} for metric {metric}. Error: {e}",level='error', module_name=os.path.basename(__file__))
    return metrics

def get_train_test_metrics(request_body : ML_RequestBody,task, y_real_train, y_pred_train, y_real_test, y_pred_test, **kwargs):
    """
    Computes both training and test metrics for a model run.

    Args:
        request_body (ML_RequestBody): Request context with target encodings.
        task (str): Either 'classification' or 'regression'.
        y_real_train (array-like): True labels for training data.
        y_pred_train (array-like): Predictions for training data.
        y_real_test (array-like): True labels for test data.
        y_pred_test (array-like): Predictions for test data.

    Returns:
        dict: Dictionary containing 'train_metrics' and 'test_metrics'.
    """

    train_metrics = get_metrics(request_body, task, y_real_train, y_pred_train, **kwargs)
    test_metrics = get_metrics(request_body, task, y_real_test, y_pred_test, **kwargs)
    return {'train_metrics' : train_metrics, 'test_metrics' : test_metrics}