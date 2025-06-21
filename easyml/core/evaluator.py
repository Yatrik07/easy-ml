from functools import partial
import os
import numpy as np
from easyml.config.GLOBAL_VARS import REGRESSION_METRICS, CLASSIFICATION_METRICS, METRICS_MAPPING, ML_RequestBody
from easyml.utils.logger import write_log
def get_classification_metrics(y_real, y_pred):
    metrics = {}
    for metric in CLASSIFICATION_METRICS:
        try:
            metrics[metric] = METRICS_MAPPING[metric](*[y_real, y_pred])
        except Exception as e:
            write_log(f"Error in get_classification_metrics for metric {metric}. Error: {e}",level='error', module_name=os.path.basename(__file__))


def get_regression_metrics(y_real, y_pred):
    metrics = {}
    for metric in REGRESSION_METRICS:
        try:
            metrics[metric] = METRICS_MAPPING[metric](*[y_real, y_pred])
        except Exception as e:
            write_log(f"Error in get_classification_metrics for metric {metric}. Error: {e}",level='error', module_name=os.path.basename(__file__))


def get_metrics(request_body : ML_RequestBody, task, y_real, y_pred):
    metrics_to_include = REGRESSION_METRICS if task == 'regression' else CLASSIFICATION_METRICS
    metrics = {}
    if task == 'classification':
        target_encodings = request_body.target_encodings
        rev_target_encodings = {val: key for key, val in target_encodings.items()}
        y_real = np.array([rev_target_encodings[tmp] for tmp in y_real])
        y_pred = np.array([rev_target_encodings[tmp] for tmp in y_pred])

    for metric in metrics_to_include:
        try:
            metrics[metric] = METRICS_MAPPING[metric](*[y_real, y_pred])
        except Exception as e:
            write_log(f"Error in get_metrics for task {task} for metric {metric}. Error: {e}",level='error', module_name=os.path.basename(__file__))
    return metrics

def get_train_test_metrics(request_body : ML_RequestBody,task, y_real_train, y_pred_train, y_real_test, y_pred_test):
    train_metrics = get_metrics(request_body, task, y_real_train, y_pred_train)
    test_metrics = get_metrics(request_body, task, y_real_test, y_pred_test)
    return {'train_metrics' : train_metrics, 'test_metrics' : test_metrics}