"""Easy-ML Studio: a no-code machine learning studio with a browser UI."""

__version__ = "0.5.0"

from easyml.trainer import MLTrainer  # noqa: E402


def load_model(run_id, model_name=None):
    """
    Load a model trained in Easy-ML Studio and use it from Python.

        model = easyml.load_model(5)          # best model of run #5
        model.predict({"age": 45, "city": "NY"})

    See :func:`easyml.predictor.load_model` for the accepted inputs.
    """
    from easyml.predictor import load_model as _load_model  # imported lazily: keeps `import easyml` light

    return _load_model(run_id, model_name=model_name)


def retrain(run_id, dataset_path, name=None, models=None):
    """
    Train a new run on new data with the same settings as an earlier run.

        result = easyml.retrain(5, "customers_october.csv")
        result.load_model().predict("new_customers.csv")

    See :func:`easyml.retraining.retrain`.
    """
    from easyml.retraining import retrain as _retrain

    return _retrain(run_id, dataset_path, name=name, models=models)


__all__ = ["MLTrainer", "load_model", "retrain", "__version__"]
