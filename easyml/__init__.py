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


__all__ = ["MLTrainer", "load_model", "__version__"]
