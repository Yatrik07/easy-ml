"""Easy-ML Studio: a no-code machine learning studio with a browser UI."""

__version__ = "0.7.0"

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


def export_run(run_id, output_path=None, include_data=True):
    """Save a run as a .zip archive that :func:`import_run` can load on another computer."""
    from easyml.run_archive import export_run as _export_run

    return _export_run(run_id, output_path=output_path, include_data=include_data)


def import_run(path, name=None, trusted=False):
    """Import a run archive and return its new run id. Only import archives from people you trust."""
    from easyml.run_archive import import_run as _import_run

    return _import_run(path, name=name, trusted=trusted)


def serve(run_id, model_name=None, host="127.0.0.1", port=8000, threads=4, api_key=None, cors_origins=None,
          max_mb=20, access_log=False):
    """
    Serve a trained run as a REST API (blocks until stopped). See :func:`easyml.serving.runner.serve`.

        easyml.serve(5)   # http://127.0.0.1:8000/predict
    """
    from easyml.serving.runner import serve as _serve

    return _serve(run_id, model_name=model_name, host=host, port=port, threads=threads, api_key=api_key,
                  cors_origins=cors_origins, max_mb=max_mb, access_log=access_log)


__all__ = ["MLTrainer", "load_model", "retrain", "export_run", "import_run", "serve", "__version__"]
