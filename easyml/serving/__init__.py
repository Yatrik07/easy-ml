"""
Serve a trained model as a local REST API: ``easy-ml serve <run>`` / ``easyml.serve(run_id)``.

The service answers ``POST /predict`` in the same format as the "Export as API" package, so a
program written against one works with the other.
"""

from easyml.serving.runner import serve  # noqa: F401

__all__ = ["serve"]
