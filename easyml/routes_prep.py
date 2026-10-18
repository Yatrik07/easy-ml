"""
Data prep endpoints: available steps, live preview, apply/download, and saved recipes.

Datasets are uploaded with the usual /load_data. Every dataset path must come from an upload made
through Easy-ML (``resolve_data_file``); cleaned files are written to the uploads folder too, so they
can be trained on like any upload.
"""

import os
import re
import tempfile
import uuid

from flask import Blueprint, current_app, jsonify, request, send_file

from easyml.config.paths import UPLOADS_DIR
from easyml.route_helpers import (
    clean_display_filename,
    ensure_writable_dir,
    error_response,
    resolve_data_file,
)

prep_bp = Blueprint("easyml_prep", __name__)


def _body():
    return request.get_json(silent=True) or {}


def _steps(body):
    steps = body.get("steps") or []
    if not isinstance(steps, list):
        raise ValueError("'steps' must be a list.")
    return steps


def _clean_name(original):
    base = clean_display_filename(original or "") or "dataset.csv"
    stem = re.sub(r"\.(csv|xlsx|xls)$", "", base, flags=re.IGNORECASE)
    stem = re.sub(r"_clean$", "", stem)
    return f"{stem}_clean.csv"


@prep_bp.route("/prep/steps", methods=["GET"])
def prep_steps():
    from easyml.dataprep.steps import CATEGORIES, public_specs

    return jsonify({"categories": list(CATEGORIES), "steps": public_specs()})


@prep_bp.route("/prep/preview", methods=["POST"])
def prep_preview():
    from easyml.dataprep.engine import preview

    body = _body()
    try:
        path = resolve_data_file(body.get("dataset_path"))
        return jsonify(preview(path, _steps(body)))
    except Exception as exc:
        return error_response(exc)


@prep_bp.route("/prep/apply", methods=["POST"])
def prep_apply():
    """Apply the steps to the whole file and save the result as a new upload (ready for training)."""
    from easyml.dataprep.engine import apply_file
    from easyml.utils.data_loader import get_data_info

    body = _body()
    try:
        path = resolve_data_file(body.get("dataset_path"))
        upload_dir = ensure_writable_dir(current_app.config.get("EASYML_UPLOADS_DIR", UPLOADS_DIR))
        output = os.path.join(upload_dir, f"clean_{uuid.uuid4().hex[:8]}.csv")
        frame, reports = apply_file(path, _steps(body), output, freeze=True)
        info = get_data_info(frame, dataset_path=output)
    except Exception as exc:
        return error_response(exc)
    failed = [r for r in reports if r.get("error")]
    return jsonify({**info, "saved_path": output, "original_filename": _clean_name(body.get("original_filename")),
                    "steps": reports, "failed_steps": len(failed)})


@prep_bp.route("/prep/download", methods=["POST"])
def prep_download():
    """The whole file with the steps applied, as a CSV download."""
    from easyml.dataprep.engine import apply_file

    body = _body()
    output = None

    def remove():
        try:
            os.remove(output)
        except (OSError, TypeError):
            pass

    try:
        path = resolve_data_file(body.get("dataset_path"))
        handle, output = tempfile.mkstemp(prefix="easyml_prep_", suffix=".csv")
        os.close(handle)
        apply_file(path, _steps(body), output)
    except Exception as exc:
        remove()
        return error_response(exc)
    response = send_file(output, mimetype="text/csv", as_attachment=True,
                         download_name=_clean_name(body.get("original_filename")))
    response.call_on_close(remove)
    return response


# ---------------------------------------------------------------------- recipes

def _recipe_payload(recipe):
    return {"id": recipe.id, "name": recipe.name, "steps": recipe.steps, "source_columns": recipe.source_columns}


@prep_bp.route("/prep/recipes", methods=["GET", "POST"])
def prep_recipes():
    from easyml.dataprep.recipe import Recipe, list_recipes

    try:
        if request.method == "GET":
            return jsonify({"recipes": list_recipes()})
        body = _body()
        recipe = Recipe(name=str(body.get("name") or "").strip(), steps=_steps(body),
                        source_columns=body.get("source_columns") or [])
        recipe.save(overwrite=bool(body.get("overwrite")))
        return jsonify(_recipe_payload(recipe)), 201
    except Exception as exc:
        return error_response(exc)


@prep_bp.route("/prep/recipes/<int:recipe_id>", methods=["GET", "PUT", "DELETE"])
def prep_recipe(recipe_id):
    from easyml.dataprep.recipe import Recipe, delete_recipe

    try:
        if request.method == "GET":
            return jsonify(_recipe_payload(Recipe.load(recipe_id)))
        if request.method == "DELETE":
            delete_recipe(recipe_id)
            return jsonify({"deleted": recipe_id})
        body = _body()
        current = Recipe.load(recipe_id)
        recipe = Recipe(name=str(body.get("name") or current.name).strip(), steps=_steps(body) or current.steps,
                        id=recipe_id, source_columns=body.get("source_columns") or current.source_columns)
        recipe.save()
        return jsonify(_recipe_payload(recipe))
    except LookupError as exc:
        return jsonify({"status": "error", "status_code": 404, "error_message": str(exc), "message": str(exc)}), 404
    except Exception as exc:
        return error_response(exc)
