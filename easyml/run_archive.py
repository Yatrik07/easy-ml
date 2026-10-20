"""
Move a trained run between machines as a single .zip file.

    path = easyml.export_run(5)                          # easyml-run-5-churn.zip in the current folder
    new_id = easyml.import_run(path, trusted=True)       # on the other machine
    easyml.load_model(new_id).predict("new_customers.csv")

Archive layout (format version 1)::

    manifest.json        what the run is, which files it holds (with sha256), library versions
    run/<file>           the run folder's files: models, preprocessing, configs, data.csv, evaluation.json

Saved configs are cleaned of absolute paths. Importing never loads a model, but model files
(``.pkl``) can run code when they are loaded later, so imports must be confirmed explicitly.
"""

import hashlib
import json
import os
import platform
import re
import shutil
import uuid
import warnings
import zipfile
from datetime import datetime

from easyml.utils.run_meta import MAX_NOTES, clean_tags, parse_tags, tags_json
from easyml.utils.database_operations import create_database_table, enter_record_to_table, get_record_by_id
from easyml.utils.logger import write_log
from easyml.utils.run_summary import TASK_LABELS, display_dataset_name, normalize_timestamp, parse_metrics
from easyml.utils.utils import clean_display_filename, sanitize_run_name

FORMAT = "easyml-run"
FORMAT_VERSION = 1
MANIFEST = "manifest.json"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,120}$")
ALLOWED_SUFFIXES = (".csv", ".json", ".pkl", ".cbm", ".pth")
MAX_ENTRIES = 256
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_RATIO = 200
TRUST_WARNING = ("Only import runs from people you trust: model files can run code on this computer "
                 "when they are loaded.")

_CATBOOST = {"CatBoostClassifier", "CatBoostRegressor"}
_TABULAR = ("MLClassification", "MLRegression")


class ArchiveError(ValueError):
    """The file is not a valid Easy-ML run archive."""


def max_import_bytes():
    """Largest archive (uncompressed) accepted by import; ``EASYML_MAX_IMPORT_MB`` overrides 2 GB."""
    try:
        megabytes = float(os.environ.get("EASYML_MAX_IMPORT_MB", "2048"))
    except ValueError:
        megabytes = 2048
    return int(megabytes * 1024 * 1024)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strip_model_paths(metrics):
    return {key: ({k: v for k, v in values.items() if k != "model_path"} if isinstance(values, dict) else values)
            for key, values in (metrics or {}).items()}


def _libraries(task, metrics, config):
    from easyml.utils.export_utils import _MODEL_PACKAGES, installed_version

    names = ["scikit-learn", "pandas", "numpy", "joblib"]
    names += sorted({_MODEL_PACKAGES[m] for m in metrics if m in _MODEL_PACKAGES})
    if config.get("feature_mode") == "embedding":
        names.append("sentence-transformers")
    if task == "ImageClassification":
        names += ["torch", "torchvision"]
    return {name: installed_version(name) for name in names if installed_version(name)}


# --------------------------------------------------------------------------- export

def _safe_tags(tags):
    try:
        return clean_tags(tags if isinstance(tags, list) else [])
    except ValueError:  # an archive from elsewhere with odd tags: keep the run, drop the tags
        return []


def export_run(run_id, output_path=None, include_data=True):
    """
    Write run ``run_id`` to a .zip archive and return its path.

    Args:
        output_path: Where to write the archive (default: ``easyml-run-<id>-<name>.zip`` in the current folder).
        include_data: Include the training data (``data.csv``, and ``raw_data.csv`` for runs trained on
            cleaned data). Explanations (SHAP), "Try a real row" and retraining from the imported run use it.
    """
    from easyml import __version__
    from easyml.core.artifacts import read_json

    record = get_record_by_id(int(run_id))
    folder, task = record["folder_path"], record["task"]
    if not folder or not os.path.isdir(folder):
        raise FileNotFoundError(f"The files of run #{run_id} are missing.")
    metrics = _strip_model_paths(parse_metrics(record.get("metrics_json")))

    config = read_json(folder, "config_json.json")
    has_data = include_data and task != "ImageClassification" and os.path.isfile(os.path.join(folder, "data.csv"))
    config["dataset_path"] = "data.csv" if has_data else None
    if "performance" in config:
        config["performance"] = _strip_model_paths(config["performance"])

    if output_path is None:
        output_path = os.path.abspath(f"easyml-run-{int(run_id)}-{sanitize_run_name(record.get('run_name'))}.zip")
    output_path = os.fspath(output_path)  # a pathlib.Path works too

    entries = []
    for entry in sorted(os.scandir(folder), key=lambda e: e.name):
        name = entry.name
        if not entry.is_file() or name == "config_json.json":
            continue
        if not NAME_RE.match(name) or not name.lower().endswith(ALLOWED_SUFFIXES):
            write_log(f"Export skips {name}", "info", os.path.basename(__file__))
            continue
        if name in ("data.csv", "raw_data.csv") and not has_data:
            continue  # raw_data.csv holds the same rows before cleaning
        entries.append((name, entry.path))

    config_bytes = json.dumps(config, indent=2, default=str).encode("utf-8")
    files = [{"name": "config_json.json", "size": len(config_bytes),
              "sha256": hashlib.sha256(config_bytes).hexdigest()}]
    files += [{"name": name, "size": os.path.getsize(path), "sha256": _sha256(path)} for name, path in entries]

    manifest = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "easyml_version": __version__,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "run": {
            "id": int(run_id),
            "run_name": record.get("run_name"),
            "task": task,
            "dataset_name": display_dataset_name(record.get("dataset_name")),
            "timestamp": normalize_timestamp(record.get("timestamp")),
            "parent_run_id": record.get("parent_run_id"),
            "origin": record.get("origin"),
            "notes": record.get("notes") or "",
            "tags": parse_tags(record.get("tags")),
        },
        "metrics": metrics,
        "includes_data": has_data,
        "files": files,
        "environment": {"python": platform.python_version(), "libraries": _libraries(task, metrics, config)},
    }

    temp_path = output_path + ".part"
    with zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MANIFEST, json.dumps(manifest, indent=2, default=str))
        archive.writestr("run/config_json.json", config_bytes)
        for name, path in entries:
            archive.write(path, f"run/{name}")
    os.replace(temp_path, output_path)
    write_log(f"Exported run {run_id} to {output_path}", "info", os.path.basename(__file__))
    return output_path


# --------------------------------------------------------------------------- import

def _validate_entries(archive):
    infos = archive.infolist()
    if len(infos) > MAX_ENTRIES:
        raise ArchiveError("The archive holds too many files.")
    limit = max_import_bytes()
    seen, files, total = set(), {}, 0
    for info in infos:
        name = info.filename
        if info.is_dir():
            if name == "run/":
                continue
            raise ArchiveError(f"Unexpected folder in the archive: {name}")
        if info.flag_bits & 0x1:
            raise ArchiveError("Encrypted archives are not supported.")
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise ArchiveError(f"The archive contains a link: {name}")
        if name != MANIFEST:
            base = name[4:] if name.startswith("run/") else None
            if (base is None or "/" in base or "\\" in base or not NAME_RE.match(base)
                    or not base.lower().endswith(ALLOWED_SUFFIXES)):
                raise ArchiveError(f"Unexpected file in the archive: {name}")
        if name.lower() in seen:
            raise ArchiveError(f"The archive contains {name} twice.")
        seen.add(name.lower())
        total += info.file_size
        if total > limit:
            raise ArchiveError("The archive is larger than the import limit (EASYML_MAX_IMPORT_MB).")
        if info.file_size > 50 * 1024 * 1024 and info.compress_size and info.file_size / info.compress_size > MAX_RATIO:
            raise ArchiveError(f"{name} is suspiciously compressible; refusing to unpack it.")
        if name != MANIFEST:
            files[name[4:]] = info
    if MANIFEST not in seen:
        raise ArchiveError("This is not an Easy-ML run archive (manifest.json is missing).")
    return files


def _read_small_json(archive, name):
    info = archive.getinfo(name)
    if info.file_size > MAX_MANIFEST_BYTES:
        raise ArchiveError(f"{name} is too large.")
    try:
        return json.loads(archive.read(name).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ArchiveError(f"{name} is not valid JSON.") from exc


def _validate_manifest(manifest):
    from easyml.core.artifacts import validate_model_name

    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise ArchiveError("This is not an Easy-ML run archive.")
    version = manifest.get("format_version")
    if not isinstance(version, int):
        raise ArchiveError("The archive has no valid format version.")
    if version > FORMAT_VERSION:
        raise ArchiveError("This run was exported by a newer Easy-ML. Upgrade easy-ml-studio to import it.")
    run = manifest.get("run")
    if not isinstance(run, dict) or run.get("task") not in TASK_LABELS:
        raise ArchiveError("The archive does not describe a supported run.")
    metrics = manifest.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ArchiveError("The archive lists no trained models.")
    for model_name in metrics:
        try:
            validate_model_name(model_name)
        except ValueError as exc:
            raise ArchiveError(str(exc)) from exc
    files = manifest.get("files")
    if not isinstance(files, list) or not all(isinstance(f, dict) and "name" in f and "sha256" in f for f in files):
        raise ArchiveError("The archive's file list is invalid.")


def _model_files(task, model_name, names):
    if task == "ImageClassification":
        return [f"{model_name}.pth"]
    if model_name in _CATBOOST:
        return [n for n in (f"{model_name}.cbm", f"{model_name}.pkl.cbm") if n in names][:1] or [f"{model_name}.cbm"]
    return [f"{model_name}.pkl"]


def _check_required(archive, task, metrics, names):
    required = ["config_json.json"]
    if task in _TABULAR:
        required += ["column_data.json", "preprocessing_pipeline.pkl"]
    elif task == "NLPTextClassification":
        required.append("fe_config.json")
        if "fe_config.json" in names:
            fe_config = _read_small_json(archive, "run/fe_config.json")
            if fe_config.get("feature_mode", "vectorizer") == "vectorizer":
                if "vectorizer.pkl" not in names and "vectorizer_config.json" not in names:
                    required.append("vectorizer.pkl")
    for model_name in metrics:
        required += _model_files(task, model_name, names)
    missing = [n for n in required if n not in names]
    if missing:
        raise ArchiveError("The archive is missing files: " + ", ".join(missing))


def _environment_warnings(task, manifest):
    from easyml import __version__
    from easyml.capabilities import get_capabilities
    from easyml.utils.export_utils import installed_version

    notes = []
    for library, version in ((manifest.get("environment") or {}).get("libraries") or {}).items():
        here = installed_version(library)
        if here is None:
            notes.append(f"{library} {version} was used to train this run but is not installed here.")
        elif str(here).split(".")[:2] != str(version).split(".")[:2]:
            notes.append(f"Trained with {library} {version}; this computer has {here}. Predictions may differ or fail.")
    if task == "ImageClassification" and not get_capabilities()["features"]["cv"]:
        notes.append("Image runs need PyTorch: " + get_capabilities()["install_hints"]["cv"])
    exported_with = manifest.get("easyml_version")
    if exported_with and exported_with.split(".")[:2] != __version__.split(".")[:2]:
        notes.append(f"Exported by Easy-ML {exported_with}; this is {__version__}.")
    return notes


def import_archive(path, name=None):
    """
    Import a run archive without asking for confirmation; returns ``(run_id, warnings)``.

    Callers must have confirmed with the user that the archive comes from a trusted source.
    """
    from easyml.config.paths import RUNS_DIR

    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    if not zipfile.is_zipfile(path):
        raise ArchiveError("This is not a .zip file.")

    staging = os.path.join(RUNS_DIR, f".import-{uuid.uuid4().hex}")
    final = None
    try:
        with zipfile.ZipFile(path) as archive:
            infos = _validate_entries(archive)
            manifest = _read_small_json(archive, MANIFEST)
            _validate_manifest(manifest)
            run, metrics = manifest["run"], manifest["metrics"]
            task = run["task"]
            listed = {f["name"]: f for f in manifest["files"]}
            if set(listed) != set(infos):
                raise ArchiveError("The archive's files do not match its manifest.")
            _check_required(archive, task, metrics, set(infos))
            if "prep.json" in infos:  # the frozen data-prep recipe must be one this version can run
                from easyml.dataprep.frozen import FrozenRecipe

                try:
                    FrozenRecipe.from_dict(_read_small_json(archive, "run/prep.json"))
                except ValueError as exc:
                    raise ArchiveError(f"The run's data-prep recipe (prep.json) cannot be used: {exc}") from exc

            os.makedirs(staging)
            limit, written = max_import_bytes(), 0
            for file_name, info in infos.items():
                digest = hashlib.sha256()
                target = os.path.join(staging, file_name)
                file_written = 0
                with archive.open(info) as source, open(target, "wb") as sink:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        written += len(chunk)
                        file_written += len(chunk)
                        if written > limit or file_written > info.file_size:
                            raise ArchiveError("The archive is larger than it claims to be.")
                        digest.update(chunk)
                        sink.write(chunk)
                if digest.hexdigest() != listed[file_name]["sha256"]:
                    raise ArchiveError(f"{file_name} is damaged (checksum mismatch).")

        run_name = (clean_display_filename(name) if name else None) or clean_display_filename(run.get("run_name")) or "run"
        stamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        final = os.path.join(RUNS_DIR, f"{sanitize_run_name(run_name)}_imported_{stamp}")
        while os.path.exists(final):
            final = f"{final}_{uuid.uuid4().hex[:4]}"
        os.replace(staging, final)

        config_path = os.path.join(final, "config_json.json")
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        has_data = os.path.isfile(os.path.join(final, "data.csv"))
        config["dataset_path"] = os.path.join(final, "data.csv") if has_data else None
        config["parent_run_id"] = None
        config["imported_from"] = {
            "run_id": run.get("id"),
            "parent_run_id": run.get("parent_run_id"),
            "easyml_version": manifest.get("easyml_version"),
            "exported_at": manifest.get("exported_at"),
        }
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, default=str)

        create_database_table("runs")
        run_id = enter_record_to_table(
            "runs",
            run_name=run_name,
            folder_path=final,
            timestamp=normalize_timestamp(run.get("timestamp")) or datetime.now().isoformat(timespec="seconds"),
            task=task,
            dataset_name=clean_display_filename(run.get("dataset_name")) or "",
            metrics_json=json.dumps(_strip_model_paths(metrics)),
            parent_run_id=None,
            origin="import",
            notes=(str(run.get("notes") or "")[:MAX_NOTES] or None),
            tags=tags_json(_safe_tags(run.get("tags"))),
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if final:
            shutil.rmtree(final, ignore_errors=True)
        raise

    notes = _environment_warnings(task, manifest)
    write_log(f"Imported run archive {path} as run {run_id}", "info", os.path.basename(__file__))
    return run_id, notes


def import_run(path, name=None, trusted=False):
    """
    Import a run archive made by :func:`export_run` and return the new run id.

    Args:
        path: The .zip file.
        name: Name for the imported run (default: the original name).
        trusted: Must be True. Model files can run code when they are loaded, so only import
            archives from people you trust.
    """
    if not trusted:
        raise PermissionError(TRUST_WARNING + " Pass trusted=True to import this archive.")
    run_id, notes = import_archive(path, name=name)
    for note in notes:
        warnings.warn(note, UserWarning, stacklevel=2)
    return run_id
