"""
Housekeeping for the uploads folder.

Every upload (and every cleaned or scheduled copy) is saved under an internal name such as
``data_1a2b3c4d.csv``. Runs keep their own copy of the data (``data.csv``), so an upload is only needed
while it is being explored, cleaned or trained on. Only files with these internal names are ever
deleted here: anything else a user put in the folder, and every sub-folder, is left alone.
"""

import os
import re
import time

from easyml.config.paths import UPLOADS_DIR

# Uploads, cleaned copies, schedule snapshots, and the side files of cleaned copies (frozen recipe, raw rows).
INTERNAL_UPLOAD = re.compile(r"^(data|data_nlp|clean|sched)_[0-9a-f]{8}(\.(csv|xlsx|xls)|\.raw\.csv|\.prep\.json)$")
DAY = 24 * 3600


def _internal_files(folder):
    try:
        entries = list(os.scandir(folder))
    except OSError:
        return []
    found = []
    for entry in entries:
        try:
            if entry.is_file(follow_symlinks=False) and INTERNAL_UPLOAD.match(entry.name):
                stat = entry.stat(follow_symlinks=False)
                found.append((entry.path, stat.st_size, stat.st_mtime))
        except OSError:
            continue
    return found


def uploads_usage(older_than_hours=24, folder=UPLOADS_DIR, now=None):
    """``{"files", "bytes", "old_files", "old_bytes"}`` of the uploads Easy-ML created."""
    now = time.time() if now is None else now
    cutoff = now - older_than_hours * 3600
    files = _internal_files(folder)
    old = [f for f in files if f[2] < cutoff]
    return {"files": len(files), "bytes": sum(f[1] for f in files),
            "old_files": len(old), "old_bytes": sum(f[1] for f in old), "older_than_hours": older_than_hours}


def clean_uploads(older_than_hours=24, folder=UPLOADS_DIR, now=None):
    """Delete the uploads Easy-ML created that are older than ``older_than_hours``. Returns what was freed."""
    now = time.time() if now is None else now
    cutoff = now - older_than_hours * 3600
    deleted, freed = 0, 0
    for path, size, mtime in _internal_files(folder):
        if mtime >= cutoff:
            continue
        try:
            os.remove(path)
        except OSError:  # in use (Windows) or already gone: try again next time
            continue
        deleted, freed = deleted + 1, freed + size
    return {"deleted": deleted, "freed_bytes": freed}
