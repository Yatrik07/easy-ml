"""
Where a schedule's data comes from, and whether it changed.

A source is a file (retrain on it whenever it changes), a folder of files (retrain on the newest file
matching the pattern), or for image runs a folder of class folders. A file only counts once it has
not been modified for ``settle_seconds``, so a file that is still being copied is never used.
"""

import fnmatch
import os
import shutil
import time
import uuid
from dataclasses import dataclass

from easyml.retraining import IMAGE_TASK

DEFAULT_PATTERN = "*.csv;*.xlsx;*.xls"
SETTLE_SECONDS = 10
_TABLE_EXTENSIONS = (".csv", ".xlsx", ".xls")


@dataclass
class SourceData:
    path: str          # the file (or image folder) to train on
    name: str          # shown in events and as the run's dataset name
    signature: str     # changes whenever the data changes


def patterns(text):
    parts = [p.strip() for p in str(text or DEFAULT_PATTERN).replace(",", ";").split(";")]
    return [p for p in parts if p] or DEFAULT_PATTERN.split(";")


def _matches(name, pattern_list):
    lower = name.lower()
    return lower.endswith(_TABLE_EXTENSIONS) and any(fnmatch.fnmatch(lower, p.lower()) for p in pattern_list)


def _settled(mtime, now, settle_seconds):
    return now - mtime >= settle_seconds


def matching_files(folder, pattern=None):
    """Files in ``folder`` (not sub-folders) matching the pattern, newest first: [(name, size, mtime)]."""
    pattern_list = patterns(pattern)
    found = []
    with os.scandir(folder) as entries:
        for entry in entries:
            if entry.is_file() and _matches(entry.name, pattern_list):
                stat = entry.stat()
                found.append((entry.name, stat.st_size, stat.st_mtime))
    found.sort(key=lambda item: (item[2], item[0]), reverse=True)
    return found


def _image_signature(folder):
    count, size, newest = 0, 0, 0.0
    for root, dirs, files in os.walk(folder):
        dirs.sort()
        for name in files:
            try:
                stat = os.stat(os.path.join(root, name))
            except OSError:
                continue
            count, size, newest = count + 1, size + stat.st_size, max(newest, stat.st_mtime)
    return count, size, newest


def find_data(source_path, task, pattern=None, now=None, settle_seconds=SETTLE_SECONDS):
    """
    The data a run would use now, or None when there is none yet (empty folder, file still being written).

    Raises FileNotFoundError when the source itself is missing.
    """
    now = time.time() if now is None else now
    path = os.path.abspath(os.path.expanduser(str(source_path or "")))
    if not source_path or not os.path.exists(path):
        raise FileNotFoundError(f"The data source {source_path!r} does not exist.")
    if task == IMAGE_TASK:
        if not os.path.isdir(path):
            raise FileNotFoundError("Image schedules need a folder with one sub-folder per class.")
        count, size, newest = _image_signature(path)
        if not count or not _settled(newest, now, settle_seconds):
            return None
        return SourceData(path=path, name=os.path.basename(path.rstrip("\\/")), signature=f"{path}|{count}|{size}|{newest:.6f}")
    if os.path.isfile(path):
        stat = os.stat(path)
        if not _settled(stat.st_mtime, now, settle_seconds):
            return None
        return SourceData(path=path, name=os.path.basename(path), signature=f"{path}|{stat.st_size}|{stat.st_mtime:.6f}")
    for name, size, mtime in matching_files(path, pattern):
        if _settled(mtime, now, settle_seconds):
            full = os.path.join(path, name)
            return SourceData(path=full, name=name, signature=f"{full}|{size}|{mtime:.6f}")
    return None


def snapshot(data, uploads_dir):
    """Copy a data file into the uploads folder, so the source can change while the model trains."""
    if os.path.isdir(data.path):
        return data.path
    os.makedirs(uploads_dir, exist_ok=True)
    extension = os.path.splitext(data.path)[1].lower() or ".csv"
    target = os.path.join(uploads_dir, f"sched_{uuid.uuid4().hex[:8]}{extension}")
    shutil.copyfile(data.path, target)
    return target
