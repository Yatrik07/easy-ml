"""
Keep recently used models in memory so repeated predictions do not reload them from disk.

Entries are keyed by the file's path, size and modification time, so a file that is replaced
(re-trained, re-imported) is loaded again instead of being served stale. The cache is bounded by
entry count and by the total size of the cached files; anything larger than the budget is simply
not cached. Cached objects are shared, so callers must only read them (predict / transform).

    EASYML_MODEL_CACHE      maximum number of cached models (default 8, 0 disables the cache)
    EASYML_MODEL_CACHE_MB   maximum total size of the cached files in MB (default 1024)
"""

import os
import threading
from collections import OrderedDict

_entries = OrderedDict()  # key -> (value, size_bytes)
_lock = threading.Lock()
_loading = {}  # key -> Lock, so two requests for the same model load it only once


def _limits():
    try:
        count = int(os.environ.get("EASYML_MODEL_CACHE", "8"))
    except ValueError:
        count = 8
    try:
        megabytes = float(os.environ.get("EASYML_MODEL_CACHE_MB", "1024"))
    except ValueError:
        megabytes = 1024
    return max(count, 0), int(max(megabytes, 0) * 1024 * 1024)


def _norm(path):
    return os.path.normcase(os.path.abspath(path))


def cached_load(kind, path, loader, extra=(), size=None):
    """
    Return ``loader()``'s result for ``path``, reusing the object loaded earlier if the file is unchanged.

    ``kind`` and ``extra`` separate different objects built from the same file (e.g. a model class).
    ``size`` is the memory to count against the budget (default: the file size).
    """
    max_count, max_bytes = _limits()
    stat = os.stat(path)
    cost = stat.st_size if size is None else size
    if max_count == 0 or cost > max_bytes:
        return loader()
    key = (kind, _norm(path), stat.st_size, stat.st_mtime_ns) + tuple(extra)
    with _lock:
        if key in _entries:
            _entries.move_to_end(key)
            return _entries[key][0]
        key_lock = _loading.setdefault(key, threading.Lock())
    with key_lock:
        with _lock:
            if key in _entries:
                _entries.move_to_end(key)
                return _entries[key][0]
        value = loader()
        with _lock:
            _entries[key] = (value, cost)
            _loading.pop(key, None)
            total = sum(entry_size for _, entry_size in _entries.values())
            while _entries and (len(_entries) > max_count or total > max_bytes):
                _, (_, entry_size) = _entries.popitem(last=False)
                total -= entry_size
    return value


def forget_folder(folder):
    """Drop every cached object loaded from ``folder`` (a deleted run)."""
    prefix = _norm(folder).rstrip("\\/") + os.sep
    with _lock:
        for key in [k for k in _entries if k[1].startswith(prefix)]:
            del _entries[key]


def clear():
    with _lock:
        _entries.clear()


def cached_count():
    with _lock:
        return len(_entries)
