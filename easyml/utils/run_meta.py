"""A run's own name, notes and tags: what people write to organise and find runs later."""

import json
import re

MAX_TAGS = 10
MAX_TAG_LENGTH = 30
MAX_NOTES = 5000
MAX_NAME = 80
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def parse_tags(raw):
    """Tags stored in the runs table (a JSON list) → list of strings; anything unreadable → []."""
    if not raw:
        return []
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return []
    return [str(tag) for tag in value] if isinstance(value, list) else []


def tags_json(tags):
    """What goes in the ``tags`` column (None when there are none)."""
    return json.dumps(list(tags)) if tags else None


def clean_tags(tags):
    """Trimmed, de-duplicated tags (case-insensitive, first spelling kept). Raises ValueError when too many or long."""
    if tags is None:
        return []
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, (list, tuple)):
        raise ValueError("Tags must be a list of words.")
    cleaned, seen = [], set()
    for tag in tags:
        if tag is None:
            continue
        text = " ".join(_CONTROL.sub(" ", str(tag)).split())
        if not text or text.lower() in seen:
            continue
        if len(text) > MAX_TAG_LENGTH:
            raise ValueError(f"Tags can have at most {MAX_TAG_LENGTH} characters: {text[:40]!r}.")
        seen.add(text.lower())
        cleaned.append(text)
    if len(cleaned) > MAX_TAGS:
        raise ValueError(f"A run can have at most {MAX_TAGS} tags.")
    return cleaned


def update_run_meta(run_id, run_name=None, notes=None, tags=None):
    """
    Change a run's name, notes and/or tags (``None`` leaves a field as it is). Returns the run's summary.
    Raises RunNotFoundError, ValueError.
    """
    from easyml.utils.database_operations import get_record_by_id, update_record
    from easyml.utils.run_summary import run_detail

    get_record_by_id(int(run_id))  # RunNotFoundError for an unknown run
    values = {}
    if run_name is not None:
        name = " ".join(_CONTROL.sub(" ", str(run_name)).split())  # a display name: any text, one line
        if not name:
            raise ValueError("Give the run a name.")
        if len(name) > MAX_NAME:
            raise ValueError(f"Run names can have at most {MAX_NAME} characters.")
        values["run_name"] = name
    if notes is not None:
        text = str(notes).replace("\x00", "").strip()
        if len(text) > MAX_NOTES:
            raise ValueError(f"Notes can have at most {MAX_NOTES} characters.")
        values["notes"] = text or None
    if tags is not None:
        values["tags"] = tags_json(clean_tags(tags))
    if values:
        update_record("runs", int(run_id), **values)
    detail = run_detail(get_record_by_id(int(run_id)))
    detail.pop("metrics", None)
    return detail
