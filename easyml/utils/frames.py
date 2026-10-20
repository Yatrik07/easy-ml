"""Small pandas helpers that give the same answer on pandas 2 and pandas 3."""

import pandas as pd


def is_text(series):
    """
    True for a column of text. pandas 2 reads text as ``object``; pandas 3 gives it the ``str`` dtype
    (a ``pd.StringDtype``), which ``dtype == object`` misses.
    """
    return series.dtype == object or isinstance(series.dtype, pd.StringDtype)


def as_texts(series):
    """The values as a list of strings, missing ones as ``"nan"``: what ``astype(str)`` gave before pandas 3."""
    return [str(value) for value in series.tolist()]
