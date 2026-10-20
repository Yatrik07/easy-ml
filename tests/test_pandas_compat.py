"""pandas 3 reads text as the ``str`` dtype instead of ``object``: text columns must still be found and cleaned."""

import io

import numpy as np
import pandas as pd

from easyml.core.artifacts import coerce_feature_frame
from easyml.dataprep.frozen import FrozenRecipe
from easyml.dataprep.steps import clean_text
from easyml.monitoring.datacheck import build_profile, check_rows
from easyml.utils.frames import as_texts, is_text


def _text(values):
    """A column with the dtype pandas 3 gives text (on older pandas, the closest string dtype)."""
    try:
        dtype = pd.StringDtype(na_value=np.nan)
    except TypeError:  # pandas < 2.3
        dtype = pd.StringDtype()
    return pd.Series(values, dtype=dtype)


def test_text_columns_are_found_whatever_their_dtype():
    assert is_text(pd.Series(["a", None], dtype=object)) and is_text(_text(["a", None]))
    assert not is_text(pd.Series([1.5, 2.0])) and not is_text(pd.Series(["a"], dtype="category"))
    read = pd.read_csv(io.StringIO("t,n\nhello,1\n,2\n"))
    assert as_texts(read["t"]) == ["hello", "nan"]  # what astype(str) gave before pandas 3


def test_clean_text_cleans_every_text_column_by_default():
    frame = pd.DataFrame({"city": _text([" NY ", None]), "n": [1, 2]})
    cleaned, _ = clean_text(frame)
    assert cleaned["city"][0] == "NY" and pd.isna(cleaned["city"][1])
    _, _, frozen = FrozenRecipe.fit(frame, [{"op": "clean_text", "params": {}}])
    assert frozen.steps[0]["frozen"]["params"]["columns"] == ["city"]


def test_blank_text_is_missing_in_the_form_and_the_data_check():
    form = pd.DataFrame({"city": _text(["  "]), "age": _text(["41"])})
    coerced = coerce_feature_frame(form, {"column_dtypes": {"city": "object", "age": "int64"}})
    assert pd.isna(coerced["city"][0]) and coerced["age"][0] == 41
    profile = build_profile(pd.DataFrame({"city": ["NY", "LA"] * 20}), ["city"])
    findings = check_rows(profile, pd.DataFrame({"city": _text(["NY", " "])}))
    assert [(f["row"], f["issue"]) for f in findings] == [(1, "missing")]
