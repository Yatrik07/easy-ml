import os
import shutil
import tempfile

# Point Easy-ML at a throwaway data directory before any easyml module is imported.
_DATA_DIR = tempfile.mkdtemp(prefix="easyml-tests-")
os.environ["EASYML_DATA_DIR"] = _DATA_DIR
os.environ["EASYML_NO_CONSOLE_LOGS"] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(scope="session")
def data_dir():
    return _DATA_DIR


@pytest.fixture(scope="session")
def app():
    from easyml.app_factory import create_app

    application = create_app()
    application.config["TESTING"] = True
    return application


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture(scope="session")
def classification_csv(data_dir):
    """Small mixed-type dataset with missing values, saved inside the uploads folder."""
    rng = np.random.default_rng(0)
    n = 160
    age = rng.integers(18, 70, n)
    income = rng.integers(20_000, 100_000, n).astype(float)
    income[::17] = np.nan
    city = rng.choice(["NY", "LA", "SF", None], n)
    owns_car = rng.choice(["yes", "no"], n)
    rooms = rng.integers(1, 5, n)  # low-cardinality integer column -> treated as categorical
    label = np.where((income > 60_000) & (age > 30), "high", np.where(rooms > 2, "mid", "low"))
    df = pd.DataFrame({"age": age, "income": income, "city": city, "owns_car": owns_car,
                       "rooms": rooms, "label": label})
    path = os.path.join(data_dir, "uploads", "classification.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def regression_csv(data_dir):
    rng = np.random.default_rng(1)
    n = 120
    x1 = rng.normal(size=n)
    x2 = rng.choice(["a", "b", "c"], n)
    y = 3 * x1 + (x2 == "b") * 2 + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    path = os.path.join(data_dir, "uploads", "regression.csv")
    df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def text_csv(data_dir):
    sports = ["the team won the football match", "a great goal in the final game",
              "players trained hard for the league", "the coach praised the striker"]
    tech = ["the new laptop has a fast processor", "software update improves the phone",
            "the computer runs machine learning code", "a faster chip for gaming laptops"]
    rows = [{"text": t, "label": "sports"} for t in sports * 10] + [{"text": t, "label": "tech"} for t in tech * 10]
    path = os.path.join(data_dir, "uploads", "text.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def pytest_sessionfinish(session, exitstatus):
    from easyml.utils.logger import close_all_file_handlers

    close_all_file_handlers()
    shutil.rmtree(_DATA_DIR, ignore_errors=True)
