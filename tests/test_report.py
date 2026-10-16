"""0.10.0: one-file HTML run reports (endpoint, CLI, Python), safe to open and send anywhere."""

import os
import re

import pytest

from easyml.trainer import MLTrainer


@pytest.fixture(scope="module")
def clf_run(app, classification_csv):
    client = app.test_client()
    run = MLTrainer(task="MLClassification", name="report-clf", dataset_path=classification_csv, target_column="label",
                    models=["logistic_regression", "random_forest"], cv_folds=3).run()
    client.put(f"/runs/{run.run_id}/meta", json={"run_name": "<script>alert('x')</script> churn",
                                                 "notes": "Use <b>this</b> one.", "tags": ["q4", "<img>"]})
    return run


def _report(client, run_id):
    response = client.get(f"/runs/{run_id}/report")
    assert response.status_code == 200, response.data[:200]
    return response, response.get_data(as_text=True)


def _no_scripts_or_links(html):
    assert not re.search(r"<script", html, re.I)
    assert "http://" not in html and "https://" not in html
    assert " on" not in re.sub(r"<[^>]*>", "", "") and not re.search(r"<[^>]+\son\w+=", html)


def test_classification_report(client, clf_run):
    response, html = _report(client, clf_run.run_id)
    assert response.mimetype == "text/html"
    disposition = response.headers["Content-Disposition"]
    assert disposition.startswith("attachment;") and disposition.endswith('.html"')
    _no_scripts_or_links(html)
    assert "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt; churn" in html  # the name is shown, escaped
    assert "Use &lt;b&gt;this&lt;/b&gt; one." in html and "&lt;img&gt;" in html
    for label in ("Logistic Regression", "Random Forest", "Leaderboard", "Confusion matrix", "Per class",
                  "ROC curve", "Precision-recall", "Feature importance", "Cross-validation"):
        assert label in html, label
    assert html.count("<svg") >= 4
    assert client.get(f"/runs/{clf_run.run_id}/report?view=1").headers["Content-Disposition"].startswith("inline;")


def test_regression_and_text_reports(client, regression_csv, text_csv):
    reg = MLTrainer(task="MLRegression", name="report-reg", dataset_path=regression_csv, target_column="y",
                    models=["linear_regression"]).run()
    _, html = _report(client, reg.run_id)
    _no_scripts_or_links(html)
    for label in ("Predicted vs actual", "Errors", "R²", "MAE"):
        assert label in html, label
    text = MLTrainer(task="NLPTextClassification", name="report-text", dataset_path=text_csv, target_column="label",
                     text_column="text", models=["logistic_regression"]).run()
    _, html = _report(client, text.run_id)
    assert "Words the model relies on" in html and "sports" in html
    assert client.get("/runs/999999/report").status_code == 404


def test_report_cli_and_python(clf_run, tmp_path, capsys):
    import easyml
    from easyml.app import main

    target = tmp_path / "shared.html"
    assert main(["report", str(clf_run.run_id), "-o", str(target)]) == 0
    assert target.exists() and "Wrote" in capsys.readouterr().out
    path = easyml.report(clf_run.run_id, str(tmp_path / "py.html"))
    assert os.path.getsize(path) > 2000
    assert main(["report", "999999"]) == 1
