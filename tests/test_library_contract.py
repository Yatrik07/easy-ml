import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


class LibraryContractTests(unittest.TestCase):
    def _run_python(self, code, extra_env=None):
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)

        with tempfile.TemporaryDirectory() as pycache_dir:
            env["PYTHONPYCACHEPREFIX"] = pycache_dir
            return subprocess.run(
                [sys.executable, "-c", code],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
            )

    def test_import_easyml_with_custom_data_dir(self):
        with tempfile.TemporaryDirectory() as data_dir:
            result = self._run_python(
                "import easyml; print(easyml.MLTrainer.__name__)",
                {"EASYML_DATA_DIR": data_dir},
            )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("MLTrainer", result.stdout)

    def test_ml_request_body_uses_isolated_mutable_defaults(self):
        from easyml.config.GLOBAL_VARS import ML_RequestBody

        first = ML_RequestBody()
        second = ML_RequestBody()

        first.models.append("logistic_regression")
        first.performance["accuracy"] = 0.99
        first.categorical_columns.append("city")

        self.assertEqual(second.models, [])
        self.assertEqual(second.performance, {})
        self.assertEqual(second.categorical_columns, [])

    def test_nlp_request_body_uses_isolated_mutable_defaults(self):
        from easyml.nlp.nlp_config import NLP_RequestBody

        first = NLP_RequestBody()
        second = NLP_RequestBody()

        first.models.append("svm")
        first.performance["accuracy"] = 0.91
        first.ngram_range.append(3)

        self.assertEqual(second.models, [])
        self.assertEqual(second.performance, {})
        self.assertEqual(second.ngram_range, [1, 2])

    def test_ml_model_factories_return_fresh_instances(self):
        from easyml.config.GLOBAL_VARS import get_ml_model_instance

        first = get_ml_model_instance("classification", "random_forest")
        second = get_ml_model_instance("classification", "random_forest")

        first.set_params(n_estimators=321)

        self.assertNotEqual(first.get_params()["n_estimators"], second.get_params()["n_estimators"])


if __name__ == "__main__":
    unittest.main()
