# Contributing to Easy-ML Studio

Thanks for helping! Bug reports, ideas and pull requests are all welcome.

- **Found a bug?** Use **Report a bug** on the studio's About page: it opens GitHub's bug form with your
  version, operating system and installed extras already filled in. Please don't attach private data; a few
  made-up rows that show the problem are enough.
- **Have an idea?** Open a [feature idea](https://github.com/Yatrik07/easy-ml/issues/new?template=feature_request.yml).
- **Want to change the code?** Read on.

## Set up a development copy

```bash
git clone https://github.com/Yatrik07/easy-ml.git
cd easy-ml
python -m venv venv
venv\Scripts\activate            # Windows; use `source venv/bin/activate` on macOS/Linux
pip install -e ".[all,dev]"      # or ".[dev]" for the basic install without optional libraries
easy-ml --no-browser             # the studio on http://127.0.0.1:5010
```

Set `EASYML_DATA_DIR` to a scratch folder if you don't want test runs mixed with your own. With
`EASYML_FRONTEND_DEBUG=1` the studio uses Vue's development build, which prints template warnings in the
browser console.

[docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) walks through every module and endpoint. Read the
section for the part you are changing first.

## Run the checks

```bash
pytest                                                        # the whole suite (uses a temporary data folder)
pytest tests/test_training_flows.py::test_regression_flow     # one test
flake8 easyml tests --select=E9,F63,F7,F82,F401,F811,F841     # the lint gate CI runs
python -m build && twine check --strict dist/*                # packaging
```

Tests that need an optional library (boosters, spaCy, PyTorch, SHAP) skip themselves when it is missing.
CI runs the suite on Python 3.10, 3.11 and 3.12, and once more on a basic install. That second run catches
optional libraries leaking into the core, and it is the one that gets pandas 3, so it is worth running
`pytest` in a Python 3.11+ venv with only `.[dev]` too.

## Ground rules

- **Optional libraries stay optional.** The basic install is scikit-learn, pandas and Flask. Import boosters,
  spaCy, sentence-transformers, PyTorch or SHAP only inside the feature that needs them, with
  `capabilities.optional_import(...)`. Never import them at the top of a core module.
- **Same results.** Speed-ups and refactors must keep metrics, predictions and saved files identical.
  Randomized code takes the run's `random_seed`, never a fixed number.
- **pandas 2 and 3 both work.** pandas 3 reads text as the `str` dtype, not `object`: test for text with
  `easyml.utils.frames.is_text(series)`, never `series.dtype == object`.
- **Errors become proper HTTP errors.** Use `route_helpers.error_response(exc)`; never return an error with
  status 200.
- **The frontend has no build step.** It is Vue 3 as plain ES modules in `easyml/unified_frontend/dist`.
  - Templates are template strings, so never write `${` or backticks inside them.
  - Never destructure a store.
  - Use no inline styles or hard-coded colors; use the CSS tokens.
  - Load nothing from a CDN.
- **Docs move with the code.** Update `docs/PYTHON_API.md` when `MLTrainer`, `Predictor` or the CLI change,
  the user guide for visible changes, and add a line under the next version in `CHANGELOG.md`.

## Pull requests

1. Branch from `main` and keep each pull request about one thing.
2. Add or update tests for what you changed.
3. In the description, say what changed and how you tested it. For UI changes, add a screenshot in light and
   dark mode.
4. CI must pass: lint, tests on three Python versions, the basic-install run and the package build.

## Releases

Maintainers bump `__version__` in `easyml/__init__.py`, finish the `CHANGELOG.md` entry and publish a GitHub
release (tag `vX.Y.Z`). The `publish.yml` workflow then uploads the package to PyPI.

## License

By contributing you agree that your work is released under the [Apache License 2.0](LICENSE), like the rest
of the project.
