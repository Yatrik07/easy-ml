"""
Data prep from Python.

    from easyml import prep

    prep.list_recipes()                              # recipes saved in the studio
    recipe = prep.Recipe.load("customers")
    clean = recipe.apply("october.csv")              # pandas DataFrame
    recipe.apply("october.csv", output="clean.csv")

    prep.profile("october.csv")                      # one summary per column

    recipe = prep.Recipe(name="mine", steps=[
        {"op": "drop_columns", "params": {"columns": ["id"]}},
        {"op": "fill_missing", "params": {"strategy": "median"}},
    ])
    recipe.save()

Step names and their settings: ``prep.steps()``.
"""


# recipe.py only imports the standard library at module level, so this stays cheap.
from easyml.dataprep.recipe import Recipe, delete_recipe, list_recipes  # noqa: E402


def profile(data):
    """A summary of every column of a DataFrame or a CSV/Excel file."""
    from easyml.dataprep.profile import profile as _profile

    return _profile(data)


def steps():
    """The available steps: ``[{"op", "label", "category", "help", "fields"}]``."""
    from easyml.dataprep.steps import public_specs

    return public_specs()


def apply(recipe, data, output=None):
    """Apply a recipe (a Recipe, a saved recipe's name or id, or a recipe .json path) to data."""
    from easyml.dataprep.recipe import Recipe

    if not isinstance(recipe, Recipe):
        recipe = Recipe.load(recipe)
    return recipe.apply(data, output=output)


__all__ = ["Recipe", "apply", "profile", "steps", "list_recipes", "delete_recipe"]
