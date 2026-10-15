"""
Recipes: an ordered list of data-prep steps, saved by name and re-applied to new files.

    from easyml import prep

    recipe = prep.Recipe.load("customers")      # saved in the studio's Data prep page
    clean = recipe.apply("october.csv")         # a pandas DataFrame
    recipe.apply("october.csv", output="october_clean.csv")

Statistic-based steps (mean/median fill, outlier limits) are computed on the data the recipe is
applied to, so a recipe adapts to each new file.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime

FORMAT_VERSION = 1


@dataclass
class Recipe:
    name: str = ""
    steps: list = field(default_factory=list)
    id: int = None
    source_columns: list = field(default_factory=list)

    def __post_init__(self):
        from easyml.dataprep.steps import validate_step

        self.steps = [validate_step(step) for step in (self.steps or [])]

    # ------------------------------------------------------------------ JSON

    def to_dict(self):
        return {"format_version": FORMAT_VERSION, "name": self.name, "steps": self.steps,
                "source_columns": list(self.source_columns or [])}

    def to_json(self, indent=2):
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
            raise ValueError("A recipe needs a list of 'steps'.")
        if int(data.get("format_version", 1)) > FORMAT_VERSION:
            raise ValueError("This recipe was made by a newer Easy-ML. Upgrade easy-ml-studio to use it.")
        return cls(name=str(data.get("name") or ""), steps=data["steps"], id=data.get("id"),
                   source_columns=list(data.get("source_columns") or []))

    @classmethod
    def from_json(cls, text):
        try:
            return cls.from_dict(json.loads(text))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Not a valid recipe file: {exc}") from exc

    # ------------------------------------------------------------------ storage

    @classmethod
    def load(cls, name_or_id):
        """A saved recipe, by name, id, or a path to a recipe .json file."""
        import os

        if isinstance(name_or_id, str) and name_or_id.lower().endswith(".json") and os.path.isfile(name_or_id):
            with open(name_or_id, "r", encoding="utf-8") as handle:
                return cls.from_json(handle.read())
        row = get_recipe_row(name_or_id)
        return cls(name=row["name"], steps=json.loads(row["steps_json"] or "[]"), id=row["id"],
                   source_columns=json.loads(row.get("source_columns_json") or "[]"))

    def save(self, overwrite=False):
        """Save under ``self.name``; returns the recipe id."""
        self.id = save_recipe(self.name, self.steps, self.source_columns, recipe_id=self.id, overwrite=overwrite)
        return self.id

    # ------------------------------------------------------------------ use

    def apply(self, data, output=None):
        """Apply to a DataFrame or a CSV/Excel path; returns the cleaned DataFrame (and writes ``output``)."""
        from easyml.dataprep.engine import apply_steps, read_table

        frame = read_table(data) if not hasattr(data, "columns") else data
        result, reports = apply_steps(frame, self.steps)
        failed = [r for r in reports if r.get("error")]
        if failed:
            raise ValueError("Recipe step(s) failed: " + "; ".join(f"{r['label']}: {r['error']}" for r in failed))
        if output:
            result.to_csv(output, index=False)
        return result

    def __repr__(self):
        return f"Recipe(name={self.name!r}, steps={len(self.steps)})"


# ---------------------------------------------------------------------- database

def _now():
    return datetime.now().isoformat(timespec="seconds")


def list_recipes():
    from easyml.utils.database_operations import create_database_table, get_data_from_database

    create_database_table("recipes")
    rows = get_data_from_database("recipes").get("data") or []
    out = []
    for row in sorted(rows, key=lambda r: (r.get("updated_at") or ""), reverse=True):
        steps = json.loads(row.get("steps_json") or "[]")
        out.append({"id": row["id"], "name": row["name"], "steps": len(steps),
                    "created_at": row.get("created_at"), "updated_at": row.get("updated_at")})
    return out


def get_recipe_row(name_or_id):
    from easyml.utils.database_operations import create_database_table, get_data_from_database

    create_database_table("recipes")
    rows = get_data_from_database("recipes").get("data") or []
    if isinstance(name_or_id, int):
        for row in rows:
            if row["id"] == name_or_id:
                return row
        raise LookupError(f"No recipe with id {name_or_id}.")
    for row in rows:
        if str(row["name"]).strip().lower() == str(name_or_id).strip().lower():
            return row
    raise LookupError(f"No recipe named {name_or_id!r}. See the Data prep page or `easy-ml prep list`.")


def save_recipe(name, steps, source_columns=(), recipe_id=None, overwrite=False):
    from easyml.dataprep.steps import validate_step
    from easyml.utils.database_operations import create_database_table, enter_record_to_table, update_record

    name = str(name or "").strip()
    if not name:
        raise ValueError("Give the recipe a name.")
    if len(name) > 80:
        raise ValueError("Recipe names can have at most 80 characters.")
    steps = [validate_step(step) for step in steps]
    create_database_table("recipes")
    try:
        same_name = get_recipe_row(name)
    except LookupError:
        same_name = None
    values = {"name": name, "steps_json": json.dumps(steps),
              "source_columns_json": json.dumps(list(source_columns or [])), "updated_at": _now()}
    if recipe_id is not None:  # updating a known recipe (it may be renamed)
        get_recipe_row(int(recipe_id))
        if same_name and same_name["id"] != int(recipe_id):
            raise ValueError(f"A recipe named {name!r} already exists.")
        update_record("recipes", int(recipe_id), **values)
        return int(recipe_id)
    if same_name:
        if not overwrite:
            raise ValueError(f"A recipe named {name!r} already exists.")
        update_record("recipes", same_name["id"], **values)
        return same_name["id"]
    return enter_record_to_table("recipes", created_at=_now(), **values)


def delete_recipe(recipe_id):
    from easyml.utils.database_operations import delete_record

    get_recipe_row(int(recipe_id))
    delete_record("recipes", int(recipe_id))
