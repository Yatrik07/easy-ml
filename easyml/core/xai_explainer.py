import base64
import io
import os

import numpy as np
import pandas as pd

from easyml.core.artifacts import (
    coerce_feature_frame,
    flatten_predictions,
    load_pipeline,
    load_tabular_model,
    read_json,
)
from easyml.utils.database_operations import get_record_by_id
from easyml.utils.logger import write_log

TREE_MODELS = {
    "RandomForestClassifier", "RandomForestRegressor",
    "XGBClassifier", "XGBRegressor",
    "LGBMClassifier", "LGBMRegressor",
    "CatBoostClassifier", "CatBoostRegressor",
}
LINEAR_MODELS = {"LogisticRegression", "LinearRegression"}
BACKGROUND_SAMPLES = 50
MAX_SHOWN_FEATURES = 12  # the rest is summed up as "other columns"


class XAIExplainer:
    """
    Generates SHAP explanations for a single inference prediction.
    It loads the model and pipeline, transforms the input, and picks a suitable SHAP explainer.
    """
    def __init__(self, run_id, model_name, data):
        self.run_id = int(run_id)
        self.model_name = model_name
        self.data = data
        self.model = None
        self.pipeline = None
        self.task = None
        self.folder_path = None
        self.column_data = {}
        self.feature_names = None

    def load_artifacts(self):
        record = get_record_by_id(self.run_id)
        self.task = record["task"]
        self.folder_path = record["folder_path"]

        self.pipeline = load_pipeline(self.folder_path)
        self.model = load_tabular_model(self.folder_path, self.model_name, self.task)
        self.column_data = read_json(self.folder_path, "column_data.json")

        try:
            if "pca" not in self.pipeline.named_steps:
                raw_names = self.pipeline.named_steps["column_transformer"].get_feature_names_out()
                # Clean up names (e.g. 'num__Age' -> 'Age', 'cat__Sex_male' -> 'Sex_male')
                self.feature_names = [n.split("__")[-1] for n in raw_names]
            else:
                pca = self.pipeline.named_steps["pca"]
                self.feature_names = [f"PCA Component {i+1}" for i in range(pca.n_components_)]
        except (AttributeError, KeyError, ValueError):
            self.feature_names = None

    def _transform(self, df):
        transformed = self.pipeline.transform(coerce_feature_frame(df, self.column_data))
        return transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)

    def _background(self):
        """A small sample of the training data, transformed, as the SHAP reference distribution."""
        from easyml.core.model_cache import cached_load

        data_path = os.path.join(self.folder_path, "data.csv")
        pipeline_stat = os.stat(os.path.join(self.folder_path, "preprocessing_pipeline.pkl"))
        # The sample is drawn with a fixed seed, so it is the same every time for an unchanged run.
        return cached_load("shap_background", data_path, self._load_background,
                           extra=(pipeline_stat.st_size, pipeline_stat.st_mtime_ns), size=1024 * 1024)

    def _load_background(self):
        data_path = os.path.join(self.folder_path, "data.csv")
        config = read_json(self.folder_path, "config_json.json")
        df = pd.read_csv(data_path, nrows=5000)
        drop = [config.get("target_column")] + (config.get("ignore_columns") or [])
        df = df.drop(columns=[c for c in drop if c in df.columns])
        df = df.sample(min(BACKGROUND_SAMPLES, len(df)), random_state=0)
        return self._transform(df)

    def _scale(self):
        """What the SHAP numbers measure for this model: "probability", "log-odds" or "value"."""
        if self.task == "MLRegression":
            return "value"
        if self.model_name in ("RandomForestClassifier",):  # sklearn forests explain their probabilities
            return "probability"
        if self.model_name in TREE_MODELS or self.model_name in LINEAR_MODELS:
            return "log-odds"  # boosters and logistic regression explain their raw score
        return "probability" if hasattr(self.model, "predict_proba") else "value"

    def _contributions(self, values, feature_names):
        """Contribution of each original input column: the parts of a one-hot encoded column added up."""
        from easyml.core.evaluation_charts import _output_columns

        columns = list((self.column_data.get("column_dtypes") or {}).keys())
        owners = None
        if self.pipeline is not None and "pca" not in self.pipeline.named_steps:
            try:
                owners = _output_columns(self.pipeline, columns)
            except Exception:
                owners = None
        totals = {}
        if owners and len(owners) == len(values):
            for owner, value in zip(owners, values):
                name = owner if owner is not None else "Other inputs"
                totals[name] = totals.get(name, 0.0) + float(value)
            grouped = True
        else:
            for name, value in zip(feature_names, values):
                totals[name] = totals.get(name, 0.0) + float(value)
            grouped = False

        def shown(column):
            raw = self.data.get(column) if grouped and isinstance(self.data, dict) else None
            return None if raw is None or (isinstance(raw, str) and not raw.strip()) else raw

        ordered = sorted(totals.items(), key=lambda item: -abs(item[1]))
        top, rest = ordered[:MAX_SHOWN_FEATURES], ordered[MAX_SHOWN_FEATURES:]
        features = [{"name": str(name), "value": shown(name), "missing": grouped and shown(name) is None,
                     "contribution": round(value, 6)} for name, value in top]
        other = {"count": len(rest), "contribution": round(sum(v for _, v in rest), 6)} if rest else None
        return features, other, grouped

    def explain(self):
        """
        SHAP explanation of one prediction: a readable breakdown by input column (``explanation``) and
        SHAP's own waterfall plot as a base64 PNG (``plot_b64``).
        """
        import shap
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        write_log("loading artifacts for XAI", "info", "xai_explainer.py")
        self.load_artifacts()
        transformed_input = self._transform(pd.DataFrame([self.data]))

        write_log("generating SHAP values", "info", "xai_explainer.py")
        if self.model_name in TREE_MODELS:
            explainer = shap.TreeExplainer(self.model)
        elif self.model_name in LINEAR_MODELS:
            explainer = shap.LinearExplainer(self.model, self._background())
        else:
            predict_fn = getattr(self.model, "predict_proba", self.model.predict)
            explainer = shap.KernelExplainer(predict_fn, self._background())
        shap_values = explainer(transformed_input)

        shap_val = shap_values[0]
        classification = self.task != "MLRegression"
        pred_class = int(flatten_predictions(self.model.predict(transformed_input))[0]) if classification else None

        # Multi-class models return one column per class: explain the predicted class.
        if len(shap_val.shape) > 1 and shap_val.shape[1] > 1:
            shap_val = shap_val[:, pred_class]
        elif classification and pred_class == 0:
            # A binary model's single output scores class 1; flip it so the explanation is about the
            # predicted class, like the multi-class case ("pushes towards the prediction").
            shap_val = shap.Explanation(values=-np.asarray(shap_val.values), base_values=-np.asarray(shap_val.base_values),
                                        data=shap_val.data, feature_names=shap_val.feature_names)

        if self.feature_names and len(self.feature_names) == len(shap_val.values):
            shap_val.feature_names = self.feature_names
        else:
            shap_val.feature_names = [f"Feature {i}" for i in range(len(shap_val.values))]

        values = np.asarray(shap_val.values, dtype=float).reshape(-1)
        base = float(np.asarray(shap_val.base_values, dtype=float).reshape(-1)[0])
        features, other, grouped = self._contributions(values, shap_val.feature_names)

        shown = min(len(values), 10)
        buffer = io.BytesIO()
        with plt.rc_context({"font.size": 13}):  # readable when shown full width
            plt.figure(figsize=(11, max(4.5, 0.55 * shown + 2)))
            shap.plots.waterfall(shap_val, max_display=10, show=False)
            plt.tight_layout()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            plt.close('all')
        buffer.seek(0)

        write_log("XAI plot generated", "info", "xai_explainer.py")
        return {
            "status": "success",
            "explanation": {
                "scale": self._scale(),
                "base_value": round(base, 6),
                "output_value": round(base + float(values.sum()), 6),
                "features": features,
                "other": other,
                "grouped": grouped,
            },
            "plot_b64": base64.b64encode(buffer.read()).decode('utf-8'),
        }
