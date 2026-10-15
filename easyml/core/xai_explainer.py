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

    def explain(self):
        """
        Calculates SHAP values for the given input data point and returns a base64 encoded Waterfall plot.
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

        # Multi-class models return one column per class: explain the predicted class.
        if len(shap_val.shape) > 1 and shap_val.shape[1] > 1:
            pred_class = int(flatten_predictions(self.model.predict(transformed_input))[0])
            shap_val = shap_val[:, pred_class]

        if self.feature_names and len(self.feature_names) == len(shap_val.values):
            shap_val.feature_names = self.feature_names
        else:
            shap_val.feature_names = [f"Feature {i}" for i in range(len(shap_val.values))]

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_val, max_display=10, show=False)
        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        plt.close('all')
        buffer.seek(0)

        write_log("XAI plot generated", "info", "xai_explainer.py")
        return {
            "status": "success",
            "plot_b64": base64.b64encode(buffer.read()).decode('utf-8'),
        }
