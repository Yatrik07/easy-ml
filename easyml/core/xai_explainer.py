import os
import json
import joblib
import traceback
import pandas as pd
import numpy as np
import base64
import io
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from easyml.utils.logger import write_log
from easyml.utils.database_operations import get_record_by_id

class XAIExplainer:
    """
    Generates SHAP explanations for a single inference prediction.
    It loads the model and pipeline, transforms the input, and runs TreeExplainer.
    """
    def __init__(self, run_id, model_name, data):
        self.run_id = int(run_id)
        self.model_name = model_name
        self.data = data
        self.model = None
        self.pipeline = None
        self.task = None
        self.feature_names = None

    def load_artifacts(self):
        record = get_record_by_id(self.run_id)
        self.task = record["task"]
        folder_path = record["folder_path"]
        
        # Load pipeline
        self.pipeline = joblib.load(os.path.join(folder_path, "preprocessing_pipeline.pkl"))
        
        # Determine feature names if possible
        try:
            if "pca" not in self.pipeline.named_steps:
                ct = self.pipeline.named_steps["column_transformer"]
                # sklearn >= 1.0 supports get_feature_names_out
                raw_names = ct.get_feature_names_out()
                # Clean up names (e.g. 'num__Age' -> 'Age', 'cat__Sex_male' -> 'Sex_male')
                self.feature_names = [n.split("__")[-1] for n in raw_names]
            else:
                pca = self.pipeline.named_steps["pca"]
                self.feature_names = [f"PCA Component {i+1}" for i in range(pca.n_components_)]
        except:
            self.feature_names = None

        # Load model
        model_path = os.path.join(folder_path, f"{self.model_name}.pkl")
        if "CatBoost" in self.model_name:
            model_path += ".cbm"
            if "classification" in self.task.lower():
                from catboost import CatBoostClassifier
                self.model = CatBoostClassifier()
            else:
                from catboost import CatBoostRegressor
                self.model = CatBoostRegressor()
            self.model.load_model(model_path)
        else:
            self.model = joblib.load(model_path)

    def explain(self):
        """
        Calculates SHAP values for the given input data point and returns a base64 encoded Waterfall plot.
        """
        try:
            write_log("loading artifacts for XAI", "info", "xai_explainer.py")
            self.load_artifacts()
            
            input_df = pd.DataFrame([self.data])
            transformed_input = self.pipeline.transform(input_df)
            
            if hasattr(transformed_input, "toarray"):
                transformed_input = transformed_input.toarray()
                
            write_log("generating SHAP values", "info", "xai_explainer.py")
            
            # Use TreeExplainer for tree models, LinearExplainer for linear, or generic explainer
            is_tree = any(x in self.model_name for x in ["RandomForest", "LGBM", "XGBoost", "DecisionTree", "CatBoost", "ExtraTrees", "GradientBoosting"])
            
            if is_tree:
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer(transformed_input)
            else:
                # KernelExplainer requires background data, we will skip or use LinearExplainer if possible
                try:
                    explainer = shap.LinearExplainer(self.model, transformed_input)
                    shap_values = explainer(transformed_input)
                except:
                    # Generic explainer on the prediction function with the single point as background (just for mechanical execution)
                    # Note: exact SHAP needs a real background. For single point, it's just an approximation.
                    predict_fn = getattr(self.model, "predict_proba", self.model.predict)
                    explainer = shap.KernelExplainer(predict_fn, transformed_input)
                    shap_values = explainer(transformed_input)

            # Extract base values and values for the single instance
            shap_val = shap_values[0]
            
            # For multi-class classification, shap_val.values is 2D
            if len(shap_val.shape) > 1 and shap_val.shape[1] > 1:
                # Get the predicted class index
                if hasattr(self.model, "predict"):
                    pred_class = int(self.model.predict(transformed_input)[0])
                else:
                    pred_class = 1
                shap_val = shap_val[:, pred_class]
            
            if self.feature_names and len(self.feature_names) == len(shap_val.values):
                shap_val.feature_names = self.feature_names
            else:
                shap_val.feature_names = [f"Feature {i}" for i in range(len(shap_val.values))]
            
            # Plot
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap_val, max_display=10, show=False)
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            img_str = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close('all')
            
            write_log("XAI plot generated", "info", "xai_explainer.py")
            return {
                "status": "success",
                "plot_b64": img_str
            }
            
        except Exception as e:
            trace = traceback.format_exc()
            write_log(f"XAI Error: {str(e)}\n{trace}", "error", "xai_explainer.py")
            return {
                "status": "error",
                "message": str(e)
            }
