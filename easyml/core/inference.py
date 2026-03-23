from easyml.utils.database_operations import *
from easyml.config.GLOBAL_VARS import ML_InferenceBody
import joblib
import pandas as pd
import traceback
from easyml.utils.utils import convert_numpy
from numpy import max

class MLInferenceController:
    """
    Controller for performing inference on a trained ML model using saved artifacts.

    Attributes:
        inference_body (ML_InferenceBody): Holds model config, input, and outputs.
    """

    def __init__(self, run_id, data, model_name):
        """
        Initializes the inference controller with the required metadata.

        Args:
            run_id (int): Unique ID corresponding to the training run.
            data (dict): Input data for which prediction is to be made.
            model_name (str): Name of the model to be loaded for inference.
        """

        self.inference_body = ML_InferenceBody(
            run_id=run_id,
            data=data,
            model_name=model_name
        )

    def get_inference(self):
        """
        Loads the model, pipeline, and target encodings, then performs prediction.

        Returns:
            dict: Contains the predicted value and associated probability (if available).

        Raises:
            Exception: Logs and re-raises errors encountered during inference.
        """

        write_log("entered get_inference", "info", os.path.basename(__file__))

        try:
            self.inference_body.model, self.inference_body.pipeline, self.inference_body.target_encodings = self.load_objects(self.inference_body)
            write_log("Loaded model and pipeline objects successfully", "info", os.path.basename(__file__))

            prediction = self.predict_single_point()
            write_log("Prediction for singple point completed successfully", "info", os.path.basename(__file__))

            return prediction

        except Exception as e:
            if self.inference_body.status_code == 200:
                self.inference_body.error_message = "InferenceError"
                self.inference_body.status_code = error_code_mappings[self.inference_body.error_message]
                self.inference_body.error_traceback = traceback.format_exc()

                write_log("error occurred- {} - {}".format(
                    self.inference_body.error_message, self.inference_body.error_traceback),
                    "error", os.path.basename(__file__))

            raise e


    def predict_single_point(self):
        """
        Applies the pipeline and model to predict a single input record.

        Returns:
            dict: Dictionary containing prediction and probability (if available).
        """

        write_log("entered predict_single_point", "info", os.path.basename(__file__))

        try:
            input_df = pd.DataFrame([self.inference_body.data])

            transformed_input = self.inference_body.pipeline.transform(input_df)
            write_log("applied pipeline on the input", "info", os.path.basename(__file__))

            prediction = self.inference_body.model.predict(transformed_input)
            write_log("applied model on the input", "info", os.path.basename(__file__))

            probabilities = self.inference_body.model.predict_proba(transformed_input) \
                if hasattr(self.inference_body.model, "predict_proba") else None

            if "classification" in self.inference_body.task.lower():
                output = {
                    "prediction": self.inference_body.target_encodings[convert_numpy(prediction[0])],
                    "probabilities": float(max(probabilities[0])) if probabilities is not None else None
                }
            else:
                output = {
                    "prediction": convert_numpy(prediction[0]),
                    "probabilities": None
                }

            write_log(f"prediction complete: {output['prediction']}", "info", os.path.basename(__file__))

            return output

        except Exception as e:
            if self.inference_body.status_code == 200:
                self.inference_body.error_message = "InferenceError"
                self.inference_body.status_code = error_code_mappings[self.inference_body.error_message]
                self.inference_body.error_traceback = traceback.format_exc()

                write_log("error occurred- {} - {}".format(
                    self.inference_body.error_message, self.inference_body.error_traceback),
                    "error", os.path.basename(__file__))

            raise e

    def predict_batch(self, df: pd.DataFrame):
        """
        Applies the pipeline and model to predict multiple records from a DataFrame.
        Returns the original DataFrame with an appended 'Prediction' column.
        """
        write_log("entered predict_batch", "info", os.path.basename(__file__))
        try:
            self.inference_body.model, self.inference_body.pipeline, self.inference_body.target_encodings = self.load_objects(self.inference_body)
            
            output_df = df.copy()
            transformed_input = self.inference_body.pipeline.transform(df)
            predictions = self.inference_body.model.predict(transformed_input)
            
            if "classification" in self.inference_body.task.lower():
                mapped_preds = [self.inference_body.target_encodings.get(convert_numpy(p), convert_numpy(p)) for p in predictions]
                output_df['Prediction'] = mapped_preds
            else:
                output_df['Prediction'] = [convert_numpy(p) for p in predictions]
                
            return output_df

        except Exception as e:
            if self.inference_body.status_code == 200:
                self.inference_body.error_message = "InferenceError"
                self.inference_body.status_code = error_code_mappings[self.inference_body.error_message]
                self.inference_body.error_traceback = traceback.format_exc()
            raise e

    def load_objects(self, inference_body):
        """
        Loads the saved model, preprocessing pipeline, and target encodings from disk.

        Args:
            inference_body (ML_InferenceBody): Contains metadata about the inference task.

        Returns:
            tuple: model, pipeline, and target_encodings (if classification).

        Raises:
            Exception: Logs and re-raises errors if file loading fails.
        """

        write_log("entered load_objects", "info", os.path.basename(__file__))

        try:
            record = get_record_by_id(inference_body.run_id)
            write_log(f"successfully loaded repors from runs table from datavase with id - {inference_body.run_id}", "info", os.path.basename(__file__))

            model_path = os.path.join(record["folder_path"], f"{inference_body.model_name}.pkl")

            if "CatBoost" in inference_body.model_name:
                model_path = model_path + ".cbm"

                if "regression" in record["task"].lower():
                    from catboost import CatBoostRegressor

                    # Initialize and load the model
                    model = CatBoostRegressor()
                    model.load_model(model_path)

                elif "classification" in record["task"].lower():
                    from catboost import CatBoostClassifier

                    # Initialize and load the model
                    model = CatBoostClassifier()
                    model.load_model(model_path)

            else:
                model = joblib.load(model_path)
            write_log("model loaded successfully", "info", os.path.basename(__file__))

            inference_body.task = record["task"]

            target_encodings = None
            if "classification" in inference_body.task.lower():
                with open(os.path.join(record["folder_path"], "config_json.json") , 'r') as file:
                    config_json = json.load(file)

                    target_encodings = config_json["target_encodings"]
                    target_encodings = {v: k for k, v in target_encodings.items()}
                write_log("target encodings loaded successfully", "info", os.path.basename(__file__))


            pipeline_path = os.path.join(record["folder_path"], "preprocessing_pipeline.pkl")
            pipeline = joblib.load(pipeline_path)
            write_log("pipeline object loaded successfully", "info", os.path.basename(__file__))

            return model, pipeline, target_encodings

        except Exception as e:
            if self.inference_body.status_code == 200:
                self.inference_body.error_message = "InferenceError"
                self.inference_body.status_code = error_code_mappings[self.inference_body.error_message]
                self.inference_body.error_traceback = traceback.format_exc()

                write_log("error occurred- {} - {}".format(
                    self.inference_body.error_message, self.inference_body.error_traceback),
                    "error", os.path.basename(__file__))

            raise e


