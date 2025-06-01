from easyml.utils.database_operations import *
from easyml.config.GLOBAL_VARS import ML_InferenceBody
import joblib
import pandas as pd
import traceback
from easyml.utils.utils import convert_numpy
from numpy import max

class MLInferenceController:
    def __init__(self, run_id, data, model_name):
        self.inference_body = ML_InferenceBody(
            run_id=run_id,
            data=data,
            model_name=model_name
        )

    def get_inference(self):

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
                print("in")
                output = {
                "prediction": self.inference_body.target_encodings[convert_numpy(prediction[0])],
                "probabilities": convert_numpy( max(probabilities[0])) if probabilities is not None else None
            }

            else:
                output = {
                    "prediction": convert_numpy(prediction[0]),
                    "probabilities": convert_numpy(max(probabilities[0])) if probabilities is not None else None
                }

            print("final return: ", output, type(output['prediction']), self.inference_body.target_encodings, self.inference_body.task.lower(), sep="\n")

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

    def load_objects(self, inference_body):
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


