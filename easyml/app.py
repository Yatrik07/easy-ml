import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request, send_from_directory, send_file, Response, send_from_directory
import logging
from easyml.config.GLOBAL_VARS import *
from easyml.config.paths import DATA_DIR, DB_DIR, RUNS_DIR, LOGS_DIR, UPLOADS_DIR, EDA_DIR, PACKAGE_ROOT
from easyml import MLTrainer
from easyml.utils.data_loader import get_data_info
from datetime import datetime
import warnings, os
from io import StringIO
from easyml.utils.logger import write_log
from easyml.utils.utils import NumpyJSONProvider
from easyml.utils.database_operations import get_data_from_database, delete_run_by_id
from easyml.core.inference import MLInferenceController
from easyml.core.xai_explainer import XAIExplainer
from easyml.utils.export_utils import create_deployment_package

# Optional NLP imports (requires pip install easy-ml[nlp])
try:
    from easyml.nlp.nlp_inference import NLPInferenceController
    from easyml.nlp.nlp_config import AVAILABLE_NLP_MODELS
    _NLP_AVAILABLE = True
except ImportError:
    _NLP_AVAILABLE = False
    AVAILABLE_NLP_MODELS = {}

# Optional CV imports (requires pip install easy-ml[cv])
try:
    from easyml.cv.cv_controllers import CVController
    from easyml.cv.cv_inference import CVInferenceController
    from easyml.cv.cv_config import AVAILABLE_CV_MODELS
    _CV_AVAILABLE = True
except ImportError:
    _CV_AVAILABLE = False
    AVAILABLE_CV_MODELS = []
from easyml.utils.database_operations import get_record_by_id
import json
warnings.filterwarnings('ignore') # , category=UserWarning -> lgbm
os.environ["LIGHTGBM_VERBOSE"] = "0"
from flask_cors import CORS
from easyml.utils.socket_emitter import init_socketio, emit_progress
import shutil



app = Flask(__name__, static_folder="unified_frontend/dist", static_url_path="/")
app.json = NumpyJSONProvider(app)
CORS(app)
socketio = init_socketio(app)

import sys
if "--no-console" in sys.argv:
    os.environ["EASYML_NO_CONSOLE_LOGS"] = "1"

# Directories are already created by paths.py — just set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOGS_DIR, "main_logs.log"), encoding='utf-8')
    ]
)

# Create the runs table at startup so GET /runs never fails with "no such table"
from easyml.utils.database_operations import create_database_table
create_database_table("runs")


@app.route('/load_data', methods=['POST'])
def load_data():
    """
    Loads a dataset either from an uploaded file or from a path in JSON.
    Returns data preview and metadata.

    - If 'file' is present in the request: processes the uploaded CSV file.
    - If 'path' is present in the JSON body: copies and loads the file from that path.

    Returns:
        JSON: Data preview and metadata, or an error response.
    """

    try:
        import uuid
        unique_filename = f"data_{uuid.uuid4().hex[:8]}.csv"
        unique_save_path = os.path.join(UPLOADS_DIR, unique_filename)

        if 'file' in request.files:
            # Handle uploaded CSV file with encoding fallback
            file = request.files['file']
            raw_bytes = file.stream.read()
            data = None
            for enc in ('utf-8', 'utf-8-sig', 'latin-1', 'cp1252'):
                try:
                    stream = StringIO(raw_bytes.decode(enc), newline=None)
                    data = pd.read_csv(stream)
                    break
                except (UnicodeDecodeError, Exception):
                    continue
            if data is None:
                try:
                    stream = StringIO(raw_bytes.decode('utf-8', errors='replace'), newline=None)
                    data = pd.read_csv(stream)
                except Exception as e:
                    return jsonify({"status": "error", "message": f"Could not decode file. Please upload a valid CSV. Error: {str(e)}"}), 400
            data.to_csv(unique_save_path, index=False)

        elif request.is_json and 'path' in request.json:
            source_path = request.json['path']
            with open(source_path, 'rb') as src_file, open(unique_save_path, 'wb') as dst_file:
                shutil.copyfileobj(src_file, dst_file)
            data = pd.read_csv(unique_save_path)

        else:
            return jsonify({"status": "error", "message": "Invalid request: must include either a file upload or a JSON body with 'path'."}), 400

        # Replace NaN with None for JSON-safe serialization
        data = data.where(pd.notna(data), None)
        data_info = get_data_info(data, dataset_path=unique_save_path)
        data_info["saved_path"] = unique_save_path
        return jsonify(data_info)

    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to load dataset: {str(e)}"}), 500

@app.route('/generate_eda', methods=['GET'])
def generate_eda():
    """
    Generates an EDA report for the currently loaded dataset.
    """
    from easyml.core.eda_generator import generate_eda_report
    write_log("generating EDA report", "info", os.path.basename(__file__))
    try:
        dataset_path = request.args.get('path')
        if not dataset_path:
            dataset_path = save_data_path  # Fallback
            
        if not os.path.exists(dataset_path):
            return jsonify({"error": "No dataset currently uploaded or path invalid"}), 400
            
        data = pd.read_csv(dataset_path)
        report_path = generate_eda_report(data, "current_dataset")
        
        return send_file(report_path, mimetype='text/html', as_attachment=False)
    except Exception as e:
        write_log(f"EDA generation failed: {str(e)}", "error", os.path.basename(__file__))
        return jsonify({"error": str(e)}), 500



@app.route('/get_models', methods=['GET'])
def get_models():
    """
    Returns available classification, regression, and NLP model names.

    Returns:
        JSON: Dictionary of model types.
    """

    model_names = { "MLClassification": list(ui2back_mappings_classification_models.keys()),
                    "MLRegression": list(ui2back_mappings_regression_models.keys()),
                    "NLPTextClassification": list(AVAILABLE_NLP_MODELS.keys()),
                    "ImageClassification": AVAILABLE_CV_MODELS}

    return jsonify(model_names)


@app.route('/ml_predict', methods=['POST'])
def ml_predict():
    """
    Endpoint to trigger an ML training run with preprocessing and evaluation.

    Returns:
        JSON: Model performance and metadata or error details.
    """

    print("Entered")
    write_log("new incoming request to /ml_predict", "info", os.path.basename(__file__))
    # Get data from the POST request
    data = request.get_json()
    write_log("received json data", "info", os.path.basename(__file__))

    task = data.get('task')
    name = data.get('name')
    dataset_path = data.get('dataset_path')
    target_column = data.get('target_column')
    
    if not task or not name or not dataset_path or not target_column:
        return jsonify({"status_code": 400, "error_message": "Missing required fields (task, name, dataset_path, or target_column)"}), 400
        
    # preprocessing = data['preprocessing']
    preprocessing = True # assinging as this is being used in some places, but internally this has no affect, we always do preprocessing
    
    imputation_strategy = data.get('imputation_strategy', 'impute')
    imputation = (imputation_strategy != 'drop')
    scaling_raw = data.get('scaling', 'No')
    scaling = scaling_raw == 'Yes' if isinstance(scaling_raw, str) else bool(scaling_raw)
    models = data.get('models', [])
    
    if not models:
        return jsonify({"status_code": 400, "error_message": "Must select at least one model to train"}), 400

    hyper_parameter_tuning = data.get('hyper_parameter_tuning', 0)
    categorical_columns = data.get('categorical_columns', [])
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    path_to_save = os.path.join(RUNS_DIR, f"{name}_{timestamp}")
    os.makedirs(path_to_save, exist_ok=False)
    reduce_features_raw = data.get('reduce_features', 'No')
    reduce_features = reduce_features_raw == 'Yes' if isinstance(reduce_features_raw, str) else bool(reduce_features_raw)
    ignore_columns = data.get('ignore_columns', [])

    write_log("creating a object of MLTrainer", "info", os.path.basename(__file__))
    trainer = MLTrainer(name=name, task=task, dataset_path=dataset_path, target_column=target_column, preprocessing=preprocessing,
                        imputation=imputation, scaling=scaling, models=models, hyper_parameter_tuning=hyper_parameter_tuning,
                        categorical_columns=categorical_columns, path_to_save=path_to_save,
                        reduce_features=reduce_features, ignore_columns= ignore_columns, timestamp=timestamp )

    write_log("starting trainer.run() in background thread", "info", os.path.basename(__file__))

    # Clear any previous cancel signal
    from easyml.cv.cv_model_builder import training_cancel_event
    training_cancel_event.clear()

    def run_ml_training():
        try:
            output = trainer.run()
        except Exception as e:
            write_log(f"ML Training background thread crashed: {str(e)}", "error", os.path.basename(__file__))
            from easyml.utils.socket_emitter import emit_progress
            emit_progress(f"Error: {str(e)}", final=True, result={"status": "error", "error": str(e), "message": "Backend crashed unexpectedly."})

    socketio.start_background_task(run_ml_training)

    return jsonify({"status": "started", "message": "Background ML training started"}), 202


@app.route('/runs', methods=['GET', 'DELETE'])
def manage_runs():
    """
    Handles fetching or deleting training run records.

    Methods:
        GET: Returns all saved run metadata.
        DELETE: Deletes a run and its associated artifacts.

    Returns:
        JSON: Data or status message.
    """

    if request.method == 'GET':
        runs_data = get_data_from_database()
        return runs_data

    elif request.method == 'DELETE':
        data = request.get_json(silent=True) or {}
        run_id = data.get('run_id')

        if run_id is None:
            return jsonify({"error": "Missing run_id"}), 400

        result = delete_run_by_id(run_id)

        if result.get("status_code") == 200:
            return jsonify({"message": f"Run with ID {run_id} deleted successfully."}), 200
        else:
            return jsonify({"error": f"Run with ID {run_id} not found or could not be deleted."}), 404

    # ✅ fallback in case another method sneaks in
    return jsonify({"error": "Unsupported request method"}), 405


@app.route("/get_run_data", methods=['GET'])
def get_run_data():
    """
    Returns metadata and column info for a specific run ID.

    Returns:
        JSON: Run configuration, column data, and list of trained models.
    """

    run_id = request.args.get('run_id')
    if not run_id:
        return jsonify({"error": "Missing run_id parameter"}), 400
    result = get_record_by_id(record_id=run_id)
    if "status_code" not in result:
        result['status_code'] = 200
    status_code = result.pop('status_code')
    if status_code == 200:
        data_path = result['folder_path']
        column_data = []
        target = ''

        # Try column_data.json first (written by tabular model_builder)
        column_data_json_path = os.path.join(data_path, "column_data.json")
        config_json_path = os.path.join(data_path, "config_json.json")

        try:
            if os.path.exists(column_data_json_path):
                with open(column_data_json_path, 'r') as f:
                    col_data = json.load(f)
                # Combine numerical + categorical column names
                num_cols = col_data.get('numerical_columns', [])
                cat_cols = col_data.get('categorical_columns', [])
                column_data = num_cols + cat_cols

            if os.path.exists(config_json_path):
                with open(config_json_path, 'r') as f:
                    config_data = json.load(f)
                target = config_data.get('target_column', '')
                # If column_data still empty, try reading columns key
                if not column_data:
                    column_data = config_data.get('columns', [])
        except Exception as e:
            write_log(f"Error reading config files for run {run_id}: {str(e)}", "error", os.path.basename(__file__))

        # Parse models from metrics_json in DB record
        models = {}
        metrics_raw = result.get('metrics_json', None)
        if metrics_raw:
            if isinstance(metrics_raw, str):
                try:
                    models = json.loads(metrics_raw)
                except:
                    models = {}
            elif isinstance(metrics_raw, dict):
                models = metrics_raw

        # Fallback: scan the run folder for model files
        if not models and data_path and os.path.isdir(data_path):
            for fname in os.listdir(data_path):
                if fname.endswith('.pkl') and fname != 'preprocessing_pipeline.pkl':
                    models[fname.replace('.pkl', '')] = {}
                elif fname.endswith('.cbm'):
                    models[fname.replace('.cbm', '')] = {}
                elif fname.endswith('.pth'):
                    models[fname.replace('.pth', '')] = {}

        models_list = list(models.keys())
        if 'preprocessing_pipeline' in models_list:
            models_list.remove('preprocessing_pipeline')

        output = {
            "columns": column_data,
            "target_column": target,
            "models": models_list
        }
    else:
        output = {"status_code": status_code, "error_message": "Run not found"}
        
    return jsonify(output), status_code


@app.route('/inference', methods=['POST'])
def inference():
    """
    Performs inference on a single data point using a trained model.

    Returns:
        JSON: Prediction and probabilities or error.
    """

    try:
        write_log("new incoming request to /inference", "info", os.path.basename(__file__))
        # Get data from the POST request
        json_data = request.get_json(force=True)
        write_log("received json data", "info", os.path.basename(__file__))

        # Extract top-level fields
        run_id = int(json_data['run_id'])
        model_name = json_data['model_name']
        input_data = json_data['data']

        # Create controller and get inference
        controller = MLInferenceController(run_id=run_id, data=input_data, model_name=model_name)
        predictions = controller.get_inference()

        return predictions

    except Exception as e:
        write_log(f"Error in /inference: {str(e)}", "error", os.path.basename(__file__))
        return jsonify({
            "status_code": 500,
            "error_message": str(e),
        }), 500

@app.route('/inference_batch', methods=['POST'])
def inference_batch():
    """
    Performs bulk inference using an uploaded CSV file.
    Returns the processed CSV with a new 'Prediction' column.
    """
    try:
        write_log("new incoming request to /inference_batch", "info", os.path.basename(__file__))
        
        run_id = int(request.form['run_id'])
        model_name = request.form['model_name']
        file = request.files['file']
        
        if not file:
            return jsonify({"status_code": 400, "error_message": "No file uploaded"}), 400
            
        import pandas as pd
        df = pd.read_csv(file)
        
        controller = MLInferenceController(run_id=run_id, data=None, model_name=model_name)
        output_df = controller.predict_batch(df)
        
        # Save to a temporary buffer and return
        from io import BytesIO
        output_buffer = BytesIO()
        output_df.to_csv(output_buffer, index=False)
        output_buffer.seek(0)
        
        return send_file(output_buffer, mimetype='text/csv', as_attachment=True, download_name=f'predictions_run_{run_id}.csv')
        
    except Exception as e:
        write_log(f"Error in /inference_batch: {str(e)}", "error", os.path.basename(__file__))
        return jsonify({
            "status_code": 500,
            "error_message": str(e),
        }), 500


@app.route('/explain', methods=['POST'])
def explain():
    """
    Endpoint to generate XAI SHAP explanations for a single inference prediction.
    """
    try:
        write_log("new incoming request to /explain", "info", os.path.basename(__file__))
        json_data = request.get_json(force=True)
        
        run_id = int(json_data['run_id'])
        model_name = json_data['model_name']
        input_data = json_data['data']

        explainer = XAIExplainer(run_id=run_id, data=input_data, model_name=model_name)
        result = explainer.explain()
        
        return jsonify(result), 200

    except Exception as e:
        write_log(f"Error in /explain: {str(e)}", "error", os.path.basename(__file__))
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/export', methods=['POST'])
def export_model():
    """
    Endpoint to generate a deployable FastAPI ZIP package for a trained model.
    """
    try:
        write_log("new incoming request to /export", "info", os.path.basename(__file__))
        json_data = request.get_json(force=True)
        
        run_id = int(json_data['run_id'])
        model_name = json_data['model_name']

        zip_path = create_deployment_package(run_id, model_name)
        
        # Send the file to the user
        return send_file(zip_path, as_attachment=True, download_name=f"deploy_{model_name}_{run_id}.zip")

    except Exception as e:
        write_log(f"Error in /export: {str(e)}", "error", os.path.basename(__file__))
        return jsonify({"status": "error", "message": str(e)}), 500

# ========================================= Stop Training Endpoint =========================================

@app.route('/stop_training', methods=['POST'])
def stop_training():
    """
    Signals all running training loops to stop at the next checkpoint.
    """
    from easyml.cv.cv_model_builder import training_cancel_event
    training_cancel_event.set()
    write_log("Training cancellation requested by user", "info", os.path.basename(__file__))
    return jsonify({"status": "ok", "message": "Training stop signal sent. Training will stop at the next checkpoint."}), 200


# ========================================= Computer Vision Endpoints =========================================

@app.route('/cv/list_datasets', methods=['GET'])
def cv_list_datasets():
    """Returns a list of folders inside DATA_DIR that could be CV datasets."""
    try:
        from easyml.config.paths import DATA_DIR
        exempt = ["runs", "database", "logs", "eda"]
        folders = []
        
        # Scan root of DATA_DIR
        if os.path.exists(DATA_DIR):
            for d in os.listdir(DATA_DIR):
                full_path = os.path.join(DATA_DIR, d)
                if os.path.isdir(full_path) and d not in exempt:
                    folders.append({"name": d, "path": full_path})
                    
        # Also scan inside uploads
        uploads_dir = os.path.join(DATA_DIR, "uploads")
        if os.path.exists(uploads_dir):
            for d in os.listdir(uploads_dir):
                full_path = os.path.join(uploads_dir, d)
                if os.path.isdir(full_path):
                    folders.append({"name": f"uploads/{d}", "path": full_path})
                    
        return jsonify({"folders": folders})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/cv/predict', methods=['POST'])
def cv_predict():
    """
    Kicks off a Computer Vision training pipeline for Image Classification.
    """
    try:
        write_log("new incoming request to /cv/predict", "info", os.path.basename(__file__))
        json_data = request.get_json(force=True)
        controller = CVController(kwargs=json_data)
        
        def run_cv_training():
            # Clear any previous cancel signal
            from easyml.cv.cv_model_builder import training_cancel_event
            training_cancel_event.clear()
            try:
                controller.execute_pipeline()
            except Exception as e:
                write_log(f"CV Training thread failed: {str(e)}", "error", os.path.basename(__file__))
                from easyml.utils.socket_emitter import emit_progress
                emit_progress(f"Error: {str(e)}", final=True, result={"status": "error", "error": str(e), "message": "CV Backend crashed unexpectedly."})
                
        socketio.start_background_task(run_cv_training)
        
        return jsonify({"status": "started", "message": "Background CV training started"}), 202

    except Exception as e:
        return {"status_code": 500, "error_message": str(e)}

@app.route('/cv/inference', methods=['POST'])
def cv_inference():
    """
    Performs Computer Vision inference using a trained PyTorch model.
    """
    try:
        write_log("new incoming request to /cv/inference", "info", os.path.basename(__file__))
        json_data = request.get_json(force=True)
        
        run_id = int(json_data['run_id'])
        model_name = json_data['model_name']
        image_base64 = json_data['data']

        controller = CVInferenceController(run_id=run_id, image_base64=image_base64, model_name=model_name)
        predictions = controller.get_inference()

        return jsonify(predictions), 200

    except Exception as e:
        return {
            "status_code": 500,
            "error_message": getattr(getattr(controller, 'inference_body', None), 'error_message', str(e)),
        }


# ---- Custom CNN Validation Endpoint ----
@app.route('/cv/validate_model', methods=['POST'])
def cv_validate_model():
    """Validates a custom CNN architecture by tracing a dummy tensor through it."""
    try:
        data = request.get_json(force=True)
        layers = data.get("layers", [])
        num_classes = data.get("num_classes", 10)
        input_size = tuple(data.get("input_size", [224, 224]))

        from easyml.cv.custom_cnn import validate_architecture
        result = validate_architecture(layers, input_channels=3,
                                       input_size=input_size,
                                       num_classes=num_classes)
        return jsonify(result)
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)}), 400


# ========================================= NLP Endpoints =========================================

@app.route('/nlp/load_data', methods=['POST'])
def nlp_load_data():
    """
    Loads a text dataset for NLP tasks.
    Returns data preview and metadata (same as /load_data).
    """
    try:
        import uuid
        unique_filename = f"data_nlp_{uuid.uuid4().hex[:8]}.csv"
        unique_save_path = os.path.join(UPLOADS_DIR, unique_filename)

        if 'file' in request.files:
            file = request.files['file']
            raw_bytes = file.stream.read()
            data = None
            for enc in ('utf-8', 'utf-8-sig', 'latin-1', 'cp1252'):
                try:
                    stream = StringIO(raw_bytes.decode(enc), newline=None)
                    data = pd.read_csv(stream)
                    break
                except (UnicodeDecodeError, Exception):
                    continue
            if data is None:
                try:
                    stream = StringIO(raw_bytes.decode('utf-8', errors='replace'), newline=None)
                    data = pd.read_csv(stream)
                except Exception as e:
                    return jsonify({"status": "error", "message": f"Could not decode file. Please upload a valid CSV. Error: {str(e)}"}), 400
            data.to_csv(unique_save_path, index=False)
        elif request.is_json and 'path' in request.json:
            source_path = request.json['path']
            with open(source_path, 'rb') as src_file, open(unique_save_path, 'wb') as dst_file:
                shutil.copyfileobj(src_file, dst_file)
            data = pd.read_csv(unique_save_path)
        else:
            return jsonify({"status": "error", "message": "Must include file upload or JSON 'path'."}), 400

        # Replace NaN with None for JSON-safe serialization
        data = data.where(pd.notna(data), None)
        from easyml.utils.data_loader import get_data_info
        data_info = get_data_info(data, dataset_path=unique_save_path)
        data_info["saved_path"] = unique_save_path
        return jsonify(data_info)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to load dataset: {str(e)}"}), 500


@app.route('/nlp/get_models', methods=['GET'])
def nlp_get_models():
    """
    Returns available NLP model names.
    """
    return jsonify({"NLPTextClassification": list(AVAILABLE_NLP_MODELS.keys())})


@app.route('/nlp/predict', methods=['POST'])
def nlp_predict():
    """
    Triggers an NLP training run with preprocessing, feature engineering, and evaluation.
    """
    write_log("new incoming request to /nlp/predict", "info", os.path.basename(__file__))

    data = request.get_json()
    write_log("received NLP json data", "info", os.path.basename(__file__))

    task = data.get('task', 'NLPTextClassification')
    name = data.get('name')
    dataset_path = data.get('dataset_path')
    text_column = data.get('text_column')
    target_column = data.get('target_column')
    
    if not name or not dataset_path or not text_column or not target_column:
        return jsonify({"status_code": 400, "error_message": "Missing required fields (name, dataset_path, text_column, target_column)"}), 400
        
    models = data.get('models', [])
    if not models:
        return jsonify({"status_code": 400, "error_message": "Must select at least one NLP model to train"}), 400
        
    hyper_parameter_tuning = data.get('hyper_parameter_tuning', 0)
    
    imputation_strategy = data.get('imputation_strategy', 'impute')
    imputation = (imputation_strategy != 'drop')

    # Preprocessing options
    lowercase = data.get('lowercase', True)
    remove_stopwords = data.get('remove_stopwords', True)
    lemmatize = data.get('lemmatize', False)
    regex_clean = data.get('regex_clean', True)
    regex_pattern = data.get('regex_pattern', r'[^a-zA-Z0-9\s]')

    # Feature engineering options
    feature_mode = data.get('feature_mode', 'vectorizer')
    vectorizer_type = data.get('vectorizer_type', 'tfidf')
    max_features = data.get('max_features', 10000)
    ngram_range = data.get('ngram_range', [1, 2])
    min_df = data.get('min_df', 2)
    max_df = data.get('max_df', 0.95)
    embedding_model = data.get('embedding_model', 'all-MiniLM-L6-v2')
    embedding_batch_size = data.get('embedding_batch_size', 32)
    embedding_max_length = data.get('embedding_max_length', 256)

    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    path_to_save = os.path.join(RUNS_DIR, f"{name}_{timestamp}")
    os.makedirs(path_to_save, exist_ok=False)

    trainer = MLTrainer(
        name=name, task=task, dataset_path=dataset_path,
        target_column=target_column, text_column=text_column,
        lowercase=lowercase, remove_stopwords=remove_stopwords,
        lemmatize=lemmatize, regex_clean=regex_clean, regex_pattern=regex_pattern,
        feature_mode=feature_mode, vectorizer_type=vectorizer_type,
        max_features=max_features, ngram_range=ngram_range,
        min_df=min_df, max_df=max_df,
        embedding_model=embedding_model, embedding_batch_size=embedding_batch_size,
        embedding_max_length=embedding_max_length,
        models=models, hyper_parameter_tuning=hyper_parameter_tuning,
        imputation=imputation,
        path_to_save=path_to_save, timestamp=timestamp
    )

    # Clear any previous cancel signal
    from easyml.cv.cv_model_builder import training_cancel_event
    training_cancel_event.clear()

    def run_nlp_training():
        try:
            output = trainer.run()
        except Exception as e:
            write_log(f"NLP Training background thread crashed: {str(e)}", "error", os.path.basename(__file__))
            from easyml.utils.socket_emitter import emit_progress
            emit_progress(f"Error: {str(e)}", final=True, result={"status": "error", "error": str(e), "message": "Backend crashed unexpectedly."})

    socketio.start_background_task(run_nlp_training)

    return jsonify({"status": "started", "message": "Background NLP training started"}), 202


@app.route('/nlp/inference', methods=['POST'])
def nlp_inference():
    """
    Performs NLP inference on a text input using a trained model.
    """
    try:
        write_log("new incoming request to /nlp/inference", "info", os.path.basename(__file__))
        json_data = request.get_json(force=True)

        run_id = int(json_data['run_id'])
        model_name = json_data['model_name']
        input_text = json_data['data']

        controller = NLPInferenceController(run_id=run_id, data=input_text, model_name=model_name)
        predictions = controller.get_inference()
        return predictions

    except Exception as e:
        return {
            "status_code": 500,
            "error_message": getattr(getattr(controller, 'inference_body', None), 'error_message', str(e)),
        }


@app.route('/debug/logs', methods=['GET'])
def debug_logs():
    """
    Returns contents of the main log file for debugging.

    Returns:
        Response: Text/plain log content.
    """

    try:
        log_file_path = os.path.join(LOGS_DIR, "main_logs.log")

        if not os.path.exists(log_file_path):
            return Response("Log file not found.", status=404, mimetype='text/plain')

        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()

        return Response(log_content, mimetype='text/plain')

    except Exception as e:
        error_msg = f"Error reading log file: {str(e)}"
        write_log(error_msg, "error", os.path.basename(__file__))
        return Response(error_msg, status=500, mimetype='text/plain')


# ========================================= Settings Endpoints =========================================

@app.route('/settings/info', methods=['GET'])
def settings_info():
    """Returns data directory path and disk usage."""
    import shutil
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(DATA_DIR):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)

    disk = shutil.disk_usage(DATA_DIR)
    return jsonify({
        "data_dir": DATA_DIR,
        "storage_used_mb": round(total_size / (1024 * 1024), 2),
        "disk_free_mb": round(disk.free / (1024 * 1024), 2),
        "version": "0.3.0",
    })


@app.route('/settings/clear-data', methods=['DELETE'])
def settings_clear_data():
    """Deletes all data (database, runs, uploads, logs)."""
    try:
        import shutil
        for d in [DB_DIR, RUNS_DIR, UPLOADS_DIR, LOGS_DIR, EDA_DIR]:
            if os.path.exists(d):
                shutil.rmtree(d)
                os.makedirs(d, exist_ok=True)

        # Re-create the runs table
        from easyml.utils.database_operations import create_database_table
        create_database_table("runs")

        write_log("All data cleared by user request", "info", os.path.basename(__file__))
        return jsonify({"message": "All data cleared successfully.", "data_dir": DATA_DIR}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/settings/device', methods=['GET'])
def settings_device():
    """Returns CUDA availability and current device."""
    cuda_available = False
    cuda_name = ""
    torch_version = "Not installed"
    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_name = torch.cuda.get_device_name(0)
        else:
            if "+cpu" in torch_version:
                cuda_name = "CPU-only PyTorch installed"
            else:
                cuda_name = "CUDA not found by PyTorch"
    except ImportError:
        cuda_name = "PyTorch not installed"
        
    return jsonify({
        "cuda_available": cuda_available,
        "cuda_device_name": cuda_name,
        "torch_version": torch_version,
        "device": "cuda" if cuda_available else "cpu",
    })


@app.route('/settings/credits', methods=['GET'])
def settings_credits():
    """Returns developer credits."""
    return jsonify({
        "project": "Easy-ML",
        "version": "0.3.0",
        "contact_email": "contact.easyml@gmail.com",
        "developers": [
            {"name": "Yatrikkumar Shah", "email": "yatriksshah@gmail.com", "linkedin": "https://www.linkedin.com/in/yatrikkumar-shah/"},
            {"name": "Pawan Kumar J", "email": "pawanjain72002@gmail.com", "linkedin": "https://www.linkedin.com/in/pawankumarj15/"},
            {"name": "Dhruv Jain", "email": "djamitjn@gmail.com", "linkedin": "https://www.linkedin.com/in/dhruv-jain-3b1539212/"},
            {"name": "Esha Mishra", "email": "emish8cool@gmail.com", "linkedin": "https://www.linkedin.com/in/eshamishra28/"},
        ]
    })


@app.route('/exit', methods=['POST'])
def shutdown():
    """
    Shuts down the Flask development server (for local use only).

    Returns:
        JSON: Confirmation message.
    """

    write_log("Forced shutdown initiated via /exit", "info", os.path.basename(__file__))
    threading.Thread(target=lambda: os._exit(0)).start()
    return jsonify({"message": "Force shutdown triggered."}), 200

# ========================================= NLP Frontend =========================================

NLP_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "nlp_frontend", "dist")

@app.route("/nlp/")
def serve_nlp_index():
    """Serves the NLP Studio SPA."""
    return send_from_directory(NLP_FRONTEND_DIR, "index.html")

@app.route("/nlp/static/<path:path>")
def serve_nlp_static(path):
    """Serves NLP static assets (CSS, JS)."""
    return send_from_directory(os.path.join(NLP_FRONTEND_DIR, "static"), path)

@app.route("/nlp/<path:path>")
def serve_nlp_routes(path):
    """Catch-all for NLP SPA client-side routing."""
    file_path = os.path.join(NLP_FRONTEND_DIR, path)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return send_from_directory(NLP_FRONTEND_DIR, path)
    return send_from_directory(NLP_FRONTEND_DIR, "index.html")

# Serve Vue frontend
@app.route("/")
def serve_index():
    """
    Serves the Vue app's index.html.
    """

    print("in /")
    return send_from_directory(app.static_folder, "index.html")


# Catch-all route for Vue.js routing
@app.route("/<path:path>")
def serve_vue_routes(path):
    """
    Serves static assets or falls back to index.html for Vue routing.
    """

    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")


def main():
    """
    Starts the Flask app with SocketIO.
    """

    # from easyml.app import app
    # write_log("🚀 easy-ml server started successfully.", "info", "app.py")
    # app.run(port=5010, threaded=True)


    from easyml.utils.socket_emitter import init_socketio
    socketio = init_socketio(app)
    write_log("🚀 easy-ml server (with WebSocket) started successfully.", "info", "app.py")
    socketio.run(app, port=5010, debug=True)


if __name__ == '__main__':
    from easyml.utils.socket_emitter import init_socketio
    socketio = init_socketio(app)
    write_log("🚀 easy-ml server (with WebSocket) started successfully.", "info", "app.py")
    socketio.run(app, host="0.0.0.0", port=5010, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)