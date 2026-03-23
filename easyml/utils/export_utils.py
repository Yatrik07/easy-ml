import os
import shutil
import tempfile
import zipfile
from easyml.utils.database_operations import get_record_by_id
from easyml.utils.logger import write_log

def create_deployment_package(run_id: int, model_name: str) -> str:
    """
    Creates a ZIP file containing a ready-to-deploy FastAPI application for the specified model.
    Returns the absolute path to the generated ZIP file.
    """
    write_log(f"Creating deployment package for run {run_id}, model {model_name}", "info", "export_utils.py")
    record = get_record_by_id(run_id)
    folder_path = record["folder_path"]
    task = record["task"]
    
    # Identify model file
    model_file_name = f"{model_name}.pkl"
    if "CatBoost" in model_name:
        model_file_name = f"{model_name}.cbm"
        
    model_source = os.path.join(folder_path, model_file_name)
    pipeline_source = os.path.join(folder_path, "preprocessing_pipeline.pkl")
    config_source = os.path.join(folder_path, "config_json.json")
    
    if not os.path.exists(model_source) or not os.path.exists(pipeline_source):
        raise FileNotFoundError(f"Model or pipeline artifacts not found in {folder_path}")

    # Create a temporary directory to build the package
    temp_dir = tempfile.mkdtemp(prefix="easyml_export_")
    
    # 1. Copy artifacts
    shutil.copy(model_source, os.path.join(temp_dir, "model" + os.path.splitext(model_file_name)[1]))
    shutil.copy(pipeline_source, os.path.join(temp_dir, "pipeline.pkl"))
    if os.path.exists(config_source):
        shutil.copy(config_source, os.path.join(temp_dir, "config.json"))

    # 2. Generate FastAPI main.py
    main_py_content = generate_fastapi_app(task, model_name, "model" + os.path.splitext(model_file_name)[1])
    with open(os.path.join(temp_dir, "main.py"), "w") as f:
        f.write(main_py_content)
        
    # 3. Generate requirements.txt
    reqs_content = generate_requirements(task, model_name)
    with open(os.path.join(temp_dir, "requirements.txt"), "w") as f:
        f.write(reqs_content)
        
    # 4. Generate Dockerfile
    docker_content = generate_dockerfile()
    with open(os.path.join(temp_dir, "Dockerfile"), "w") as f:
        f.write(docker_content)
        
    # 5. Generate README.md
    readme_content = generate_readme(task, model_name)
    with open(os.path.join(temp_dir, "README.md"), "w") as f:
        f.write(readme_content)

    # Zip it up
    zip_path = os.path.join(tempfile.gettempdir(), f"deploy_{model_name}_{run_id}.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, temp_dir)
                zipf.write(file_path, arcname)

    # Clean up temp dir
    shutil.rmtree(temp_dir)
    
    write_log(f"Deployment package created at {zip_path}", "info", "export_utils.py")
    return zip_path

def generate_fastapi_app(task: str, model_name: str, model_filename: str) -> str:
    is_nlp = "nlp" in task.lower()
    is_classification = "classification" in task.lower()
    is_catboost = "catboost" in model_name.lower()
    
    imports = ["from fastapi import FastAPI, HTTPException", "from pydantic import BaseModel", "import joblib", "import pandas as pd", "import json", "import os", "from typing import Dict, Any"]
    if is_catboost:
        if is_classification:
            imports.append("from catboost import CatBoostClassifier")
        else:
            imports.append("from catboost import CatBoostRegressor")

    code = "\n".join(imports) + "\n\n"
    code += "app = FastAPI(title='Easy-ML Exported Model API', version='1.0.0')\n\n"
    
    # Input schema
    if is_nlp:
        code += "class InferenceInput(BaseModel):\n    text: str\n\n"
    else:
        code += "class InferenceInput(BaseModel):\n    features: Dict[str, Any]\n\n"
        
    # Load function
    code += "@app.on_event('startup')\n"
    code += "def load_artifacts():\n"
    code += "    global model, pipeline, target_encodings\n"
    code += "    try:\n"
    code += "        pipeline = joblib.load('pipeline.pkl')\n"
    if is_catboost:
        if is_classification:
            code += f"        model = CatBoostClassifier()\n"
        else:
            code += f"        model = CatBoostRegressor()\n"
        code += f"        model.load_model('{model_filename}')\n"
    else:
        code += f"        model = joblib.load('{model_filename}')\n"
        
    if is_classification:
        code += "        if os.path.exists('config.json'):\n"
        code += "            with open('config.json', 'r') as f:\n"
        code += "                conf = json.load(f)\n"
        code += "                target_encodings = conf.get('target_encodings', {})\n"
        code += "                target_encodings = {v: k for k, v in target_encodings.items()}\n"
        code += "        else:\n"
        code += "            target_encodings = None\n"
    else:
        code += "        target_encodings = None\n"
        
    code += "    except Exception as e:\n"
    code += "        print(f'Error loading artifacts: {e}')\n"
    code += "        raise e\n\n"

    # Prediction endpoint
    code += "@app.post('/predict')\n"
    code += "def predict(data: InferenceInput):\n"
    code += "    try:\n"
    if is_nlp:
        code += "        input_df = pd.DataFrame([{'text': data.text}])\n"
    else:
        code += "        input_df = pd.DataFrame([data.features])\n"
        
    code += "        transformed = pipeline.transform(input_df)\n"
    code += "        prediction = model.predict(transformed)\n"
    
    if is_classification:
        code += "        prob = None\n"
        code += "        if hasattr(model, 'predict_proba'):\n"
        code += "            prob_arr = model.predict_proba(transformed)\n"
        code += "            prob = float(max(prob_arr[0]))\n"
        code += "        pred_val = int(prediction[0]) if hasattr(prediction[0], 'item') else prediction[0]\n"
        code += "        if target_encodings and pred_val in target_encodings:\n"
        code += "            pred_val = target_encodings[pred_val]\n"
        code += "        return {'prediction': pred_val, 'confidence': prob}\n"
    else:
        code += "        pred_val = float(prediction[0]) if hasattr(prediction[0], 'item') else prediction[0]\n"
        code += "        return {'prediction': pred_val}\n"
        
    code += "    except Exception as e:\n"
    code += "        raise HTTPException(status_code=500, detail=str(e))\n\n"
    
    code += "@app.get('/health')\n"
    code += "def health():\n"
    code += "    return {'status': 'ok'}\n"
    return code
    
def generate_requirements(task: str, model_name: str) -> str:
    reqs = ["fastapi", "uvicorn", "pydantic", "scikit-learn==1.6.1", "pandas==2.2.3", "joblib==1.4.2", "numpy==1.26.4"]
    if "nlp" in task.lower():
        reqs.extend(["sentence-transformers", "nltk", "spacy"])
    if "xgboost" in model_name.lower():
        reqs.append("xgboost==3.0.0")
    if "catboost" in model_name.lower():
        reqs.append("catboost==1.2.7")
    if "lightgbm" in model_name.lower():
        reqs.append("lightgbm==4.6.0")
    return "\n".join(reqs)

def generate_dockerfile() -> str:
    return """FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

def generate_readme(task: str, model_name: str) -> str:
    return f"""# Generated Deployment for {model_name}

This package contains a production-ready FastAPI wrapper for your Easy-ML model.

## Running Locally
1. `pip install -r requirements.txt`
2. `uvicorn main:app --reload`
3. Visit `http://127.0.0.1:8000/docs` to test the API.

## Running via Docker
1. `docker build -t my-model-api .`
2. `docker run -p 8000:8000 my-model-api`
"""
