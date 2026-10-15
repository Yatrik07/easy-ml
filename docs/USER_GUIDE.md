# Easy-ML User Guide

## Overview

Easy-ML helps you train machine learning models from a web browser.

You do not need to write model-training code to use the main application. You provide data, choose a task, pick models, and let the app run the pipeline for you.

Easy-ML currently supports:

- Tabular classification
- Tabular regression
- NLP text classification
- Image classification

It also lets you:

- inspect saved runs
- run inference on new data
- export trained models
- generate EDA reports for tabular data

## Who This Is For

Easy-ML is useful for:

- analysts who want results without building full training scripts
- students learning machine learning workflows
- researchers testing multiple baseline models quickly
- developers who want a local ML workbench with a browser UI

## Key Concepts

### Run

A run is one saved training session. Easy-ML stores:

- the dataset snapshot used for that run
- the trained model files
- the preprocessing pipeline
- metadata and metrics

### Preprocessing

Preprocessing means cleaning or transforming data before model training.

Examples:

- filling in missing values
- scaling numeric columns
- encoding categorical values
- cleaning text
- turning text into numeric features

### Inference

Inference means making predictions with an already trained model.

## Installation

### Base install

The base package includes the studio, tabular classification and regression with scikit-learn models, and TF-IDF text classification:

```bash
pip install easy-ml-studio
```

### Gradient boosting models

Adds XGBoost, LightGBM and CatBoost to the model list:

```bash
pip install "easy-ml-studio[boost]"
```

### NLP extras

Adds lemmatization (spaCy) and sentence-embedding features. Download the spaCy English model once afterwards:

```bash
pip install "easy-ml-studio[nlp]"
easy-ml download-models
```

### Computer vision support

```bash
pip install "easy-ml-studio[cv]"
```

### EDA and explainability support

```bash
pip install "easy-ml-studio[eda]"
```

### Everything

```bash
pip install "easy-ml-studio[all]"
```

Features that need an extra you have not installed are greyed out in the studio, with the install command shown next to them.

## Start the Application

```bash
easy-ml
```

Your browser opens [http://127.0.0.1:5010](http://127.0.0.1:5010). Use `easy-ml --port 8080` for another port, `easy-ml --no-browser` to skip opening a tab, and `easy-ml --help` for all options.

## Where Your Data Is Stored

Easy-ML keeps runtime files in an application data directory, not inside the package source folder.

Default locations:

- Windows: `%LOCALAPPDATA%\easy-ml`
- macOS: `~/Library/Application Support/easy-ml`
- Linux: `~/.local/share/easy-ml`
- Docker: `/data/easy-ml`

Inside that folder, Easy-ML creates:

- `database/`
- `runs/`
- `logs/`
- `uploads/`
- `eda/`

To choose a custom location, set `EASYML_DATA_DIR` before starting the app.

## Main Workflow

The normal workflow looks like this:

1. Open the Easy-ML studio in your browser.
2. Choose a task type.
3. Upload a dataset or point Easy-ML to a dataset path.
4. Configure the run.
5. Select one or more models.
6. Start training.
7. Review results.
8. Reuse the saved run for inference, export, or later comparison.

## Tabular Classification and Regression

### Typical use cases

- churn prediction
- fraud detection
- sales prediction
- house price prediction

### Supported tabular models

Classification:

- `logistic_regression`
- `random_forest`
- `xg_boost`
- `light_gbm`
- `cat_boost`
- `knn`
- `ada_boost`

Regression:

- `linear_regression`
- `random_forest`
- `xg_boost`
- `light_gbm`
- `cat_boost`
- `knn`
- `ada_boost`

### What you configure

- run name
- dataset path
- target column
- columns to ignore
- missing-value handling
- scaling
- optional feature reduction
- optional imbalance handling for classification
- one or more models
- optional hyperparameter tuning level

### Common tabular workflow

1. Choose `Tabular Classification` or `Tabular Regression`.
2. Upload a `.csv` or `.xlsx` file.
3. Select the target column.
4. Choose the models you want to train.
5. Start training.
6. Open the results or run history view.

### Imbalance handling

If one class is much rarer than another, the dataset is imbalanced.

Easy-ML can try:

- `smote`
- `tomek`
- `smotetomek`

If the data does not have enough minority examples, Easy-ML will skip the balancing method and continue with the original training data.

## NLP Text Classification

### Typical use cases

- sentiment analysis
- support-ticket classification
- spam detection
- topic labeling

### What you need in your dataset

Your dataset must include:

- one text column
- one target or label column

### NLP preprocessing options

- lowercase text
- regex cleaning
- stopword removal
- lemmatization

### Feature engineering modes

#### Vectorizer mode

This converts text into token-based numeric features.

Available vectorizers:

- `count`
- `tfidf`
- `hashing`

#### Embedding mode

This converts text into dense semantic vectors using sentence-transformers.

Default embedding model:

- `all-MiniLM-L6-v2`

### Supported NLP models

- `logistic_regression`
- `naive_bayes`
- `complement_nb`
- `svm`
- `random_forest`
- `mlp`
- `light_gbm` when LightGBM is installed

### Common NLP workflow

1. Choose `NLP Text Classification`.
2. Upload the dataset.
3. Select the text column and target column.
4. Choose preprocessing rules.
5. Choose `vectorizer` or `embedding` mode.
6. Select one or more models.
7. Start training.

## Image Classification

### Dataset structure

For image classification, the dataset must be organized by class folders.

Example:

```text
animals/
  cats/
    cat1.jpg
    cat2.jpg
  dogs/
    dog1.jpg
    dog2.jpg
```

Each subfolder name becomes a class label.

### Supported CV models

- `SimpleCNN`
- `ResNet18`
- `MobileNetV2`
- `CustomCNN`

### What you configure

- dataset folder path
- image size
- batch size
- validation split
- whether to use augmentation
- epochs
- learning rate
- early stopping patience
- learning-rate reduction settings
- model selection

### Custom CNN

If you choose `CustomCNN`, the UI lets you define layers manually.

Supported layer types include:

- `Conv2d`
- `BatchNorm2d`
- `ReLU`
- `LeakyReLU`
- `MaxPool2d`
- `AvgPool2d`
- `Dropout`
- `Flatten`
- `Linear`

Easy-ML also includes a validation endpoint to check whether the architecture can run on a dummy input before training begins.

## Run History

Each training run is saved in the SQLite runs database and in a run folder under `runs/`.

Run history helps you:

- inspect metrics and result charts (open a run's details, then select a model)
- reuse older models for predictions
- **retrain** a run on new data: the retrain button (or **Retrain** in the details) asks for a new file
  with the same columns and trains a new run with exactly the same settings. Columns the original run did
  not use are ignored; missing columns are listed before anything starts. The new run shows
  "Retrained from #N".
- **compare** two runs of the same task: tick both and choose **Compare**, or use **Compare with parent**
  on a retrained run. You see both runs' scores per model and their settings with the differences highlighted.
- **download** a run as a .zip (**Download run (.zip)** in the details) and **import** it on another
  computer with **Import run**. Only import files from people you trust.
- delete runs you no longer need

## Result Charts

Runs trained with Easy-ML 0.6.0 or later save charts for every model. Expand a model in the
leaderboard (Results page or run details) to see them:

- **ROC curve** and **precision-recall** (classification): how well the model separates each class,
  with the area under the curve (AUC) and average precision. 0.5 AUC is a coin toss; 1.0 is perfect.
- **Predicted vs actual** and **errors** (regression): points close to the diagonal are good predictions;
  the error histogram should be centred on zero.
- **Feature importance**: which columns the model relies on most. For PCA runs and models such as
  K-Nearest Neighbors it shows how much the score drops when a column is shuffled.
- **Top words** (text runs): the words that push a text towards each class.

## Inference

Easy-ML supports inference after training.

### Tabular inference

Provide a single record using the fields expected by the saved preprocessing pipeline.

### Batch tabular inference

Upload a CSV file and Easy-ML returns a new CSV with a `Prediction` column.

### NLP inference

Provide a single text string and Easy-ML applies the same saved preprocessing and feature engineering rules before prediction.

### CV inference

Upload or send an image, and Easy-ML returns the predicted class with a confidence score.

## EDA Reports

For tabular data, Easy-ML can generate an HTML exploratory data analysis report using `ydata-profiling`.

This is useful when you want a quick overview of:

- missing values
- distributions
- correlations
- unusual values

You need the `[eda]` extra installed for this feature to work reliably.

## Model Export

**Export as API** (on the prediction page of a tabular run) turns the selected model into a small,
standalone web service. It runs without Easy-ML Studio, on any computer or server with Python or
Docker, and other programs send it a row of data to get a prediction back.

Clicking the button opens a short guide with the steps and a ready-to-copy example request built from
your run's columns (using the values in the form, or a real row of your training data). **Download
.zip** saves the package. It contains:

- the trained model and its preprocessing pipeline
- `main.py`: the web service (FastAPI)
- `requirements.txt` with the exact library versions the model was trained with, and a `Dockerfile`
- `README.md` with the same steps and examples for this model

To run it: unzip, then in that folder

```bash
python -m venv .venv
.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --port 8000
```

Open <http://127.0.0.1:8000/docs> to try it in the browser, or send a `POST` to
`http://127.0.0.1:8000/predict` with `{"features": {"column": value, ...}}`. Classification answers
`{"prediction": ..., "confidence": ...}`; regression answers `{"prediction": ...}`. The predictions are
the same as the studio's.

## Frequently Asked Questions

### Do I have to use the browser UI?

No. Developers can also call the backend routes or use the internal Python modules, but the main product experience is the browser UI.

### Can I use Excel files?

Yes. The upload workflow supports `.xlsx` as well as `.csv`.

### Are my trained models saved after I close the app?

Yes. Runs are stored in the external data directory unless you manually delete them.

### Can I move my storage somewhere else?

Yes. Set the `EASYML_DATA_DIR` environment variable before starting the app.

### Can I use a trained model on another computer?

Yes. Download the run as a .zip from its details in Run history (or `easy-ml export <run id>`), then
use **Import run** in Run history (or `easy-ml import file.zip`) on the other computer. Installing the
same versions of the libraries avoids differences; the import warns when they differ.

### Does Easy-ML support GPUs?

Computer vision uses PyTorch and will use CUDA if the installed PyTorch build supports it and a compatible GPU is available.

## Troubleshooting

### The app does not start

Check:

- Python version is 3.10 or newer
- port `5010` is not already in use (`easy-ml` tells you if it is; pick another with `--port`)
- your installation includes the features you want

### NLP embedding mode or lemmatization is unavailable

Install the NLP extra and the spaCy model:

```bash
pip install "easy-ml-studio[nlp]"
easy-ml download-models
```

### Image classification is unavailable

Install the CV extra:

```bash
pip install "easy-ml-studio[cv]"
```

### EDA or explainability does not work

Install the EDA extra:

```bash
pip install "easy-ml-studio[eda]"
```

### My CSV upload fails

Check:

- the file really is `.csv` or `.xlsx`
- the first row contains column names
- the target column exists

### SMOTE is skipped

This usually means the minority class is too small for the requested resampling method. Easy-ML falls back to the original data instead of failing the entire run.

### A saved run predicts incorrectly because fields do not match

Inference must use the same input columns expected by the saved preprocessing pipeline. In the studio, categorical columns are offered as dropdowns of the values seen during training; fields you leave empty are filled in the same way as missing values were during training.
