# 📘 Easy ML Web App – User Manual

Welcome to the **Easy ML Web App**, a beginner-friendly tool that simplifies machine learning workflows. No coding required – just upload your dataset, select your options, and let the app do the rest!

---

## 🚀 Overview

The Easy ML Web App allows users to:
- Upload datasets
- Select ML models (regression/classification tasks)
- Apply preprocessing and scaling
- Perform model training and evaluation
- Download trained models and results

---

## 🧭 Navigation & Layout

| Section         | Purpose                                               |
|----------------|--------------------------------------------------------|
| **Home**        | Introduction and app summary                          |
| **Upload**      | Upload your CSV dataset                               |
| **Train Model** | Choose ML algorithms and configure training settings  |
| **Results**     | View metrics, download model or predictions           |

> The app runs on `Flask` and serves a frontend from `frontend/dist`.

---

## 📂 Input Requirements

### ✅ Dataset Format
- Format: `.csv`
- Must contain a **target column**
- Categorical columns should be clearly named if encoding is required

### 📥 Uploading
- Navigate to **Upload**
- Select and upload your dataset
- Choose your **target column**

---

## ⚙️ Training Workflow

1. **Choose Model Options**
   - Select one or more regression models
   - Enable/disable:
     - Preprocessing
     - Imputation
     - Feature scaling
     - Feature reduction (PCA)
     - Hyperparameter tuning

2. **Specify Columns**
   - Target column (required)
   - Categorical columns (optional)
   - Columns to ignore (optional)

3. **Start Training**
   - Click the **Train** button
   - Wait for training to complete
   - View logs or progress if enabled

---

## 📊 Results & Output

After training:
- View model evaluation metrics (e.g., RMSE, MAE)
- Download:
  - Trained model file
  - Predictions
  - Log files

---

## 🧠 Supported ML Features

- Regression Models (exact list auto-loaded from backend)
- Preprocessing pipeline
- Feature selection
- Hyperparameter tuning (optional)
- Logging and export support

---

## 🛠️ Technical Stack

| Component | Technology Used |
|----------|------------------|
| Backend  | Python (Flask)   |
| Frontend | Served from `frontend/dist` |
| ML Stack | scikit-learn, LightGBM, etc. |
| Logging  | Built-in Python logging and file handler |

---

## 🆘 Help & Troubleshooting

- **No results?** Make sure your CSV has no missing target column values.
- **Model not training?** Check file format and column names.
- **Need support?** Reach out to the app maintainer or raise an issue on the GitHub repository (if applicable).

---

## 📄 Version Info

- App Version: `0.1.0`
- Last Updated: June 2025

---

## 📬 Contact

For feature requests, support, or feedback, contact the developer or submit issues via the app’s official support contact.easyml@gmail.com.

