# Easy-ML 🚀

An **easy-to-use, no-code ML training and inference platform** with a modern browser-based UI. Upload your data, pick your models, train, evaluate, and deploy — all without writing a single line of code.

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📊 **Tabular Classification** | Train Random Forest, XGBoost, LightGBM, CatBoost, SVM, and more on CSV data |
| 📈 **Tabular Regression** | Predict continuous values with Linear Regression, Decision Trees, and ensemble models |
| 📝 **NLP Text Classification** | Classify text using TF-IDF or sentence embeddings with pre-trained transformers |
| 🖼️ **Image Classification** | Train SimpleCNN, ResNet18, MobileNetV2, or build your own **Custom CNN** |
| 🧪 **EDA Reports** | One-click exploratory data analysis with interactive HTML reports |
| 🔮 **Inference** | Run predictions on new data with trained models |
| 🧠 **XAI / Explainability** | SHAP-based feature importance plots for model interpretability |
| 📦 **Model Export** | Export trained models as deployable FastAPI packages |
| ⚡ **GPU/CUDA Support** | Auto-detects NVIDIA GPUs for accelerated deep learning training |
| ⚙️ **Settings & Storage** | Centralized data storage with path visibility and one-click data clearing |

## 🏗️ Quick Start

### Option 1: pip install (recommended)

```bash
# Core (tabular ML only — lightweight)
pip install easy-ml

# With NLP support
pip install "easy-ml[nlp]"

# With Computer Vision support
pip install "easy-ml[cv]"

# Everything
pip install "easy-ml[all]"
```

Then run:
```bash
easy-ml
```
Open **http://localhost:5010** in your browser.

### Option 2: Docker

```bash
# Build and run
docker-compose up --build

# Or pull and run
docker build -t easyml .
docker run -p 5010:5010 -v easyml-data:/data/easy-ml easyml
```

### Option 3: From Source

```bash
git clone https://github.com/Yatrik07/easy-ml.git
cd easy-ml
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -e ".[all]"
python easyml/app.py
```

## 📂 Data Storage

Easy-ML stores all runtime data (databases, runs, logs, uploads) in a **centralized, platform-standard directory** — not inside the library installation:

| Platform | Location |
|----------|----------|
| Windows | `%LOCALAPPDATA%\easy-ml\` |
| macOS | `~/Library/Application Support/easy-ml/` |
| Linux | `~/.local/share/easy-ml/` |
| Docker | `/data/easy-ml` (mounted volume) |

Override with: `EASYML_DATA_DIR=/your/custom/path`

View the path and clear all data from the **Settings** page in the UI.

## 🎯 Usage Guide

### For Non-Technical Users

1. **Open the app** at http://localhost:5010
2. **Choose a task** (Classification, Regression, NLP, or Image Classification)
3. **Upload your data** (CSV file or folder path)
4. **Configure** — select your target column, models, and options
5. **Train** — click "Start Training" and watch the progress
6. **View Results** — see accuracy, metrics, and confusion matrices
7. **Run Predictions** — use trained models on new data

### Custom CNN Builder

When selecting Image Classification, choose "CustomCNN" to build your own architecture:
- Add layers (Conv2d, BatchNorm2d, ReLU, MaxPool2d, Dropout, Flatten, Linear)
- Configure parameters for each layer
- Validate the architecture before training

## 🐳 Docker Volumes

Data persists across container rebuilds using a named Docker volume:

```yaml
volumes:
  - easyml-data:/data/easy-ml
```

## 📦 Optional Dependencies

| Extra | Packages | Use Case |
|-------|----------|----------|
| `[nlp]` | spaCy, NLTK, sentence-transformers | Text classification |
| `[cv]` | PyTorch, torchvision, Pillow | Image classification |
| `[eda]` | ydata-profiling, SHAP | EDA reports, XAI |
| `[all]` | All of the above | Full functionality |

## 👥 Developers

- **Yatrikkumar Shah** — [LinkedIn](https://www.linkedin.com/in/yatrikkumar-shah/)
- **Pawan Kumar J** — [LinkedIn](https://www.linkedin.com/in/pawankumarj15/)
- **Dhruv Jain** — [LinkedIn](https://www.linkedin.com/in/dhruv-jain-3b1539212/)
- **Esha Mishra** — [LinkedIn](https://www.linkedin.com/in/eshamishra28/)

📧 **Contact:** contact.easyml@gmail.com

## 📄 License

Apache License 2.0
