FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (build-essential and libgomp1 for LightGBM/XGBoost)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    procps \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy ALL source code first (needed for pip install . to find packages)
COPY . .

# Install Python dependencies with CPU-only PyTorch (saves ~8 GB)
RUN pip install --no-cache-dir ".[all]" eventlet \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Pre-download NLP models so they are available at runtime (avoids timeout/disconnect)
RUN python -m spacy download en_core_web_sm
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Tell Easy-ML to store all runtime data under /data/easy-ml (mounted volume)
ENV EASYML_DATA_DIR=/data/easy-ml

# Create data directory (will be overridden by volume mount)
RUN mkdir -p /data/easy-ml

# Set PYTHONPATH to root so internal imports resolve correctly
ENV PYTHONPATH=/app

# Expose Flask port
EXPOSE 5010

# Command to run the application on 0.0.0.0
CMD ["python", "easyml/app.py"]
