# ==========================================
# STAGE 1: base (core studio: tabular + TF-IDF text models)
# ==========================================
FROM python:3.11-slim AS base

WORKDIR /app

# libgomp1 is the OpenMP runtime needed by LightGBM / XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements-core.txt

COPY pyproject.toml README.md LICENSE MANIFEST.in ./
COPY easyml ./easyml
RUN pip install --no-cache-dir --no-deps .

ENV EASYML_DATA_DIR=/data/easy-ml
RUN mkdir -p /data/easy-ml
EXPOSE 5010

# Inside a container the server must listen on all interfaces to be reachable.
CMD ["easy-ml", "--host", "0.0.0.0", "--port", "5010", "--no-browser"]

# ==========================================
# STAGE 2: ml (core + gradient boosting + explainability/EDA)
# ==========================================
FROM base AS ml
RUN pip install --no-cache-dir -r requirements-ml.txt

# ==========================================
# STAGE 3: nlp (core + spaCy + sentence embeddings)
# ==========================================
FROM base AS nlp
RUN pip install --no-cache-dir -r requirements-nlp.txt
RUN python -m spacy download en_core_web_sm
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ==========================================
# STAGE 4: cv (core + PyTorch, CPU wheels)
# ==========================================
FROM base AS cv
RUN pip install --no-cache-dir -r requirements-cv.txt

# ==========================================
# STAGE 5: full (everything)
# ==========================================
FROM base AS full
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
