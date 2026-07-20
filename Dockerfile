FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ANALYST_MODE=offline

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# requirements-demo.txt is the slim, product-UI-only install (used for the
# hosted offline demo); requirements.txt (everything, incl. prophet/chromadb/
# sentence-transformers/snowflake/bigquery) stays available for a self-hosted
# full-mode container.
COPY requirements-demo.txt requirements-demo.txt
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements-demo.txt python-multipart uvicorn

COPY . .

EXPOSE 8501 8000

# Render/Railway/most container hosts inject $PORT; Streamlit Cloud manages
# its own port and ignores this Dockerfile entirely (see docs/deployment.md).
CMD ["sh", "-c", "streamlit run app/ui/product_app.py --server.address 0.0.0.0 --server.port ${PORT:-8501}"]
