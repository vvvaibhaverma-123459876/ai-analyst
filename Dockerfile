FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-ci.txt requirements-ci.txt
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements-ci.txt && pip install streamlit plotly python-multipart uvicorn

COPY . .

EXPOSE 8501 8000

CMD ["streamlit", "run", "app/ui/product_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
