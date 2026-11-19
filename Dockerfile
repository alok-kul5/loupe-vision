# Serving image: ONNX Runtime + FastAPI, no PyTorch (about 400 MB instead of several GB).
FROM python:3.11-slim
WORKDIR /srv
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY defect/__init__.py defect/inference.py ./defect/
COPY serve ./serve
COPY scripts/download_models.py ./scripts/
# Models come from the GitHub release at build time (or mount your own at /srv/models_onnx).
ARG MODEL_RELEASE=v1.0.0
RUN python scripts/download_models.py --release ${MODEL_RELEASE}
RUN useradd --create-home app
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
