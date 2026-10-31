FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY churn/ churn/
COPY models/ models/
# Render injects PORT (default 10000)
CMD gunicorn --bind=0.0.0.0:${PORT:-10000} --timeout 600 -w 1 -k uvicorn.workers.UvicornWorker app:app
