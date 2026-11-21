FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-api.txt

COPY src ./src
COPY api ./api
COPY app.py train.py download_data.py README.md ./
COPY models ./models
COPY results ./results

RUN mkdir -p /app/data/raw /app/data/processed
EXPOSE 8501 8000

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
