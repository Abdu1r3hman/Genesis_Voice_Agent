# Python 3.13 for broad Linux wheel coverage; the app also runs on 3.14 (local dev).
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# deps first so this layer caches unless requirements change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY voice/setup_models.py voice/setup_models.py
RUN python voice/setup_models.py

COPY . .

RUN python -c "from rag.embedder import embed_query; embed_query('warm up the embedder')"

EXPOSE 8000

# GROQ_API_KEY must be provided at runtime (-e GROQ_API_KEY=...).
CMD ["python", "web/server.py"]
