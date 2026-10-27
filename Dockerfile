# Stage 1: build the React app.
FROM node:20-slim AS web
WORKDIR /web
COPY web/package*.json ./
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi
COPY web/ ./
RUN npm run build

# Stage 2: the API. Keyword search, filters, visually-similar products
# (precomputed CLIP vectors) and the React app work without torch; free-text
# semantic and photo search need requirements-ml.txt and PVA_CLIP=1.
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PVA_DATA_DIR=/tmp/pva
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ backend/
COPY frontend/ frontend/
COPY data/ data/
COPY --from=web /web/dist web/dist
RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
