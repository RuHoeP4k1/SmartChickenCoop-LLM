# ── Stage 1: Build React frontend ──────────────────────────────────────────
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python backend + built frontend ────────────────────────────────
FROM python:3.11-slim
WORKDIR /app

# Install Python deps first (own layer — only rebuilds when requirements.txt changes)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# Drop the React build in from stage 1 (Node.js itself is discarded)
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Ensure uploads directory exists with correct ownership before dropping privileges
RUN mkdir -p /app/uploads/heatmaps

# Non-root user — limits damage if something goes wrong inside the container
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
