# Multi-stage build: produce the frontend static bundle, then bake it
# into the FastAPI image so the trial-app serves SPA + API from one
# origin (and Caddy forwards everything to a single backend).

# ── Stage 1 — build the frontend ─────────────────────────────────────────────
FROM node:22-alpine AS frontend
WORKDIR /fe

# Same-origin in production: the SPA fetches /api/* and ws://.../ws/* on its
# own host. Build-time env baked in via VITE_API_URL='' for relative URLs.
ARG VITE_API_URL=""
ENV VITE_API_URL=${VITE_API_URL}

COPY frontend/package.json frontend/pnpm-lock.yaml* frontend/package-lock.json* ./
RUN if [ -f pnpm-lock.yaml ]; then \
      npm install -g pnpm && pnpm install --frozen-lockfile; \
    elif [ -f package-lock.json ]; then \
      npm ci; \
    else \
      npm install; \
    fi

COPY frontend/ .
RUN if [ -f pnpm-lock.yaml ]; then pnpm build; else npm run build; fi

# ── Stage 2 — backend + frontend bundle ─────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./
COPY --from=frontend /fe/dist /app/frontend_dist

ENV PYTHONUNBUFFERED=1 \
    FRONTEND_DIST=/app/frontend_dist \
    DATA_DIR=/app/data

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
