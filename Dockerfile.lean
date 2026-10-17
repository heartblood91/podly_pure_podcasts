# Lean CPU image: acoustic-only processing with groq/remote transcription.
# No torch / openai-whisper, bounded glibc arenas, fixed-time feed refresh.
FROM node:18-alpine AS frontend-build

WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build && test -d dist

FROM python:3.11-slim AS backend

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MALLOC_ARENA_MAX=2 \
    PODLY_REFRESH_CRON_HOURS=9,12,18 \
    PODLY_REFRESH_CRON_MINUTE=0 \
    PODLY_REFRESH_TIMEZONE=Europe/Paris

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates ffmpeg sqlite3 gosu tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/requirements-lean.txt ./requirements-lean.txt
RUN pip install -r requirements-lean.txt && \
    python -c "import importlib.util as u, sys; sys.exit(any(u.find_spec(m) for m in ('torch', 'whisper')))"

COPY src/ ./src/
RUN rm -rf ./src/instance
COPY scripts/ ./scripts/
RUN chmod +x scripts/start_services.sh

COPY --from=frontend-build /app/dist ./src/app/static

RUN groupadd -r appuser && \
    useradd --no-log-init -r -g appuser -d /home/appuser appuser && \
    mkdir -p /home/appuser /app/processing /app/src/instance/data/in \
    /app/src/instance/data/srv /app/src/instance/config /app/src/instance/db \
    /app/src/instance/logs && \
    chown -R appuser:appuser /home/appuser /app

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod 755 /docker-entrypoint.sh

EXPOSE 5001

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["./scripts/start_services.sh"]
