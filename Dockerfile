FROM python:3.11-slim

# System dependencies required by this repo (PDF/text extraction + optional office/OCR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    ghostscript \
    libreoffice \
    tesseract-ocr tesseract-ocr-deu \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
# Note: keep in sync with docs/DEPENDENCIES.md / your venv installs.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
    fastapi "uvicorn[standard]" \
    flask \
    sqlite-web \
    requests \
    python-dotenv \
    openpyxl

# Copy code (for production images). In development, docker-compose bind-mounts the repo over /app.
COPY . /app

# Supervisor config + entrypoint
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PYTHONUNBUFFERED=1
EXPOSE 8000 8088 8081

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/usr/bin/supervisord", "-n"]
