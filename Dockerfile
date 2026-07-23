FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system deps required by OpenCV and Tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    tesseract-ocr \
 && rm -rf /var/lib/apt/lists/*

# Copy entire repo into the image first so files like requirements.txt are
# available regardless of the builder's working-directory settings.
COPY . .

# Ensure `requirements.txt` is present for the pip install step. Some PaaS
# builders may use alternative build contexts — copy the file explicitly and
# fail early with a clear message if it's missing.
COPY requirements.txt requirements.txt

RUN if [ -f requirements.txt ]; then \
            python -m pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt; \
        else \
            echo "ERROR: requirements.txt not found in build context" >&2; exit 1; \
        fi

EXPOSE 8501

# Use the host-provided PORT when available (Render sets $PORT).
CMD ["sh", "-c", "streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0"]
