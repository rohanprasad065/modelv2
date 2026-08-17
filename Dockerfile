# ---- Base image ----
FROM python:3.11-slim

# xgboost's compiled wheel needs libgomp (OpenMP runtime) present on the
# system, or it fails to import with "libgomp.so.1: cannot open shared
# object file". python:3.11-slim doesn't ship it by default.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---- App setup ----
WORKDIR /app

# Copy only requirements first so Docker can cache this layer -- it only
# re-installs dependencies when requirements.txt actually changes, not
# every time you edit a .py file.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the source (main.py, processing.py, models/, etc.)
COPY . .

# storage.py writes CSVs to ./data/hourly_base at runtime -- make sure
# that directory exists even if it wasn't copied in (e.g. if it's in
# .dockerignore because you mount it as a volume instead).
RUN mkdir -p /app/data/hourly_base

# Document the port the app listens on (informational only -- doesn't
# publish it, you still need -p at `docker run`)
EXPOSE 8000

# Run with uvicorn directly, bound to 0.0.0.0 (not 127.0.0.1) so the
# container's network interface accepts connections from outside the
# container. --reload is dropped -- that's a dev-only flag, it watches
# the filesystem for changes and adds overhead you don't want in prod.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
