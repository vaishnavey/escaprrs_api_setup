# Use an official slim Python base
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# install system deps incl. mafft
RUN apt-get update && apt-get install -y --no-install-recommends \
    mafft \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy server code into image
COPY app ./app
WORKDIR /app

EXPOSE 80

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]