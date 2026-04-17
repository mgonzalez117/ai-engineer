FROM python:3.11-slim

WORKDIR /app

# Installer dépendances système de build
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade outils de build Python
RUN pip install --upgrade pip setuptools wheel

# Copier tout le projet dans /app
COPY . .

# Copier requirements et installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 7860

ENTRYPOINT ["/app/entrypoint.sh"]