FROM python:3.12-slim

WORKDIR /app

# Dependencias del sistema necesarias para lightgbm/statsmodels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar definición del paquete e instalar dependencias primero
# (aprovecha el cache de capas de Docker si el código cambia pero no las deps)
COPY pyproject.toml .
COPY src/ ./src
RUN pip install --no-cache-dir -e .

# Copiar el resto del proyecto
COPY api/ ./api
COPY dashboard/ ./dashboard
COPY data/processed ./data/processed
COPY models_artifacts/ ./models_artifacts

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Puerto interno de la API (no expuesto públicamente)
EXPOSE 8000
# Puerto público del dashboard (Render/Railway inyectan $PORT en runtime)
EXPOSE 8501

ENTRYPOINT ["./entrypoint.sh"]