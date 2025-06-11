ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim
ARG DATA_OWNER=1000:1000
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN addgroup --gid ${GROUP_ID} appgroup && \
    adduser --uid ${USER_ID} --gid ${GROUP_ID} --disabled-password --gecos "" appuser
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
USER appuser
ENTRYPOINT ["python", "anomaly_detection.py"]