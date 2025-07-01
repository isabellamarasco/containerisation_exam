ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim
RUN mkdir -p /experiment
VOLUME "/data"
ENV DATA_DIR=/data
WORKDIR /experiment
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY * /experiment/
ENV OWNER=1000:1000
CMD export OUTPUT_DIR=$DATA_DIR/$(date +%Y-%m-%d-%H-%M-%S)-$(hostname) && \
    mkdir -p $OUTPUT_DIR && \
    python anomaly_detection.py | tee $OUTPUT_DIR/output.log && \
    chown -R $OWNER $DATA_DIR
