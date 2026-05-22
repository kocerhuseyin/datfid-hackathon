FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# 🟢 Add this block BEFORE pip install:
RUN apt-get update && apt-get install -y libgomp1

RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Create shutdown script
RUN echo '#!/bin/bash\nsleep 300\nkill -15 1' > /app/shutdown.sh && chmod +x /app/shutdown.sh

EXPOSE 7860

# Keep math libraries from oversubscribing CPU inside job worker threads.
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV DATFID_MAX_WORKERS=2
ENV DATFID_MAX_QUEUE_SIZE=16
ENV DATFID_MAX_STORED_JOBS=100
ENV DATFID_DAG_CACHE_ENABLED=true
ENV DATFID_MAX_MATERIALIZED_ARTIFACTS=128

# Start both the API and shutdown script.
# Keep a single worker because fit/forecast state is in-memory per process.
CMD ["/bin/bash", "-c", "/app/shutdown.sh & uvicorn main:app --host 0.0.0.0 --port 7860 --workers 1"]
