FROM python:3.11-slim

WORKDIR /app
COPY run_md.py /app/run_md.py

CMD ["python", "/app/run_md.py", "--config", "/inputs/config.json", "--output", "/outputs/result.json"]
