FROM python:3.11-slim

WORKDIR /srv/laap_v2

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 11546

ENV LAAP_HOST=0.0.0.0 \
    LAAP_PORT=11546 \
    PYTHONUNBUFFERED=1

HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:11546/health/live', timeout=4)"

CMD ["python", "api_server.py"]
