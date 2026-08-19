FROM python:3.11-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# data/ is populated at container start by entrypoint.sh (ingest + index),
# and also mounted as a volume in docker-compose so it persists.
RUN mkdir -p data

EXPOSE 8501

ENTRYPOINT ["./entrypoint.sh"]
