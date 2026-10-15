# custmatch-server: the library plus the service, one image for every process
# (API, scheduler, one-off jobs; server/README.md is the operator guide). Build from the repository root:
#   docker build -f server/Dockerfile -t custmatch-server .
FROM python:3.14-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY custmatch ./custmatch
COPY server ./server
# optional library extras, e.g. --build-arg CUSTMATCH_EXTRAS=catboost for "matcher_model": "forest_catboost"
ARG CUSTMATCH_EXTRAS=""
RUN pip install --no-cache-dir "pandas>=3" "./${CUSTMATCH_EXTRAS:+[$CUSTMATCH_EXTRAS]}" ./server \
 && useradd --system --uid 10001 --home-dir /app custmatch && mkdir -p /data && chown custmatch /data
# third-party name lists, fetched (and checked against their SHA-256) at build time: a container
# needs no network to match, and the non-root user needs no writable home
ENV CUSTMATCH_DATA=/opt/custmatch-data
RUN python -c "from custmatch.country import THIRD_PARTY, data_file; [data_file(n) for n in THIRD_PARTY]" \
 && test -n "$(ls /opt/custmatch-data)"
# unbuffered: the scheduler's and API's logs reach `docker logs` as they happen
ENV CUSTMATCH_DATA_DIR=/data CUSTMATCH_DB_URL=sqlite:////data/custmatch.db PYTHONUNBUFFERED=1
USER custmatch
VOLUME /data
EXPOSE 8000
ENTRYPOINT ["custmatch-server"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
