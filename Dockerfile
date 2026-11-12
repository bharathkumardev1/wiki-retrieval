# syntax=docker/dockerfile:1
# Build with a different registry if Docker Hub rate-limits you:
#   docker build --build-arg PYTHON_IMAGE=mirror.gcr.io/library/python:3.13-slim -t wikiret .
# Behind a TLS-intercepting proxy, pass its CA and proxy settings instead of disabling verification:
#   docker build --network host --build-arg HTTPS_PROXY=$HTTPS_PROXY --secret id=ca,src=/path/to/ca-bundle.crt -t wikiret .
ARG PYTHON_IMAGE=python:3.13-slim

FROM ${PYTHON_IMAGE} AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HOME=/opt/hf
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /src
# CPU wheels first: otherwise sentence-transformers pulls the CUDA build of torch (several GB) into a CPU-only service.
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi; \
    pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md* ./
COPY src ./src
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi; \
    pip install ".[serve]"
# Fetch the base encoder at build time so the running container needs no network access.
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export SSL_CERT_FILE=/run/secrets/ca REQUESTS_CA_BUNDLE=/run/secrets/ca; fi; \
    python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/msmarco-MiniLM-L6-cos-v5')"

FROM ${PYTHON_IMAGE}
RUN useradd --create-home --uid 10001 app
COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/hf /opt/hf
ENV PATH=/opt/venv/bin:$PATH HF_HOME=/opt/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
    WIKIRET_DATA_DIR=/data WIKIRET_ADAPTER=/artifacts/lora_easy WIKIRET_RERANKER_DIR=/artifacts/reranker_int8 \
    WIKIRET_HOST=0.0.0.0
USER app
EXPOSE 8000
# The corpus, embeddings and model artifacts are mounted, not baked in: they are large and not redistributable.
CMD ["python", "-m", "wikiret.serve"]
