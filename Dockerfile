FROM mambaorg/micromamba:latest
USER root
ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV ENV_NAME=inference-api
ENV UV_PROJECT_ENVIRONMENT=/opt/conda/envs/inference-api
COPY inference-api/environment.yml /tmp/env.yml
RUN micromamba create -y -f /tmp/env.yml && \
    micromamba clean --all --yes
WORKDIR /inference-api
COPY inference-api/pyproject.toml inference-api/uv.lock ./
RUN uv sync --locked --inexact && \
    uv cache clean
