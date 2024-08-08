FROM mambaorg/micromamba:latest
USER root
RUN apt update && apt install -y make
COPY inference-api/environment.yml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1
WORKDIR /inference-api
