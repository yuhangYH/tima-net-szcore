# syntax=docker/dockerfile:1
# TIMA-Net SzCORE submission
# Based on the official SzCORE template: https://github.com/esl-epfl/szcore/blob/main/config/template.Dockerfile

ARG PYTHON_VERSION=3.10
FROM --platform=linux/amd64 python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg
ENV MNE_DONTWRITE_HOME=true
ENV MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

COPY algo/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir "torch==2.5.1+cpu" --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /app/requirements.txt

# Algorithm package with the five seed checkpoints and the post-processing setting
COPY algo/ /app/

USER appuser

VOLUME ["/data"]
VOLUME ["/output"]

ENV INPUT=""
ENV OUTPUT=""

# SzCORE entrypoint: reads /data/${INPUT}, writes /output/${OUTPUT}
CMD python3 -m tima_szcore "/data/${INPUT}" "/output/${OUTPUT}"
