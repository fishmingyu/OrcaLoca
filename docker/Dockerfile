# Use an official NVIDIA CUDA 12.1 base image
ARG BASEIMAGE=ubuntu:22.04
FROM ${BASEIMAGE} as base
ARG PLATFORM

# Use bash for RUN commands
SHELL ["/bin/bash", "-c"]

ENV BUILD_PLATFORM=${PLATFORM}

# Note: The list of packages you install might differ based on your application's requirements
RUN apt-get update && apt-get install -y \
    build-essential \
    curl wget file sudo\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY install/entry.sh /install/entry.sh
RUN bash /install/entry.sh

COPY install/conda_env.sh /install/conda_env.sh
RUN bash /install/conda_env.sh

RUN source ~/.bashrc \
    && conda --version \
    && conda init bash \
    && conda config --append channels conda-forge
