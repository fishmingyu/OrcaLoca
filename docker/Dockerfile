# Use an official NVIDIA CUDA 12.1 base image
ARG BASEIMAGE=nvcr.io/nvidia/nvhpc:23.5-devel-cuda12.1-ubuntu22.04
FROM ${BASEIMAGE} as base
ARG PLATFORM
ARG OPENAI_API_KEY

ENV BUILD_PLATFORM=${PLATFORM}

# Note: The list of packages you install might differ based on your application's requirements
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY install/entry.sh /install/entry.sh
RUN bash /install/entry.sh

# CUDA Environment variables
ENV PATH=/usr/local/nvidia/bin:${PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/23.5/cuda
ENV CPLUS_INCLUDE_PATH=${CUDA_HOME}:${CPLUS_INCLUDE_PATH}
ENV C_INCLUDE_PATH=${CUDA_HOME}:${C_INCLUDE_PATH}
ENV LIBRARY_PATH=${CUDA_HOME}/lib64:${LIBRARY_PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}:${LD_LIBRARY_PATH}

# OpenAI API Key
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

COPY install/conda_env.sh /install/conda_env.sh
RUN bash /install/conda_env.sh