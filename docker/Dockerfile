# Use an official NVIDIA CUDA 12.1 base image
ARG BASEIMAGE=ubuntu:22.04
FROM ${BASEIMAGE} as base
ARG PLATFORM
ARG OPENAI_API_KEY

RUN useradd -ms /bin/bash user
RUN echo 'user:123456' | chpasswd
RUN usermod -aG sudo user

ENV BUILD_PLATFORM=${PLATFORM}

# Note: The list of packages you install might differ based on your application's requirements
RUN apt-get update && apt-get install -y \
    build-essential \
    curl wget file sudo\
    && rm -rf /var/lib/apt/lists/*

# OpenAI API Key
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

USER user
WORKDIR /home/user

COPY install/entry.sh /install/entry.sh
RUN bash /install/entry.sh

COPY install/conda_env.sh /install/conda_env.sh
RUN bash /install/conda_env.sh

