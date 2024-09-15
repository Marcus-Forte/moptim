FROM ubuntu:latest

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    libeigen3-dev \
    build-essential \
    clangd \
    clang-tidy \
    clang-format \
    libgtest-dev \
    gdb \
    libtbb-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* 