FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    bison \
    build-essential \
    ccache \
    cmake \
    flex \
    gfortran \
    graphviz \
    libegl1 \
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libglu1-mesa \
    libosmesa6 \
    libsm6 \
    libxcursor1 \
    libxext6 \
    libxrender1 \
    ninja-build \
    openmpi-bin \
    libopenmpi-dev \
    pkg-config \
    xvfb \
 && rm -rf /var/lib/apt/lists/*

ENV LIBGL_ALWAYS_SOFTWARE=1 \
    MESA_LOADER_DRIVER_OVERRIDE=llvmpipe \
    OMPI_MCA_btl_vader_single_copy_mechanism=none \
    PYVISTA_OFF_SCREEN=true \
    PYVISTA_TRAME_SERVER_PROXY_ENABLED=true \
    PYVISTA_TRAME_SERVER_PROXY_PREFIX=/proxy/{port}/
