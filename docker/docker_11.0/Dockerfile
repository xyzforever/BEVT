FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu16.04

# Labels for the docker
LABEL description="This docker has pytorch 1.7.1 in python37 with cuda 11.0, and cudnn 8.0." \
      tag="pytorch1.7.1-py37-cuda11.0-openmpi-mmcv-apex-timm" \
      creator="ninja" tooltype="pytorch" \
      tooltypeversion="1.7.1" \
      createtime="06/08/2021"

USER root:root

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV DEBIAN_FRONTEND=noninteractive

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

ENV NCCL_DEBUG=INFO

ENV HOROVOD_GPU_ALLREDUCE=NCCL

RUN apt-get update && apt-get install -y --no-install-recommends \
    libmlx4-1 \
    libmlx5-1 \
    librdmacm1 \
    libibverbs1 \
    libmthca1 \
    libdapl2 \
    dapl2-utils openssh-client \
    openssh-server \
    iproute2 && \
    apt-get install -y \
    build-essential \
    bzip2 \
    git \
    wget \
    unzip \
    libjpeg-dev \
    libpng-dev \
    ca-certificates \
    libopenblas-dev \
    libopencv-dev \
    libyaml-dev \
    cpio && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/anaconda.sh && \
    sh ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

ENV PATH=/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN conda install -y python=3.7 numpy pyyaml scipy ipython mkl scikit-learn matplotlib pandas setuptools Cython h5py graphviz

RUN conda clean -ya

RUN conda install -y mkl-include cmake cffi typing cython

RUN conda install -y -c mingfeima mkldnn

ENV OPENMPI_VERSION=3.1.2

RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-${OPENMPI_VERSION}.tar.gz && \
    tar zxf openmpi-${OPENMPI_VERSION}.tar.gz && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

RUN conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

RUN export CUDA_HOME="/usr/local/cuda"

RUN pip install terminaltables

RUN pip install future tensorboard

RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && \
    cd ..

RUN pip install timm==0.3.2
RUN conda config --add channels conda-forge
RUN conda install -y imgaug

RUN pip install mmcv-full==1.3.3 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.1/index.html

RUN pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
RUN apt-get update && apt-get install -y --no-install-recommends net-tools

WORKDIR /root