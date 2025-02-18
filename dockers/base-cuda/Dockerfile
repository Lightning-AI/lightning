# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=11.7.1


FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

ARG PYTHON_VERSION=3.10
ARG PYTORCH_VERSION=2.1
ARG MAX_ALLOWED_NCCL=2.22.3

SHELL ["/bin/bash", "-c"]
# https://techoverflow.net/2019/05/18/how-to-fix-configuring-tzdata-interactive-input-when-building-docker-images/
ENV \
    DEBIAN_FRONTEND="noninteractive" \
    TZ="Etc/UTC" \
    PATH="$PATH:/root/.local/bin" \
    CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda" \
    MKL_THREADING_LAYER="GNU" \
    # MAKEFLAGS="-j$(nproc)"
    MAKEFLAGS="-j2"

RUN \
    apt-get update && apt-get install -y wget && \
    apt-get update -qq --fix-missing && \
    NCCL_VER=$(dpkg -s libnccl2 | grep '^Version:' | awk -F ' ' '{print $2}' | awk -F '-' '{print $1}' | grep -ve '^\s*$') && \
    CUDA_VERSION_MM=${CUDA_VERSION%.*} && \
    TO_INSTALL_NCCL=$(echo -e "$MAX_ALLOWED_NCCL\n$NCCL_VER" | sort -V  | head -n1)-1+cuda${CUDA_VERSION_MM} && \
    apt-get install -y --no-install-recommends --allow-downgrades --allow-change-held-packages \
        build-essential \
        pkg-config \
        cmake \
        git \
        wget \
        curl \
        unzip \
        ca-certificates \
        software-properties-common \
        libopenmpi-dev \
        openmpi-bin \
        ssh \
        ninja-build \
        libnccl2=$TO_INSTALL_NCCL \
        libnccl-dev=$TO_INSTALL_NCCL && \
    # Install python
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
    && \
    update-alternatives --install /usr/bin/python${PYTHON_VERSION%%.*} python${PYTHON_VERSION%%.*} /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    # Cleaning
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /root/.cache && \
    rm -rf /var/lib/apt/lists/*

COPY requirements/pytorch/ requirements/pytorch/
COPY requirements/_integrations/ requirements/_integrations/

ENV PYTHONPATH="/usr/lib/python${PYTHON_VERSION}/site-packages"

RUN \
    curl https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} && \
    # Disable cache \
    pip config set global.cache-dir false && \
    # Install recent setuptools to obtain pkg_resources \
    pip install setuptools==75.6.0 && \
    # set particular PyTorch version \
    pip install -q wget packaging && \
    python -m wget https://raw.githubusercontent.com/Lightning-AI/utilities/main/scripts/adjust-torch-versions.py  && \
    for fpath in `ls requirements/**/*.txt`; do \
      python ./adjust-torch-versions.py $fpath ${PYTORCH_VERSION}; \
    done && \
    CUDA_VERSION_MM=${CUDA_VERSION%.*} && \
    pip install --no-cache-dir \
      -r requirements/pytorch/base.txt \
      -r requirements/pytorch/extra.txt \
      -r requirements/pytorch/test.txt \
      -r requirements/pytorch/strategies.txt \
      --extra-index-url="https://download.pytorch.org/whl/cu${CUDA_VERSION_MM//'.'/''}/" \
      --extra-index-url="https://download.pytorch.org/whl/test/cu${CUDA_VERSION_MM//'.'/''}/"

RUN \
    # Show what we have
    pip --version && \
    pip list && \
    python -c "import sys; ver = sys.version_info ; assert f'{ver.major}.{ver.minor}' == '$PYTHON_VERSION', ver" && \
    python -c "import torch; assert torch.__version__.startswith('$PYTORCH_VERSION'), torch.__version__" && \
    python requirements/pytorch/check-avail-extras.py && \
    rm -rf requirements/
