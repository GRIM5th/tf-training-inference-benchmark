#!/bin/bash

if [ -f /etc/os-release ]; then
    # freedesktop.org and systemd
    . /etc/os-release
    OS=$NAME
    VER=$VERSION_ID
    STATUS="supported"
else
    # Fall back to uname, e.g. "Linux <version>", also works for BSD, etc.
    OS=$(uname -s)
    VER=$(uname -r)
    STATUS="unsupported"
    echo "Unsupported Distribution"
fi

if [[ "$STATUS" == "supported" ]]; then
  if [[ "$OS" == "Ubuntu" ]]; then
    mkdir environment-setup
    cd environment-setup
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update
    sudo apt upgrade
    sudo apt-get -y install cuda #install cuda toolkit
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh #install miniconda3
    bash Miniconda3-latest-Linux-x86_64.sh
    conda create --name tf python=3.9
    conda activate tf
    conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
    mkdir -p $CONDA_PREFIX/etc/conda/activate.d
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    pip install --upgrade pip
    pip install tensorflow==2.9.0 #if using Nvidia TensorRT, use tf 2.9.0, don't use 2.10 or use 2.10 nightly, there's an issue on TensorRT version request
    python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    #https://www.tensorflow.org/install/source#gpu

  elif [[ $OS == "Rocky" ]]; then
    mkdir environment-setup
    cd environment-setup
    sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
    sudo dnf clean all
    sudo dnf -y module install nvidia-driver:latest-dkms
    sudo dnf -y install cuda
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh #install miniconda3
    bash Miniconda3-latest-Linux-x86_64.sh
    conda create --name tf python=3.9
    conda activate tf
    conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
    mkdir -p $CONDA_PREFIX/etc/conda/activate.d
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    pip install --upgrade pip
    pip install tensorflow==2.9.0 #if using Nvidia TensorRT, use tf 2.9.0, don't use 2.10 or use 2.10 nightly, there's an issue on TensorRT version request
    python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    #https://www.tensorflow.org/install/source#gpu
  else
    echo "Unsupported Distribution"
  fi
else
  echo "Wait until developer work on this"
fi
#echo "Thank you"
