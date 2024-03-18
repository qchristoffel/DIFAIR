#!/bin/bash

# This script is used to create a virtual environment using conda.
# Follow https://www.tensorflow.org/install/pip?hl=en instructions to install Tensorflow GPU.

read -p "Enter the name of the virtual environment (default 'tf'): " venv_name

venv_name=${venv_name:-tf}

# Create the virtual environment
# (conda is used to install cudatoolkit)
conda create -n $venv_name python=3.10.* -y

# Activate the virtual environment
source activate $venv_name

# Install pip
conda install pip -y

# GPU setup
conda install -c conda-forge cudatoolkit=11.8.0 -y
pip install nvidia-cudnn-cu11==8.6.0.163
# conda install -c conda-forge cudatoolkit=11.2.* cudnn=8.1.* -y
# conda install -c conda-forge tensorflow=2.11.* -y

# GPU setup
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Install Tensorflow GPU
pip install tensorflow==2.12.*

# Install NVCC
conda install -c nvidia cuda-nvcc==11.3.58 -y

# Configure the XLA cuda directory
printf 'export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/\n' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Copy libdevice file to the required path
mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice
cp $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/

# Install other packages
pip install -r requirements.txt

# Test Tensorflow GPU installation
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"




