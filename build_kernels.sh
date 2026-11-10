#!/bin/bash
export PYTHONNOUSERSITE=1
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.9"
export CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE
export MAX_JOBS=24
export CC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++ CUDAHOSTCXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
export PATH=/opt/rh/gcc-toolset-12/root/usr/bin:$PATH LD_LIBRARY_PATH=/opt/rh/gcc-toolset-12/root/usr/lib64:$LD_LIBRARY_PATH
cd ~/climp/work/sdists
PIP=~/climp/venv/bin/pip
$PIP install -v --no-build-isolation ./causal_conv1d-1.5.2.tar.gz && echo CC_OK && \
$PIP install -v --no-build-isolation ./mamba_ssm-2.2.5.tar.gz && echo BUILD_OK
