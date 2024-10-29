# Instruction for kairos

## Setup

### Setup CUDA

CUDA 12.4 on H100

### Clone repository

```sh
git clone https://github.com/killight98/sarathi-serve.git -b kairos
```

### Create mamba environment

Setup mamba if you don't already have it,

```sh
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
bash Mambaforge-Linux-x86_64.sh # follow the instructions from there
```

Create a Python 3.10 environment,

```sh
mamba create -p ./env python=3.10
```

### Build dependencs

- pytorch

    ```bash
    git clone https://github.com/pytorch/pytorch.git -b v2.4.0
    cd pytorch
    pip -r requirements.txt
    # set envs (NCCL_ROOT, NCCL_LIB_DIR and NCCL_INCLUDE_DIR) before build
    USE_SYSTEM_NCCL=1 CMAKE_CUDA_COMPILER=<path to nvcc>/nvcc python setup.py develop --cmake
    ```

- vllm-flash-attn

    ```bash
    git clone https://github.com/vllm-project/flash-attention.git -b v2.6.2
    cd flash-attention
    python setup.py install
    ```

- flashinfer

    ```bash
    git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
    cd flashinfer
    git apply <proj_path>/docs/0001-disable-pytorch-version-checking.patch
    cd python
    pip install -e .
    ```

### Install Sarathi-Serve

```sh
pip install -r requirements.txt
python setup.py develop
```
