# RetinexGS

Official implementation of **RetinexGS: Enhancing 3D Gaussian Splatting for Low-Light Scenes via Retinex-Guided Decomposition**.

## Installation

The code has been developed for Linux with an NVIDIA GPU. We recommend using Ubuntu 20.04/22.04, CUDA 11.x, and Python 3.8.

### 1. Clone the repository

```bash
git clone https://github.com/XLR-man/RetinexGS.git
cd RetinexGS
```

### 2. Create a conda environment

```bash
conda create -n retinexgs python=3.8 -y
conda activate retinexgs
```

### 3. Install PyTorch

Install the PyTorch version matching your CUDA toolkit. For CUDA 11.8:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For other CUDA versions, please follow the official PyTorch installation instructions.

### 4. Install Python dependencies

```bash
pip install numpy opencv-python pillow tqdm matplotlib plyfile tensorboard diptest
pip install ninja
```

Install tiny-cuda-nn with PyTorch bindings:

```bash
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### 5. Install CUDA extensions

```bash
pip install submodules/simple-knn
pip install submodules/gaustudio-diff-gaussian-rasterization
```

If compilation fails, check that your CUDA toolkit, PyTorch CUDA version, and compiler are compatible.

## Data Preparation

RetinexGS follows the COLMAP-style dataset layout used by 3D Gaussian Splatting.

Expected structure:

```text
data/
  scene_name/
    images/
      0001.png
      0002.png
      ...
    sparse/
      0/
        cameras.bin
        images.bin
        points3D.bin
```

## Training and Rendering

Train a scene with:

```bash
python train.py -s ./data/scene_name -m ./output/scene_name --eval
```

Render the trained model with:

```bash
python render.py -m ./output/scene_name
```

The script saves rendered results, reflection, illumination, enhanced color, ground truth views, and depth visualizations under the model output directory.

You can also edit `train.sh` to set dataset/output paths and run training followed by rendering:

```bash
bash train.sh
```

Make sure the paths in `train.sh` match your local dataset location.

## Outputs

A typical output directory contains:

```text
output/
  scene_name/
    cfg_args
    cfg_args_opt
    cfg_args_pipe
    cameras.json
    input.ply
    point_cloud/
      iteration_xxxxx/
        point_cloud.ply
        network.pth
    train/
    test/
```

## Citation

If you find this project useful, please consider citing:

```bibtex
@article{retinexgs,
  title   = {RetinexGS: Enhancing 3D Gaussian Splatting for Low-Light Scenes via Retinex-Guided Decomposition},
  author  = {Xu Wang, Langren Xie, Zhiye Tang, Qiudan Zhang, Longhao Zou, You Yang, Wenhui Wu},
  journal = {IEEE Transactions on Multimedia},
  year    = {2026}
}
```

## Acknowledgements

This codebase builds upon the following excellent projects:

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) by GraphDECO-INRIA.
- [GauStudio](https://github.com/GAP-LAB-CUHK-SZ/gaustudio) and its customized differentiable Gaussian rasterizer.
- [simple-knn](https://gitlab.inria.fr/bkerbl/simple-knn) from the original 3DGS implementation.
- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) for efficient hash-grid encoding.

We thank the authors for releasing their code and tools.