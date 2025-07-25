<h1 align="center"> GravlensX </h1>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2507.15775-b31b1b)](https://arxiv.org/abs/2507.15775)
[![Project](https://img.shields.io/badge/Project-GravLensX-ff6a00)](https://myuansun.github.io/gravlensx/)
</div>

<div align="center">
  <img src="images/GravLensX.gif" alt="GravLensX Demo" width="800" />
</div>

This codebase is the official implementation of **Learning Null Geodesics for Gravitational Lensing Rendering in General Relativity** published in ICCV 2025.

## 📖 Overview

GravLensX replaces expensive iterative geodesic integration with trained neural networks that can predict any point along a light ray in a single forward pass.

The pipeline has three main stages:  
1. **Data Generation**: sample, calculate and save null geodesics under the superposed Kerr metric.  
2. **Training**: Train Physics‑Informed Neural Networks (PINNs) for each near field and the far field.
3. **Neural Rendering**: Render the image using the trained PINNs.

An **Euler‑based** baseline (`euler_render.py`) is also included.

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/NEU-REAL/GravLensX
   ```
2. Our code is implemented in Python 3.10 with PyTorch 2.2.1 and Taichi 1.7.2. You can either set up the environment on your own or use the provided conda environment file as follows:
   ```bash
   conda env create -f environment.yaml
   ```
3. Download the [texture files](https://drive.google.com/file/d/1sXdCWbHxTJQXioAsAwzf9BzIYcIyooqs/view?usp=sharing) and unzip the `texture` folder into the project directory.

## 🛠️ Usage
### 1. Learning‑Based Rendering
#### 1.1 Generate Geodesic Dataset
First, specify your dataset directory in `dataset.yaml` in which the calculated geodesic data will be saved. For example:

```dataset_dir: /home/dataset/BH-Space```

Next, run the data generation script:

```bash
python generate_geodesic_data.py
```

**Key parameters**
- `black_hole_positions` (N, 3): The positions of black holes.
- `black_hole_masses` (N): The masses of black holes.
- `black_hole_spins` (N): The spin angular momenta of black holes.
- `near_field_radius`: The radius of the near field.
- `sample_min_radius`: The minimum radius (w.r.t. the black hole center or the world center) for sampling the ray origins.
- `sample_max_radius`: The maximum radius (w.r.t. the black hole center or the world center) for sampling the ray origins.
- `min_radius`: The radius (w.r.t. the black hole center) for judging whether a ray is ended (into the near field or into the black hole).
- `max_radius`: The radius (w.r.t. black hole center or the world center) for judging whether a ray is ended (into the far field or reaching the sky sphere).
- `data_length`: The number of samples to generate for each near field and far field.
- `near_field_radius`: The radius of the near field.

#### 1.2 Train PINNs
**Near‑Field (one process&GPU per BH)**

Before the training, please align the following parameters in `near_field_train_distributed.py`:

- `--black_hole_radius`: aligned to `near_field_radius` in `generate_geodesic_data.py`.
- `black_hole_positions`: aligned to `black_hole_positions` in `generate_geodesic_data.py`.
- `--data_length`: aligned to `data_length` in `generate_geodesic_data.py`.

Then, run the training script:

```bash
python near_field_train_distributed.py
```

**Far‑Field (One process with one or multiple GPU(s))**

Ensure `--num_bh` is correctly set to the number of black holes in your dataset, and `data_length` is aligned to `data_length` in `generate_geodesic_data.py`.

Then, run the training script:
```bash
python far_field_train_distributed.py
```

#### 1.3 Neural Rendering
Make sure the checkpoints saved from training are correctly loaded. Ensure the following parameters in `neural_render.py` are aligned with your dataset:

- `pos_bh`: aligned to `black_hole_positions` in `generate_geodesic_data.py`.
- `mass_bh`: aligned to `black_hole_masses` in `generate_geodesic_data.py`.

Then run the rendering script:

```bash
python neural_render.py
```

**Key Parameters**

- `image_width` & `image_height`: The size of the output image.
- `look_from`: The camera position.
- `look_at`: The point that the camera is looking at.
- `--min_radius`: $l^{in}$ in our paper.
- `--near_bh_distance`: in-black-hole radius in our paper, the radius for judging whether a ray falls into the black hole.
- `--loose_boundary`: $\epsilon$ in our paper.

### 2. Euler‑Based Rendering
```bash
python euler_render.py
```
The key parameters are similar to those in `neural_render.py`.

## 📜 Acknowledgements
We would like to thank [Taichi](https://taichi-lang.cn/) for their powerful graphics framework and [BlackHoleRayMarching](https://github.com/theAfish/BlackHoleRayMarching) for the open source code. Their work served as both an important foundation and an inspiration for this project.

## ✍️ Citation
If you find this code useful, please consider citing our work:

```bibtex
@inproceedings{sun2025learning,
  title={Learning Null Geodesics for Gravitational Lensing Rendering in General Relativity},
  author={Sun, Mingyuan and Fang, Zheng and Wang, Jiaxu and Zhang, Kunyi and Zhang, Qiang and Xu, Renjing},
  booktitle={2025 IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025},
  organization={IEEE}
}
```

