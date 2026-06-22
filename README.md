# Gaussian Grouping [ECCV'24]

> [**Gaussian Grouping: Segment and Edit Anything in 3D Scenes**](https://arxiv.org/abs/2312.00732)           
> [[Project Page]](https://ymq2017.github.io/gaussian-grouping)           
> ECCV 2024  
> ETH Zurich

We propose Gaussian Grouping, which extends Gaussian Splatting to jointly **reconstruct** and **segment** anything in open-world 3D scenes via **lifting 2D SAM**. It also efficiently supports versatile 3D scene **editing** tasks. Refer to our [paper](https://arxiv.org/abs/2312.00732) for more details.

<img width="1000" alt="image" src='media/teaser_github_demo.gif'>

# This Fork: gsplat Backend + Profiling

This repository is based on the original **Gaussian Grouping** codebase. The main engineering change in this fork is that the differentiable rendering path has been refactored to support a faster **gsplat** backend while keeping the original `diff-gaussian-rasterization` backend available for comparison and debugging.

The backend switch is centralized in `gaussian_renderer.render()`, so training, rendering, evaluation, and downstream editing scripts continue to use the same high-level entry point.

## Render Backend Modes

| Backend | How to enable | Notes |
| --- | --- | --- |
| `gsplat` shared-geometry split raster | default, or `USE_GSPLAT=1 USE_GSPLAT_NATIVE19=0` | Default accelerated backend. It shares projection/tile sorting and rasterizes RGB and 16-D identity features separately. It works with stock gsplat. |
| `gsplat` native19 single pass | `USE_GSPLAT=1 USE_GSPLAT_NATIVE19=1` | Fastest measured backend. It rasterizes RGB + 16-D identity features as a 19-channel payload in one pass, and requires a patched gsplat build with native CDIM=19 support. |
| Original diff rasterizer | `USE_GSPLAT=0` | Original Gaussian Grouping backend, kept as a reference path. |

Example training commands:

```bash
# Default accelerated backend.
CUDA_VISIBLE_DEVICES=0 python train.py \
  -s data/lerf/figurines \
  -r 2 \
  -m output/lerf/figurines \
  --config_file config/gaussian_dataset/train.json \
  --train_split

# Fastest backend, only if the installed gsplat supports native 19-channel rasterization.
CUDA_VISIBLE_DEVICES=0 USE_GSPLAT_NATIVE19=1 python train.py \
  -s data/lerf/figurines \
  -r 2 \
  -m output/lerf/figurines \
  --config_file config/gaussian_dataset/train.json \
  --train_split

# Original backend for debugging or ablations.
CUDA_VISIBLE_DEVICES=0 USE_GSPLAT=0 python train.py \
  -s data/lerf/figurines \
  -r 2 \
  -m output/lerf/figurines \
  --config_file config/gaussian_dataset/train.json \
  --train_split
```

When `USE_GSPLAT_NATIVE19=1` is requested, the code checks the installed gsplat wrapper and raises an error if native 19-channel rasterization is not available. It does not silently fall back to padded 32-channel rasterization.

## Profiling Summary

Clean timing was measured with `script/profile_render_backends.py` using CUDA events only in the timing path, 10 training views per scene, interleaved backend order, 50 warmup rounds, 20 measured rounds per view, and `nvidia-smi pmon` logs to verify that the target GPU was not shared during the run. The table below reports forward+backward training-mode render time from [`profiling_results/THREE_SCENE_SUMMARY.md`](profiling_results/THREE_SCENE_SUMMARY.md).

| Scene | Gaussians | Original diff ms | gsplat split ms | gsplat native19 ms | native19 vs split |
| --- | ---: | ---: | ---: | ---: | ---: |
| ramen | 1,114,069 | 22.415 | 19.675 | 16.722 | 15.0% faster |
| figurines | 280,344 | 12.901 | 7.699 | 6.457 | 16.1% faster |
| teatime | 279,705 | 11.462 | 6.803 | 6.019 | 11.5% faster |

The safe conclusion from these runs is that native19 is consistently the fastest measured backend across the three LERF scenes, while the default split gsplat backend remains the portable accelerated option because it does not require modifying upstream gsplat.

Additional profiling artifacts:

- Clean three-scene timing report: [`profiling_results/THREE_SCENE_SUMMARY.md`](profiling_results/THREE_SCENE_SUMMARY.md)
- PTXAS register/occupancy audit: [`profiling_results/PTXAS_OCCUPANCY.md`](profiling_results/PTXAS_OCCUPANCY.md)
- Timing script: [`script/profile_render_backends.py`](script/profile_render_backends.py)
- PTXAS/occupancy audit script: [`script/occupancy_gsplat_raster.py`](script/occupancy_gsplat_raster.py)

Updates
-----------------
:fire::fire: 2024/01/16: We released the [LERF-Mask dataset](docs/dataset.md) and evaluation code.

2024/01/06: We released the [3D Object Removal & Inpainting](docs/edit_removal_inpaint.md) code.

2023/12/20: We released the [Install Notes](docs/install.md) and [Training & Rendering](docs/train.md) code.


# Introduction
The recent Gaussian Splatting achieves high-quality and real-time novel-view synthesis of the 3D scenes. However, it is solely concentrated on the appearance and geometry modeling, while lacking in fine-grained object-level scene understanding. To address this issue, we propose Gaussian Grouping, which extends Gaussian Splatting to jointly reconstruct and segment anything in open-world 3D scenes. We augment each Gaussian with a compact Identity Encoding, allowing the Gaussians to be grouped according to their object instance or stuff membership in the 3D scene. Instead of resorting to expensive 3D labels, we supervise the Identity Encodings during the differentiable rendering by leveraging the 2D mask predictions by SAM, along with introduced 3D spatial consistency regularization. Comparing to the implicit NeRF representation, we show that the discrete and grouped 3D Gaussians can reconstruct, segment and edit anything in 3D with high visual quality, fine granularity and efficiency. Based on Gaussian Grouping, we further propose a local Gaussian Editing scheme, which shows efficacy in versatile scene editing applications, including 3D object removal, inpainting, colorization and scene recomposition.

<img width="1096" alt="image" src='media/github_method.png'>

# Application Overview
**Local Gaussian Editing scheme**: Grouped Gaussians after training. Each group represents a specific instance / stuff of the 3D scene and can be fully decoupled.
<img width="1096" alt="image" src='media/editing_operation.png'>

## 3D Object Removal
Our Gaussian Grouping can remove the large-scale objects on the Tanks & Temples dataset, from the whole 3D scene with greatly reduced artifacts. Zoom for better view.

https://github.com/lkeab/gaussian-grouping/assets/17427852/f3b0f964-a610-49ab-8332-f2caa64fbf45

## 3D Object Inpainting
Comparison on 3D object inpainting cases, where SPIn-NeRF requires 5h training while our method with better inpainting quality only needs 1 hour training and 20 minutes tuning.

https://github.com/lkeab/gaussian-grouping/assets/17427852/9f5050da-6a50-4a5f-a755-3bdc55eab1bc

https://github.com/lkeab/gaussian-grouping/assets/17427852/3ed0203c-0047-4333-8bf0-0c10f5a078d1

## 3D Object Style Transfer
Comparison on 3D object style transfer cases, Our Gaussian Grouping produces more coherent and natural transfer results across views, with faithfully preserved background.

https://github.com/lkeab/gaussian-grouping/assets/17427852/2f00eab5-590b-4295-bb1c-2076acc63d4a

## 3D Open-world Segmentation
Our Gaussian Grouping approach jointly reconstructs and segments anything in full open-world 3D scenes. The masks predicted by Gaussian Grouping contains much sharp and accurate boundary than LERF.

https://github.com/lkeab/gaussian-grouping/assets/60028943/38241b99-1497-4a7c-bd22-5b018b85548c



## 3D Multi-Object Editing
Our Gaussian Grouping approach jointly reconstructs and segments anything in full open-world 3D scenes. Then we concurrently perform 3D object editing for several objects.

https://github.com/lkeab/gaussian-grouping/assets/17427852/d9638a1c-1569-4c72-91b9-ee68e9e017e5

# Installation
You can refer to the [install document](./docs/install.md) to build the Python environment.

# Training and Masks Rendering
Then refer to the [train document](./docs/train.md) to train your own scene.

# Open-Vocabulary Segmentation
For evaluation on the **LERF-Mask dataset** proposed in our paper, you can refer to the [dataset document](./docs/dataset.md).

# 3D Object Removal and Inpainting
You can select the 3D object for removal and inpainting after training. Details are in the [edit removal inpaint document](./docs/edit_removal_inpaint.md).


Citation
---------------
If you find Gaussian Grouping useful in your research or refer to the provided baseline results, please star :star: this repository and consider citing :pencil::
```
@inproceedings{gaussian_grouping,
    title={Gaussian Grouping: Segment and Edit Anything in 3D Scenes},
    author={Ye, Mingqiao and Danelljan, Martin and Yu, Fisher and Ke, Lei},
    booktitle={ECCV},
    year={2024}
}
```
