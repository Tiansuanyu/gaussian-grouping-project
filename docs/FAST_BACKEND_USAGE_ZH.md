# 高速 gsplat 后端使用说明

这份说明给同组同学使用。当前分支已经接入了更快的 gsplat 渲染后端，训练、渲染和下游编辑脚本都会通过统一的 `gaussian_renderer.render()` 入口自动使用它。

## 1. 先切到加速分支

```bash
ssh <服务器账号>@<服务器地址>
cd /data2/cse12311966/projects/gaussian-grouping
git fetch origin
git checkout feature/gsplat-single-pass
git pull --ff-only
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gs
```

跑之前先看空闲 GPU：

```bash
nvidia-smi
```

后面的命令都建议显式指定 GPU，例如：

```bash
CUDA_VISIBLE_DEVICES=5 <command>
```

## 2. 后端开关怎么选

默认配置已经是加速后端：

```bash
USE_GSPLAT=1
USE_GSPLAT_NATIVE19=0
```

实际含义：

- 默认：shared-geometry split-raster gsplat 后端。它只依赖普通 gsplat，任何 stock gsplat 环境都能跑。
- 最快：`USE_GSPLAT_NATIVE19=1`。它使用 native 19-channel single-pass kernel，需要安装过 native19 patch 的 gsplat。
- 回退：`USE_GSPLAT=0`。使用原始 `diff-gaussian-rasterization` 后端，主要用于对照和排错。

推荐日常使用默认后端，不需要额外设置：

```bash
CUDA_VISIBLE_DEVICES=5 bash script/train_lerf.sh lerf/figurines 2
```

如果确认当前环境的 gsplat 已经打过 native19 patch，可以启用最快后端：

```bash
CUDA_VISIBLE_DEVICES=5 USE_GSPLAT_NATIVE19=1 bash script/train_lerf.sh lerf/figurines 2
```

注意：`USE_GSPLAT_NATIVE19=1` 会做能力检查。如果当前 gsplat 不支持 19 通道 kernel，程序会直接报错，不会静默退回 padded 32 通道。

## 3. 快速检查当前后端

```bash
python - <<'PY'
import gaussian_renderer
print("USE_GSPLAT =", gaussian_renderer.USE_GSPLAT)
print("USE_GSPLAT_NATIVE19 =", gaussian_renderer.USE_GSPLAT_NATIVE19)
if gaussian_renderer.USE_GSPLAT:
    print("native19_supported =", gaussian_renderer._native19_supported())
PY
```

预期：

- 默认加速后端：`USE_GSPLAT=True`, `USE_GSPLAT_NATIVE19=False`
- native19 最快后端：`USE_GSPLAT=True`, `USE_GSPLAT_NATIVE19=True`, `native19_supported=True`

## 4. 训练

LERF figurines 示例：

```bash
CUDA_VISIBLE_DEVICES=5 bash script/train_lerf.sh lerf/figurines 2
```

通用数据集示例：

```bash
CUDA_VISIBLE_DEVICES=5 bash script/train.sh bear 1
```

直接调用 `train.py` 也可以：

```bash
CUDA_VISIBLE_DEVICES=5 python train.py \
  -s data/lerf/figurines \
  -r 2 \
  -m output/lerf/figurines \
  --config_file config/gaussian_dataset/train.json \
  --train_split
```

训练输出会写到 `output/<dataset>`，例如 `output/lerf/figurines`。

## 5. 渲染和分割结果导出

训练结束后渲染 RGB、object feature 和预测 mask：

```bash
CUDA_VISIBLE_DEVICES=5 python render.py \
  -m output/lerf/figurines \
  --num_classes 256 \
  --images images
```

也可以直接用脚本训练加渲染：

```bash
CUDA_VISIBLE_DEVICES=5 bash script/train_lerf.sh lerf/figurines 2
```

## 6. 下游应用

### 6.1 物体删除

先准备好对应 config，例如 `config/object_removal/bear.json`，里面指定要删除的 object id 和阈值。

```bash
CUDA_VISIBLE_DEVICES=5 bash script/edit_object_removal.sh \
  output/bear \
  config/object_removal/bear.json
```

直接调用：

```bash
CUDA_VISIBLE_DEVICES=5 python edit_object_removal.py \
  -m output/bear \
  --config_file config/object_removal/bear.json
```

### 6.2 物体 inpaint

```bash
CUDA_VISIBLE_DEVICES=5 bash script/edit_object_inpaint.sh \
  output/bear \
  config/object_inpaint/bear.json
```

直接调用：

```bash
CUDA_VISIBLE_DEVICES=5 python edit_object_inpaint.py \
  -m output/bear \
  --config_file config/object_inpaint/bear.json
```

## 7. 推荐工作流

日常训练和应用：

```bash
CUDA_VISIBLE_DEVICES=5 bash script/train_lerf.sh lerf/figurines 2
CUDA_VISIBLE_DEVICES=5 python render.py -m output/lerf/figurines --num_classes 256 --images images
```

如果要压最快性能，并且确认 native19 patch 可用：

```bash
CUDA_VISIBLE_DEVICES=5 USE_GSPLAT_NATIVE19=1 bash script/train_lerf.sh lerf/figurines 2
CUDA_VISIBLE_DEVICES=5 USE_GSPLAT_NATIVE19=1 python render.py -m output/lerf/figurines --num_classes 256 --images images
```

排错时回到原始后端：

```bash
CUDA_VISIBLE_DEVICES=5 USE_GSPLAT=0 python render.py -m output/lerf/figurines --num_classes 256 --images images
```

## 8. 性能结论

clean profiling 结果见：

```text
profiling_results/TASK_B_PROFILING_SUMMARY.md
```

简要结论：

- 默认 shared-geometry split-raster 后端比早期 gsplat two-pass 更快，并且不需要改上游 gsplat。
- native19 single-pass 是当前 clean benchmark 里的最快后端。
- r2 figurines forward+backward 中，native19 比 shared split-raster 还快约 `15.9%`。
- r1 figurines 中，native19 仍然最快，但相对 shared split-raster 只快约 `2.1%`，两者很接近。

所以默认给大家用稳定兼容的 split 后端；需要极限速度时再显式打开 `USE_GSPLAT_NATIVE19=1`。

