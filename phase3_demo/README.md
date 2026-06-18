# Phase 3：3D 对象移除 Gradio Demo

这是一个轻量级前端，用于：

- 选择 `ramen`、`figurines` 或 `teatime` 场景。
- 浏览已有的 concat 结果和对象 ID 可视化。
- 从对象 ID 可视化图中多选对象 ID，并校验一个或多个非负整数对象 ID。
- 调整对象移除阈值 `removal_thresh`。
- 生成 `config/object_removal/lerf/{scene}_frontend.json`。
- 准备对象移除命令，或通过独立按钮显式执行 GPU 对象移除。
- 浏览已经存在的对象移除结果。

## SSHFS 与 Remote-SSH

可以通过本地 SSHFS 挂载目录，使用 VS Code/Codex 编辑本项目代码。项目中的
`output` 可能是指向远程绝对路径的符号链接，因此本地 SSHFS 环境不一定能访问
结果文件。

Conda、CUDA、训练、渲染和对象移除命令必须在连接真实服务器的 VS Code
Remote-SSH 终端中运行，不要在本地 SSHFS 终端运行。

## 启动 Demo

Demo 需要较新的 Gradio（建议 Gradio 4 或 5）。可在激活 `gs` 环境后检查：

```bash
python -c "import gradio; print(gradio.__version__)"
```

如果远端 `gs` 环境尚未安装 Gradio，只安装到当前 Conda 环境，不要全局安装：

```bash
python -m pip install "gradio>=4,<6"
```

在 VS Code Remote-SSH 终端中逐行运行：

```bash
conda activate gs
cd /data2/cse12311966/projects/gaussian-grouping-app
CUDA_VISIBLE_DEVICES=0 python phase3_demo/app.py
```

默认监听远程服务器的 `7860` 端口。若需要指定其他端口：

```bash
GRADIO_SERVER_PORT=7861 CUDA_VISIBLE_DEVICES=0 python phase3_demo/app.py
```

启动 Demo 本身不会执行对象移除。对象 ID Gallery 固定高度并支持纵向滚动；
点击 `id_037.png` 等图片会把 `37` 追加到对象 ID 输入框。重复点击同一个 ID
不会产生重复项。需要重选时点击 **Clear selected IDs**。

页面提供 `Removal threshold` 滑块，范围为 `0.05` 到 `0.9`，默认 `0.3`，
步长 `0.05`。该值会写入配置文件中的 `removal_thresh`。

页面还提供 **Max images to display** 滑块，默认 `24`，范围 `4` 到 `80`。
大场景可能包含数百张图片，例如 `figurines` 的 concat 目录较大；Demo 只预览
排序后的前 N 张图片，避免一次性把所有图片加载进 Gradio Gallery。切换场景或
调整该数值时，页面会先清空旧 Gallery，再加载新预览。

点击“生成配置并准备命令”后，页面会写入配置文件并显示类似下面的命令：

```bash
CUDA_VISIBLE_DEVICES=0 bash script/edit_object_removal.sh output/lerf/ramen config/object_removal/lerf/ramen_frontend.json
```

请检查对象 ID 和配置内容，再在 Remote-SSH 终端中手动执行该命令。

也可以点击独立的 **Run object removal on server** 按钮显式执行。该按钮会：

1. 校验场景、对象 ID 和 0–6 范围内的 GPU ID。
2. 使用当前对象 ID 和 `Removal threshold` 生成或更新当前场景的 frontend JSON
   配置。
3. 使用固定参数列表运行现有 `script/edit_object_removal.sh`，不使用
   `shell=True`。
4. 无论后端是否在原始 concat 阶段返回错误，都会继续运行
   `phase3_demo/make_removal_concat.py`。
5. 完成后刷新 Gallery，并优先显示 `concat_fixed`。
6. 在页面底部追加一条实验记录，包括 scene、selected IDs、removal threshold、
   output folder 和 `concat_fixed` 是否存在。
7. 同时把实验记录追加写入 `phase3_demo/experiment_log.md`，并在页面显示最新
   experiment summary。点击 **Refresh experiment log** 可重新读取完整 Markdown
   日志。

该操作会占用所选 GPU，且可能持续数分钟。日志会显示在页面中。只能在真实的
Remote-SSH GPU 服务器中使用该按钮，不要从本地 SSHFS 环境启动后点击运行。

## Suggested Experiment Protocol

建议按照以下顺序记录实验，方便最终报告中比较不同下游编辑效果：

1. Single-object removal：每个场景选择一个清晰可见的 ID，使用默认
   `removal_thresh=0.3`。
2. Multi-object removal：在同一场景中选择两个或更多 ID，观察多对象同时删除
   的视觉一致性。
3. Threshold comparison：固定同一组对象 ID，分别运行 `removal_thresh=0.2`、
   `0.3` 和 `0.6`，比较删除范围、边界残留和误删情况。
4. Repeat on all scenes：在 `ramen`、`figurines` 和 `teatime` 上重复上述实验。

每次点击 **Run object removal on server** 后，Demo 会追加记录到：

```text
phase3_demo/experiment_log.md
```

记录包含 scene、selected object IDs、removal threshold、GPU ID、timestamp、
output folder 和运行 status。

## 在 VS Code 转发 Gradio 端口

1. 保持 Remote-SSH 窗口及 Demo 进程运行。
2. 打开 VS Code 底部面板的 **PORTS（端口）** 标签。
3. 点击 **Forward a Port（转发端口）**，输入 `7860`。
4. 将端口可见性保持为本机私有，然后点击转发地址，或在本机浏览器打开
   `http://localhost:7860`。

如果使用了 `GRADIO_SERVER_PORT`，请转发对应端口。

## 目录约定

Demo 会检查以下目录：

```text
output/lerf/{scene}/train/ours_30000/concat/
output/lerf/{scene}/id_visualization/
output/lerf/{scene}/train/ours_object_removal/iteration_30000/concat/
output/lerf/{scene}/train/ours_object_removal/iteration_30000/concat_fixed/
```

如果目录、符号链接或图片不可访问，页面会显示具体提示。支持显示
`.png`、`.jpg`、`.jpeg`、`.webp` 和 `.bmp` 图片。

Gallery 默认只显示每个目录排序后的前 24 张图片。状态文字会显示总数和当前
展示数量，例如 `Found 300 images; displaying first 24.`。如果对象移除结果
尚未生成，After Gallery 会保持为空，并显示 `No removal result yet.`。

配置文件在用户点击按钮后生成：

```text
config/object_removal/lerf/{scene}_frontend.json
```

配置包含：

```json
{
    "num_classes": 256,
    "removal_thresh": 0.3,
    "select_obj_id": [37, 58]
}
```

## 生成对象 ID 高亮图

如果还不清楚场景中的物体对应哪个 ID，可以在真实的 Remote-SSH GPU 终端中
运行辅助脚本。它会加载指定迭代的 Gaussian 模型和分类器，只渲染第一个训练
视角，并为该视角中预测到的每个非零对象 ID 保存一张高亮图片。

```bash
conda activate gs
cd /data2/cse12311966/projects/gaussian-grouping-app
CUDA_VISIBLE_DEVICES=0 python phase3_demo/generate_id_visualization.py --scene ramen --gpu 0 --iteration 30000
```

脚本使用：

```text
模型目录：output/lerf/{scene}
数据目录：data/lerf/{scene}
分类器：output/lerf/{scene}/point_cloud/iteration_{iteration}/classifier.pth
输出目录：output/lerf/{scene}/id_visualization/
```

输出文件名形如 `id_037.png`、`id_058.png`。执行结束后，终端会打印已保存的
ID 列表和输出目录。刷新 Gradio 页面后，即可在“对象 ID 可视化”区域浏览这些
图片。

该脚本会调用 CUDA 渲染器。不要在本地 SSHFS 终端中运行，也不要通过 Gradio
页面自动执行；必须在真实的 Remote-SSH GPU 终端中手动运行。

## 修复对象移除拼接图

如果对象移除已经生成各个图像子目录，但后端因图片高度不同而无法完成
`np.hstack`，可运行纯 CPU 后处理脚本：

```bash
conda activate gs
cd /data2/cse12311966/projects/gaussian-grouping-app
python phase3_demo/make_removal_concat.py --scene ramen --iteration 30000
```

脚本读取：

```text
output/lerf/{scene}/train/ours_object_removal/iteration_{iteration}/
```

它以 `renders` 的每帧高度为基准，按比例缩放可用的 `gt`、`renders`、
`gt_objects_color`、`objects_pred` 和 `objects_feature16` 图片，再按该顺序横向
拼接。缺少整个子目录或某一帧时会打印警告，并继续处理其他可用图片。

修复后的比较图保存到：

```text
output/lerf/{scene}/train/ours_object_removal/iteration_{iteration}/concat_fixed/
```

该脚本仅使用 Pillow 读取、缩放和保存图片，不运行 CUDA、训练或后端渲染。

## Git 工作流

提交前先检查状态和本 Demo 的变更：

```bash
git status
git diff -- phase3_demo
git add phase3_demo
git commit -m "add phase3 object removal demo"
git push -u origin feature/phase3-app-yujiahua
```

不要直接使用 `git add -A`，除非已经检查并确认所有生成文件和其他变更都应该
进入提交。
