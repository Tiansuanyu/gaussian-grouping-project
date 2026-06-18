"""Gradio frontend for inspecting and preparing 3D object removal."""

from __future__ import annotations

import inspect
import os
import re
from pathlib import Path
from typing import Any

import gradio as gr

from backend_runner import (
    DEFAULT_MAX_GALLERY_IMAGES,
    MAX_GALLERY_IMAGES,
    MIN_GALLERY_IMAGES,
    SCENES,
    inspect_scene,
    parse_object_ids,
    prepare_removal,
    read_experiment_log,
    run_object_removal,
)


APP_TITLE = "Gaussian Grouping：3D 对象移除 Demo"
APP_CSS = """
#id-gallery-scroll {
    height: 500px;
    max-height: 500px;
    overflow-y: auto !important;
}
#id-gallery-scroll .grid-container,
#id-gallery-scroll [data-testid="gallery"] {
    max-height: 460px !important;
    overflow-y: auto !important;
}
"""


def refresh_scene(scene: str, max_images: int):
    try:
        result = inspect_scene(scene, max_images=max_images)
        return (
            result["source_images"],
            result["id_images"],
            result["removal_images"],
            result["status"],
            result["output_directory"],
        )
    except Exception as exc:
        message = f"**读取失败**：{exc}"
        return [], [], [], message, ""


def clear_scene_galleries():
    return [], [], [], "**Loading scene preview...**", ""


def generate_config_and_command(
    scene: str,
    object_ids: str,
    gpu_id: str,
    removal_thresh: float,
):
    try:
        return prepare_removal(scene, object_ids, gpu_id, removal_thresh)
    except (ValueError, RuntimeError) as exc:
        return f"**生成失败**：{exc}", "", ""
    except Exception as exc:
        return f"**意外错误**：{exc}", "", ""


def _find_id_filename(value: Any) -> str | None:
    if isinstance(value, str):
        name = Path(value).name
        if re.fullmatch(r"id_\d+\.[A-Za-z0-9]+", name):
            return name
        return None
    if isinstance(value, dict):
        for key in ("caption", "path", "name", "image", "value"):
            if key in value:
                found = _find_id_filename(value[key])
                if found:
                    return found
        for nested_value in value.values():
            found = _find_id_filename(nested_value)
            if found:
                return found
    if isinstance(value, (list, tuple)):
        for nested_value in value:
            found = _find_id_filename(nested_value)
            if found:
                return found
    return None


def select_object_id(current_ids: str, event: gr.SelectData):
    filename = _find_id_filename(event.value)
    if not filename:
        raise gr.Error("无法从所选图片中解析对象 ID。")
    match = re.fullmatch(r"id_(\d+)\.[A-Za-z0-9]+", filename)
    if not match:
        raise gr.Error(f"文件名格式不支持：{filename}")
    selected_id = int(match.group(1))
    try:
        existing_ids = parse_object_ids(current_ids) if current_ids.strip() else []
    except ValueError as exc:
        raise gr.Error(f"当前对象 ID 输入无效，请先清空或修正：{exc}") from exc
    if selected_id not in existing_ids:
        existing_ids.append(selected_id)
    return ",".join(str(object_id) for object_id in existing_ids)


def clear_selected_ids():
    return ""


def format_experiment_log(experiments: list[dict[str, object]] | None) -> str:
    if not experiments:
        return "_No experiments recorded yet._"
    lines = [
        "| # | Time | Scene | IDs | Threshold | GPU | Output Folder | concat_fixed |",
        "|---:|---|---|---|---:|---:|---|---|",
    ]
    for index, experiment in enumerate(experiments, start=1):
        ids = ",".join(str(value) for value in experiment["selected_ids"])
        concat_fixed = "yes" if experiment["concat_fixed_exists"] else "no"
        lines.append(
            "| {index} | `{time}` | {scene} | `{ids}` | {threshold:.2f} | {gpu} | `{folder}` | {fixed} |".format(
                index=index,
                time=experiment.get("timestamp", "-"),
                scene=experiment["scene"],
                ids=ids,
                threshold=float(experiment["removal_thresh"]),
                gpu=experiment.get("gpu_id", "-"),
                folder=experiment["output_folder"],
                fixed=concat_fixed,
            )
        )
    return "\n".join(lines)


def run_removal_from_ui(
    scene: str,
    object_ids: str,
    gpu_id: str,
    removal_thresh: float,
    max_images: int,
    experiments: list[dict[str, object]] | None,
):
    try:
        result = run_object_removal(
            scene,
            object_ids,
            gpu_id,
            removal_thresh,
            max_images=max_images,
        )
        scene_result = result["scene_result"]
        updated_experiments = list(experiments or [])
        updated_experiments.append(result["experiment"])
        return (
            result["status"],
            result["log"],
            result["config_preview"],
            result["command_preview"],
            scene_result["source_images"],
            scene_result["id_images"],
            scene_result["removal_images"],
            scene_result["status"],
            scene_result["output_directory"],
            updated_experiments,
            format_experiment_log(updated_experiments),
            result["latest_summary"],
            read_experiment_log(),
        )
    except (ValueError, RuntimeError) as exc:
        scene_data = refresh_scene(scene, max_images)
        current_experiments = list(experiments or [])
        return (
            f"**执行前校验失败**：{exc}",
            str(exc),
            "",
            "",
            *scene_data,
            current_experiments,
            format_experiment_log(current_experiments),
            "_No new experiment was recorded._",
            read_experiment_log(),
        )
    except Exception as exc:
        scene_data = refresh_scene(scene, max_images)
        current_experiments = list(experiments or [])
        return (
            f"**执行失败**：{exc}",
            str(exc),
            "",
            "",
            *scene_data,
            current_experiments,
            format_experiment_log(current_experiments),
            "_No new experiment was recorded._",
            read_experiment_log(),
        )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE, css=APP_CSS) as demo:
        gr.Markdown(
            f"# {APP_TITLE}\n"
            "浏览已有分割结果、生成配置，并可显式启动对象移除。"
        )
        gr.Markdown(
            "### Multi-Object Editing\n"
            "This demo supports 3D object removal and multi-object editing "
            "based on Gaussian Grouping object IDs."
        )

        with gr.Row():
            scene = gr.Dropdown(
                choices=list(SCENES),
                value=SCENES[0],
                label="场景",
                interactive=True,
            )
            max_images = gr.Slider(
                minimum=MIN_GALLERY_IMAGES,
                maximum=MAX_GALLERY_IMAGES,
                value=DEFAULT_MAX_GALLERY_IMAGES,
                step=1,
                label="Max images to display",
            )
            refresh_button = gr.Button("刷新场景结果", variant="secondary")

        scene_status = gr.Markdown()
        output_directory = gr.Textbox(
            label="当前移除后输出目录（可编辑）",
            interactive=True,
        )

        gr.Markdown("## Before / After")
        with gr.Row():
            source_gallery = gr.Gallery(
                label="Before：原始 concat 结果",
                columns=2,
                height=420,
                object_fit="contain",
            )
            removal_gallery = gr.Gallery(
                label="After：已有对象移除结果",
                columns=2,
                height=420,
                object_fit="contain",
            )

        gr.Markdown(
            "## 对象 ID 可视化\n"
            "每张图片下方显示完整文件名，例如 `id_037.png` 对应对象 ID 37。"
        )
        with gr.Column(elem_id="id-gallery-scroll"):
            id_gallery = gr.Gallery(
                label="对象 ID 可视化（点击图片即可选择 ID）",
                columns=4,
                height=460,
                object_fit="contain",
            )

        gr.Markdown("## 生成配置与命令")
        with gr.Row():
            object_ids = gr.Textbox(
                label="对象 ID",
                value="37",
                placeholder="例如：37 或 37,58",
            )
            gpu_id = gr.Textbox(
                label="GPU ID",
                value="0",
                placeholder="例如：0",
            )
        with gr.Row():
            clear_ids_button = gr.Button("Clear selected IDs", variant="secondary")
            removal_thresh = gr.Slider(
                minimum=0.05,
                maximum=0.9,
                value=0.3,
                step=0.05,
                label="Removal threshold",
            )

        generate_button = gr.Button("生成配置并准备命令", variant="primary")
        generation_status = gr.Markdown()
        with gr.Row():
            config_preview = gr.Code(
                label="配置 JSON",
                language="json",
                interactive=False,
            )
            command_preview = gr.Code(
                label="在 Remote-SSH 终端手动运行的命令",
                language="shell",
                interactive=False,
            )

        gr.Markdown(
            "### 在服务器执行\n"
            "**警告：下方按钮会使用所选 GPU 运行对象移除，可能需要数分钟。**"
            "仅应在真实的 Remote-SSH GPU 服务器上启动本 Demo。"
        )
        run_button = gr.Button(
            "Run object removal on server",
            variant="stop",
        )
        run_status = gr.Markdown()
        run_log = gr.Textbox(
            label="对象移除与后处理日志",
            lines=20,
            max_lines=40,
            interactive=False,
        )
        experiment_state = gr.State([])
        gr.Markdown("## Experiment Results")
        latest_experiment_summary = gr.Markdown("_No experiment has finished yet._")
        experiment_log = gr.Markdown(format_experiment_log([]))
        refresh_log_button = gr.Button("Refresh experiment log", variant="secondary")
        persisted_experiment_log = gr.Markdown(read_experiment_log())

        refresh_outputs = [
            source_gallery,
            id_gallery,
            removal_gallery,
            scene_status,
            output_directory,
        ]
        scene.change(
            clear_scene_galleries,
            inputs=[],
            outputs=refresh_outputs,
        ).then(
            refresh_scene,
            inputs=[scene, max_images],
            outputs=refresh_outputs,
        )
        max_images.change(
            clear_scene_galleries,
            inputs=[],
            outputs=refresh_outputs,
        ).then(
            refresh_scene,
            inputs=[scene, max_images],
            outputs=refresh_outputs,
        )
        refresh_button.click(
            clear_scene_galleries,
            inputs=[],
            outputs=refresh_outputs,
        ).then(
            refresh_scene,
            inputs=[scene, max_images],
            outputs=refresh_outputs,
        )
        demo.load(refresh_scene, inputs=[scene, max_images], outputs=refresh_outputs)
        id_gallery.select(
            select_object_id,
            inputs=object_ids,
            outputs=object_ids,
        )
        clear_ids_button.click(clear_selected_ids, inputs=[], outputs=object_ids)

        generate_button.click(
            generate_config_and_command,
            inputs=[scene, object_ids, gpu_id, removal_thresh],
            outputs=[generation_status, config_preview, command_preview],
        )
        run_button.click(
            run_removal_from_ui,
            inputs=[
                scene,
                object_ids,
                gpu_id,
                removal_thresh,
                max_images,
                experiment_state,
            ],
            outputs=[
                run_status,
                run_log,
                config_preview,
                command_preview,
                source_gallery,
                id_gallery,
                removal_gallery,
                scene_status,
                output_directory,
                experiment_state,
                experiment_log,
                latest_experiment_summary,
                persisted_experiment_log,
            ],
        )
        refresh_log_button.click(
            read_experiment_log,
            inputs=[],
            outputs=persisted_experiment_log,
        )

    return demo


if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo = build_demo()
    launch_options = {
        "server_name": "127.0.0.1",
        "server_port": port,
        "share": False,
    }
    if "show_api" in inspect.signature(demo.launch).parameters:
        launch_options["show_api"] = False
    demo.launch(**launch_options)
