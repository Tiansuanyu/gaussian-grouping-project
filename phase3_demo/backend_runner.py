"""Filesystem and controlled execution helpers for the Phase 3 demo."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


SCENES = ("ramen", "figurines", "teatime")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_ITERATION = 30000
DEFAULT_NUM_CLASSES = 256
DEFAULT_REMOVAL_THRESHOLD = 0.3
DEFAULT_MAX_GALLERY_IMAGES = 24
MIN_GALLERY_IMAGES = 4
MAX_GALLERY_IMAGES = 80
MIN_GPU_ID = 0
MAX_GPU_ID = 6

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "output"
CONFIG_ROOT = PROJECT_ROOT / "config" / "object_removal" / "lerf"
EXPERIMENT_LOG_PATH = PROJECT_ROOT / "phase3_demo" / "experiment_log.md"


@dataclass(frozen=True)
class ScenePaths:
    scene_root: Path
    source_results: Path
    id_visualizations: Path
    removal_fixed_results: Path
    removal_results: Path
    config_file: Path


def validate_scene(scene: str) -> str:
    if scene not in SCENES:
        raise ValueError(f"未知场景：{scene!r}。可选值为：{', '.join(SCENES)}。")
    return scene


def parse_object_ids(raw_ids: str) -> list[int]:
    """Parse comma-separated, non-negative integer object IDs."""
    text = (raw_ids or "").strip()
    if not text:
        raise ValueError("请输入至少一个对象 ID，例如 37 或 37,58。")

    parts = [part.strip() for part in text.split(",")]
    if any(not part for part in parts):
        raise ValueError("对象 ID 列表中存在空项，请使用例如 37,58 的格式。")

    object_ids: list[int] = []
    for part in parts:
        if not re.fullmatch(r"\d+", part):
            raise ValueError(f"对象 ID {part!r} 无效；ID 必须是非负整数。")
        value = int(part)
        if value not in object_ids:
            object_ids.append(value)
    return object_ids


def validate_gpu_id(raw_gpu_id: str) -> str:
    gpu_id = str(raw_gpu_id).strip()
    if not re.fullmatch(r"\d+", gpu_id):
        raise ValueError("GPU ID 必须是 0 到 6 之间的整数。")
    if not MIN_GPU_ID <= int(gpu_id) <= MAX_GPU_ID:
        raise ValueError("GPU ID 必须是 0 到 6 之间的整数。")
    return gpu_id


def validate_removal_thresh(raw_thresh: float | str) -> float:
    try:
        removal_thresh = float(raw_thresh)
    except (TypeError, ValueError) as exc:
        raise ValueError("removal_thresh 必须是 0.05 到 0.9 之间的数字。") from exc
    if not 0.05 <= removal_thresh <= 0.9:
        raise ValueError("removal_thresh 必须在 0.05 到 0.9 之间。")
    return round(removal_thresh, 4)


def validate_max_images(raw_max_images: int | float | str) -> int:
    try:
        max_images = int(raw_max_images)
    except (TypeError, ValueError) as exc:
        raise ValueError("Max images to display 必须是 4 到 80 之间的整数。") from exc
    if not MIN_GALLERY_IMAGES <= max_images <= MAX_GALLERY_IMAGES:
        raise ValueError("Max images to display 必须在 4 到 80 之间。")
    return max_images


def get_scene_paths(scene: str) -> ScenePaths:
    scene = validate_scene(scene)
    scene_root = OUTPUT_ROOT / "lerf" / scene
    return ScenePaths(
        scene_root=scene_root,
        source_results=scene_root
        / "train"
        / f"ours_{DEFAULT_ITERATION}"
        / "concat",
        id_visualizations=scene_root / "id_visualization",
        removal_fixed_results=scene_root
        / "train"
        / "ours_object_removal"
        / f"iteration_{DEFAULT_ITERATION}"
        / "concat_fixed",
        removal_results=scene_root
        / "train"
        / "ours_object_removal"
        / f"iteration_{DEFAULT_ITERATION}"
        / "concat",
        config_file=CONFIG_ROOT / f"{scene}_frontend.json",
    )


def relative_to_project(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def output_access_warning() -> str | None:
    """Explain the common SSHFS failure mode for the absolute output symlink."""
    if OUTPUT_ROOT.exists():
        if OUTPUT_ROOT.is_dir():
            return None
        return f"`{relative_to_project(OUTPUT_ROOT)}` 存在，但不是目录。"

    if OUTPUT_ROOT.is_symlink():
        target = os.readlink(OUTPUT_ROOT)
        return (
            f"`output` 是指向 `{target}` 的符号链接，但当前环境无法访问该目标。"
            "这通常表示应用运行在本地 SSHFS 终端中；请在真实的 VS Code "
            "Remote-SSH 终端启动本 Demo。"
        )

    return (
        f"未找到输出目录 `{relative_to_project(OUTPUT_ROOT)}`。"
        "请确认结果已生成，并在真实的远程仓库中启动 Demo。"
    )


def list_images(
    directory: Path,
    max_images: int = DEFAULT_MAX_GALLERY_IMAGES,
    empty_message: str | None = None,
) -> tuple[list[tuple[str, str]], str, int]:
    """Return a Gradio gallery value and a readable directory status."""
    max_images = validate_max_images(max_images)
    warning = output_access_warning()
    if warning:
        return [], warning, 0
    if not directory.exists():
        message = empty_message or f"目录不存在：`{relative_to_project(directory)}`"
        return [], message, 0
    if not directory.is_dir():
        return [], f"路径不是目录：`{relative_to_project(directory)}`", 0

    try:
        images = sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: path.name,
        )
    except OSError as exc:
        return [], (
            f"无法读取目录 `{relative_to_project(directory)}`："
            f"{exc.strerror or str(exc)}"
        ), 0

    if not images:
        message = empty_message or (
            f"目录可访问，但没有支持的图片：`{relative_to_project(directory)}`"
        )
        return [], message, 0

    shown = images[:max_images]
    gallery = [(str(path), path.name) for path in shown]
    return gallery, (
        f"Found {len(images)} images; displaying first {len(shown)}. "
        f"`{relative_to_project(directory)}`"
    ), len(images)


def list_removal_images(
    directory: Path,
    max_images: int,
) -> tuple[list[tuple[str, str]], str, int]:
    return list_images(
        directory,
        max_images=max_images,
        empty_message="No removal result yet.",
    )


def _format_status(title: str, status: str, ok: bool) -> str:
    marker = "可用" if ok else "提示"
    return f"**{title}（{marker}）**：{status}"


def inspect_scene(
    scene: str,
    max_images: int = DEFAULT_MAX_GALLERY_IMAGES,
) -> dict[str, object]:
    """Collect gallery values and status text without changing the filesystem."""
    max_images = validate_max_images(max_images)
    paths = get_scene_paths(scene)
    source_images, source_status, _ = list_images(paths.source_results, max_images)
    id_images, id_status, _ = list_images(paths.id_visualizations, max_images)
    fixed_images, fixed_status, _ = list_removal_images(
        paths.removal_fixed_results,
        max_images,
    )
    if fixed_images:
        removal_images = fixed_images
        removal_status = (
            f"正在使用 `concat_fixed`。{fixed_status}"
        )
        selected_removal_path = paths.removal_fixed_results
    else:
        removal_images, fallback_status, _ = list_removal_images(
            paths.removal_results,
            max_images,
        )
        if removal_images:
            removal_status = (
                "`concat_fixed` 不可用或没有图片，已回退到 `concat`。"
                f"`concat_fixed` 状态：{fixed_status}；"
                f"`concat` 状态：{fallback_status}"
            )
        else:
            removal_status = "No removal result yet."
        selected_removal_path = paths.removal_results

    source_ok = bool(source_images)
    ids_ok = bool(id_images)
    removal_ok = bool(removal_images)
    status = "\n\n".join(
        (
            _format_status("原始可视化", source_status, source_ok),
            _format_status("对象 ID 可视化", id_status, ids_ok),
            _format_status("移除后结果", removal_status, removal_ok),
            f"**当前移除后目录**：`{relative_to_project(selected_removal_path)}`",
        )
    )
    return {
        "source_images": source_images,
        "id_images": id_images,
        "removal_images": removal_images,
        "status": status,
        "output_directory": relative_to_project(selected_removal_path),
        "removal_source": "concat_fixed" if fixed_images else "concat",
        "concat_fixed_available": bool(fixed_images),
    }


def build_config(
    object_ids: Iterable[int],
    removal_thresh: float = DEFAULT_REMOVAL_THRESHOLD,
) -> dict[str, object]:
    return {
        "num_classes": DEFAULT_NUM_CLASSES,
        "removal_thresh": validate_removal_thresh(removal_thresh),
        "select_obj_id": list(object_ids),
    }


def write_config(
    scene: str,
    raw_ids: str,
    removal_thresh: float = DEFAULT_REMOVAL_THRESHOLD,
) -> tuple[Path, dict[str, object]]:
    """Write the frontend config after explicit user interaction."""
    paths = get_scene_paths(scene)
    config = build_config(parse_object_ids(raw_ids), removal_thresh)
    try:
        paths.config_file.parent.mkdir(parents=True, exist_ok=True)
        with paths.config_file.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=4)
            handle.write("\n")
    except OSError as exc:
        raise RuntimeError(
            f"无法写入配置 `{relative_to_project(paths.config_file)}`："
            f"{exc.strerror or str(exc)}"
        ) from exc
    return paths.config_file, config


def build_removal_command(scene: str, gpu_id: str) -> str:
    scene = validate_scene(scene)
    gpu_id = validate_gpu_id(gpu_id)
    scene_path = f"output/lerf/{scene}"
    config_path = f"config/object_removal/lerf/{scene}_frontend.json"
    return (
        f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu_id)} "
        "bash script/edit_object_removal.sh "
        f"{shlex.quote(scene_path)} {shlex.quote(config_path)}"
    )


def prepare_removal(
    scene: str,
    raw_ids: str,
    gpu_id: str,
    removal_thresh: float = DEFAULT_REMOVAL_THRESHOLD,
) -> tuple[str, str, str]:
    """Generate a config and command, but never execute the command."""
    gpu_id = validate_gpu_id(gpu_id)
    config_path, config = write_config(scene, raw_ids, removal_thresh)
    command = build_removal_command(scene, gpu_id)
    config_preview = json.dumps(config, indent=4)
    status = (
        f"配置已写入 `{relative_to_project(config_path)}`，"
        f"`removal_thresh={config['removal_thresh']}`。\n\n"
        "GPU 命令未执行。请复制下方命令到真实的 Remote-SSH 终端运行。"
    )
    return status, config_preview, command


def _ids_to_text(object_ids: Iterable[int]) -> str:
    return ",".join(str(object_id) for object_id in object_ids)


def format_latest_experiment_summary(experiment: dict[str, object]) -> str:
    ids = _ids_to_text(experiment["selected_ids"])
    concat_fixed = "yes" if experiment["concat_fixed_exists"] else "no"
    return "\n".join(
        (
            "### Latest Experiment Summary",
            f"- Timestamp: `{experiment['timestamp']}`",
            f"- Scene: `{experiment['scene']}`",
            f"- Selected object IDs: `{ids}`",
            f"- removal_thresh: `{experiment['removal_thresh']}`",
            f"- GPU ID: `{experiment['gpu_id']}`",
            f"- Output folder: `{experiment['output_folder']}`",
            f"- concat_fixed exists: `{concat_fixed}`",
            f"- Status: {experiment['status']}",
        )
    )


def append_experiment_log(experiment: dict[str, object]) -> None:
    EXPERIMENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not EXPERIMENT_LOG_PATH.exists():
        EXPERIMENT_LOG_PATH.write_text("# Phase 3 Experiment Log\n\n", encoding="utf-8")

    ids = _ids_to_text(experiment["selected_ids"])
    concat_fixed = "yes" if experiment["concat_fixed_exists"] else "no"
    record = "\n".join(
        (
            f"## {experiment['timestamp']} - {experiment['scene']}",
            "",
            f"- Scene: `{experiment['scene']}`",
            f"- Selected object IDs: `{ids}`",
            f"- removal_thresh: `{experiment['removal_thresh']}`",
            f"- GPU ID: `{experiment['gpu_id']}`",
            f"- Output folder: `{experiment['output_folder']}`",
            f"- concat_fixed exists: `{concat_fixed}`",
            f"- Status: {experiment['status']}",
            "",
        )
    )
    with EXPERIMENT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(record)


def read_experiment_log() -> str:
    if not EXPERIMENT_LOG_PATH.exists():
        return "_No persisted experiment log yet._"
    try:
        text = EXPERIMENT_LOG_PATH.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return f"_Could not read experiment log: {exc}_"
    return text or "_Experiment log is empty._"


def _format_process_log(
    title: str,
    command: list[str],
    completed: subprocess.CompletedProcess[str] | None,
    error: OSError | None = None,
) -> str:
    lines = [f"===== {title} =====", f"$ {shlex.join(command)}"]
    if error is not None:
        lines.extend((f"启动失败：{error}", ""))
        return "\n".join(lines)

    assert completed is not None
    lines.append(f"退出码：{completed.returncode}")
    if completed.stdout:
        lines.extend(("", "--- stdout ---", completed.stdout.rstrip()))
    if completed.stderr:
        lines.extend(("", "--- stderr ---", completed.stderr.rstrip()))
    lines.append("")
    return "\n".join(lines)


def _run_captured(
    command: list[str],
    env: dict[str, str],
) -> tuple[subprocess.CompletedProcess[str] | None, OSError | None]:
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
        return completed, None
    except OSError as exc:
        return None, exc


def run_object_removal(
    scene: str,
    raw_ids: str,
    gpu_id: str,
    removal_thresh: float = DEFAULT_REMOVAL_THRESHOLD,
    max_images: int = DEFAULT_MAX_GALLERY_IMAGES,
) -> dict[str, object]:
    """Run the fixed backend command, then always attempt CPU post-processing."""
    scene = validate_scene(scene)
    gpu_id = validate_gpu_id(gpu_id)
    object_ids = parse_object_ids(raw_ids)
    config_path, config = write_config(
        scene,
        ",".join(str(object_id) for object_id in object_ids),
        removal_thresh,
    )

    relative_config = relative_to_project(config_path)
    model_path = f"output/lerf/{scene}"
    backend_command = [
        "bash",
        "script/edit_object_removal.sh",
        model_path,
        relative_config,
    ]
    postprocess_command = [
        sys.executable,
        "phase3_demo/make_removal_concat.py",
        "--scene",
        scene,
        "--iteration",
        str(DEFAULT_ITERATION),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    backend_result, backend_error = _run_captured(backend_command, env)
    postprocess_result, postprocess_error = _run_captured(postprocess_command, env)

    backend_ok = (
        backend_error is None
        and backend_result is not None
        and backend_result.returncode == 0
    )
    postprocess_ok = (
        postprocess_error is None
        and postprocess_result is not None
        and postprocess_result.returncode == 0
    )

    logs = "\n".join(
        (
            _format_process_log(
                "对象移除后端",
                backend_command,
                backend_result,
                backend_error,
            ),
            _format_process_log(
                "concat_fixed 后处理",
                postprocess_command,
                postprocess_result,
                postprocess_error,
            ),
        )
    )

    scene_result = inspect_scene(scene, max_images=max_images)
    fixed_available = str(scene_result["output_directory"]).endswith(
        "/concat_fixed"
    ) and bool(scene_result["removal_images"])

    if backend_ok and postprocess_ok:
        status = (
            f"**执行成功**：对象移除和 `concat_fixed` 后处理均已完成。"
            f"使用 GPU {gpu_id}。"
        )
    elif fixed_available:
        status = (
            "**部分成功**：后端或后处理返回了非零退出码，但已找到可显示的 "
            "`concat_fixed` 结果。请检查运行日志。"
        )
    elif backend_ok:
        status = (
            "**对象移除已完成，但后处理失败**：未找到可显示的 "
            "`concat_fixed` 结果，请检查日志。"
        )
    else:
        status = (
            "**执行失败**：对象移除未正常完成，且没有找到可显示的 "
            "`concat_fixed` 结果。请检查日志。"
        )

    timestamp = datetime.now().isoformat(timespec="seconds")
    experiment = {
        "timestamp": timestamp,
        "scene": scene,
        "selected_ids": object_ids,
        "removal_thresh": config["removal_thresh"],
        "gpu_id": gpu_id,
        "output_folder": scene_result["output_directory"],
        "concat_fixed_exists": bool(scene_result["concat_fixed_available"]),
        "status": status,
    }
    append_experiment_log(experiment)

    return {
        "status": status,
        "log": logs,
        "config_preview": json.dumps(config, indent=4),
        "command_preview": build_removal_command(scene, gpu_id),
        "scene_result": scene_result,
        "experiment": experiment,
        "latest_summary": format_latest_experiment_summary(experiment),
    }
