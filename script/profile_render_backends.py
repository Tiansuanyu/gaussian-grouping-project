import argparse
import csv
import inspect
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arguments import OptimizationParams, PipelineParams
import gaussian_renderer
from gaussian_renderer import _render_gsplat, _render_original
from scene import GaussianModel, Scene
from utils.sh_utils import eval_sh


try:
    from gsplat import rasterization as gsplat_rasterize
except ImportError:
    gsplat_rasterize = None


def _gsplat_metadata() -> Dict:
    try:
        import gsplat
        from gsplat.cuda import _wrapper

        return {
            "gsplat_file": getattr(gsplat, "__file__", None),
            "gsplat_version": getattr(gsplat, "__version__", None),
            "gsplat_wrapper_file": getattr(_wrapper, "__file__", None),
            "native19_supported": _native19_supported(),
        }
    except Exception as exc:
        return {
            "gsplat_file": None,
            "gsplat_version": None,
            "gsplat_wrapper_file": None,
            "native19_supported": False,
            "gsplat_metadata_error": str(exc),
        }


def _native19_supported() -> bool:
    from gsplat.cuda import _wrapper

    try:
        src = inspect.getsource(_wrapper.rasterize_to_pixels)
    except OSError:
        src = inspect.getsource(_wrapper)

    match = re.search(r"if channels not in\s*\((.*?)\):", src, re.DOTALL)
    if match is None:
        return False
    supported_channels = {
        int(value)
        for value in re.findall(r"\b\d+\b", match.group(1))
    }
    return 19 in supported_channels


def _assert_native19_supported():
    if _native19_supported():
        return
    metadata = _gsplat_metadata()
    raise RuntimeError(
        "Installed gsplat does not appear to have the native-19 patch; "
        "'gsplat_native19' would silently pad to 32. "
        "Re-apply profiling_results/gsplat_native19_remote.patch. "
        f"gsplat_file={metadata.get('gsplat_file')} "
        f"wrapper_file={metadata.get('gsplat_wrapper_file')}"
    )

try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )

    gaussian_renderer.GaussianRasterizationSettings = GaussianRasterizationSettings
    gaussian_renderer.GaussianRasterizer = GaussianRasterizer
except ImportError:
    GaussianRasterizationSettings = None
    GaussianRasterizer = None


def _load_cfg_args(model_path: str) -> Namespace:
    cfg_path = Path(model_path) / "cfg_args"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing cfg_args: {cfg_path}")

    namespace = {"Namespace": Namespace}
    cfg = eval(cfg_path.read_text(), namespace)
    if not isinstance(cfg, Namespace):
        raise ValueError(f"cfg_args did not evaluate to argparse.Namespace: {cfg_path}")
    return cfg


def _make_pipeline(convert_shs_python: bool = False) -> Namespace:
    parser = argparse.ArgumentParser()
    PipelineParams(parser)
    pipe = parser.parse_args([])
    pipe.convert_SHs_python = convert_shs_python
    return pipe


def _make_optimization_args() -> Namespace:
    parser = argparse.ArgumentParser()
    OptimizationParams(parser)
    return parser.parse_args([])


def _render_gsplat_two_pass(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0, override_color=None):
    if gsplat_rasterize is None:
        raise ImportError("gsplat is not available")

    W = int(viewpoint_camera.image_width)
    H = int(viewpoint_camera.image_height)

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    fx = W / (2.0 * tanfovx)
    fy = H / (2.0 * tanfovy)

    Ks = torch.tensor(
        [
            [fx, 0.0, W / 2.0],
            [0.0, fy, H / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device="cuda",
    ).unsqueeze(0)

    viewmats = viewpoint_camera.world_view_transform.transpose(0, 1).unsqueeze(0)

    means = pc.get_xyz
    opacities = pc.get_opacity.squeeze(-1)
    scales = pc.get_scaling
    quats = pc.get_rotation

    if scaling_modifier != 1.0:
        scales = scales * scaling_modifier

    if override_color is not None:
        colors = override_color
        sh_degree_to_use = None
    elif pipe.convert_SHs_python:
        shs_view = pc.get_features.transpose(1, 2).view(
            -1, 3, (pc.max_sh_degree + 1) ** 2
        )
        dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
            pc.get_features.shape[0], 1
        )
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors = torch.clamp_min(sh2rgb + 0.5, 0.0)
        sh_degree_to_use = None
    else:
        colors = pc.get_features
        sh_degree_to_use = pc.active_sh_degree

    render_colors, _, meta = gsplat_rasterize(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=W,
        height=H,
        sh_degree=sh_degree_to_use,
        backgrounds=bg_color.unsqueeze(0),
        packed=False,
    )
    rendered_image = render_colors[0].permute(2, 0, 1)

    N = means.shape[0]
    obj_feat = pc.get_objects.view(N, -1)
    render_objs, _, _ = gsplat_rasterize(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=obj_feat,
        viewmats=viewmats,
        Ks=Ks,
        width=W,
        height=H,
        sh_degree=None,
        packed=False,
    )
    rendered_objects = render_objs[0].permute(2, 0, 1)

    radii = meta["radii"].squeeze(0)
    radii_scalar = radii.max(dim=-1).values.int()

    screenspace_points = torch.zeros(
        N, 3, dtype=means.dtype, device="cuda", requires_grad=True
    )

    def _capture_grad(grad):
        g = grad.squeeze(0)
        screenspace_points.grad = torch.cat(
            [g, torch.zeros(N, 1, device=g.device, dtype=g.dtype)], dim=-1
        )

    if meta["means2d"].requires_grad:
        meta["means2d"].register_hook(_capture_grad)

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii_scalar > 0,
        "radii": radii_scalar,
        "render_object": rendered_objects,
    }


def _render_gsplat_padded_single(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0, override_color=None):
    from gsplat.cuda._wrapper import (
        fully_fused_projection,
        isect_offset_encode,
        isect_tiles,
        rasterize_to_pixels,
        spherical_harmonics,
    )

    W = int(viewpoint_camera.image_width)
    H = int(viewpoint_camera.image_height)

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    fx = W / (2.0 * tanfovx)
    fy = H / (2.0 * tanfovy)

    Ks = torch.tensor(
        [
            [fx, 0.0, W / 2.0],
            [0.0, fy, H / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device="cuda",
    ).unsqueeze(0)
    viewmats = viewpoint_camera.world_view_transform.transpose(0, 1).unsqueeze(0)

    means = pc.get_xyz
    opacities = pc.get_opacity.squeeze(-1)
    scales = pc.get_scaling
    quats = pc.get_rotation
    if scaling_modifier != 1.0:
        scales = scales * scaling_modifier

    radii, means2d, depths, conics, compensations = fully_fused_projection(
        means=means,
        covars=None,
        quats=quats,
        scales=scales,
        viewmats=viewmats,
        Ks=Ks,
        width=W,
        height=H,
        eps2d=0.3,
        packed=False,
        near_plane=0.01,
        far_plane=1e10,
        radius_clip=0.0,
        sparse_grad=False,
        calc_compensations=False,
        camera_model="pinhole",
        opacities=opacities,
    )

    opacities = opacities.unsqueeze(0)
    if compensations is not None:
        opacities = opacities * compensations

    if override_color is not None:
        rgb = override_color.unsqueeze(0)
    elif pipe.convert_SHs_python:
        shs_view = pc.get_features.transpose(1, 2).view(
            -1, 3, (pc.max_sh_degree + 1) ** 2
        )
        dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
            pc.get_features.shape[0], 1
        )
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        rgb = torch.clamp_min(sh2rgb + 0.5, 0.0).unsqueeze(0)
    else:
        campos = torch.inverse(viewmats)[..., :3, 3]
        dirs = means[None, :, :] - campos[:, None, :]
        masks = (radii > 0).all(dim=-1)
        shs = pc.get_features.unsqueeze(0)
        rgb = spherical_harmonics(pc.active_sh_degree, dirs, shs, masks=masks)
        rgb = torch.clamp_min(rgb + 0.5, 0.0)

    N = means.shape[0]
    obj_feat = pc.get_objects.view(N, -1).unsqueeze(0)
    colors = torch.cat([rgb, obj_feat], dim=-1)
    backgrounds = torch.cat(
        [
            bg_color,
            torch.zeros(obj_feat.shape[-1], dtype=bg_color.dtype, device=bg_color.device),
        ],
        dim=0,
    ).unsqueeze(0)

    tile_size = 16
    tile_width = math.ceil(W / float(tile_size))
    tile_height = math.ceil(H / float(tile_size))
    _, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=False,
        n_images=1,
        image_ids=None,
        gaussian_ids=None,
    )
    isect_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)

    render_colors, _ = rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities,
        W,
        H,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
        packed=False,
        absgrad=False,
    )
    render_colors = render_colors[0]
    rendered_image = render_colors[..., :3].permute(2, 0, 1)
    rendered_objects = render_colors[..., 3:].permute(2, 0, 1)

    radii_scalar = radii.squeeze(0).max(dim=-1).values.int()
    screenspace_points = torch.zeros(
        N, 3, dtype=means.dtype, device="cuda", requires_grad=True
    )

    def _capture_grad(grad):
        g = grad.squeeze(0)
        screenspace_points.grad = torch.cat(
            [g, torch.zeros(N, 1, device=g.device, dtype=g.dtype)], dim=-1
        )

    if means2d.requires_grad:
        means2d.register_hook(_capture_grad)

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii_scalar > 0,
        "radii": radii_scalar,
        "render_object": rendered_objects,
    }


def _render_diff(viewpoint_camera, pc, pipe, bg_color):
    if GaussianRasterizationSettings is None or GaussianRasterizer is None:
        raise ImportError("diff_gaussian_rasterization is not available")
    return _render_original(viewpoint_camera, pc, pipe, bg_color)


def _zero_grad(gaussians: GaussianModel):
    for tensor in (
        gaussians._xyz,
        gaussians._features_dc,
        gaussians._features_rest,
        gaussians._scaling,
        gaussians._rotation,
        gaussians._opacity,
        gaussians._objects_dc,
    ):
        if tensor.grad is not None:
            tensor.grad = None


def _run_once(render_fn: Callable, view, gaussians, pipe, background, include_backward: bool):
    out = render_fn(view, gaussians, pipe, background)
    loss = out["render"].mean() + out["render_object"].mean()
    if include_backward:
        loss.backward()
    return out


def _percentile(values: List[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of an empty list")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summarize_times(times_ms: List[float]) -> Dict:
    if not times_ms:
        raise ValueError("No timing samples collected")
    return {
        "samples": len(times_ms),
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.fmean(times_ms),
        "std_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min_ms": min(times_ms),
        "p05_ms": _percentile(times_ms, 0.05),
        "p95_ms": _percentile(times_ms, 0.95),
        "max_ms": max(times_ms),
    }


def _event_times_interleaved(
    cases: List[Dict],
    views: List[Dict],
    gaussians,
    pipe,
    background,
    warmup: int,
    iters: int,
    shuffle_seed: int,
) -> Dict[str, List[Dict]]:
    timings = {case["key"]: [] for case in cases}
    rng = random.Random(shuffle_seed)

    for _ in range(warmup):
        for view_info in views:
            case_order = cases[:]
            rng.shuffle(case_order)
            for case in case_order:
                _zero_grad(gaussians)
                _run_once(
                    case["render_fn"],
                    view_info["view"],
                    gaussians,
                    pipe,
                    background,
                    case["include_backward"],
                )
    torch.cuda.synchronize()

    for _ in range(iters):
        for view_info in views:
            case_order = cases[:]
            rng.shuffle(case_order)
            for case in case_order:
                _zero_grad(gaussians)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _run_once(
                    case["render_fn"],
                    view_info["view"],
                    gaussians,
                    pipe,
                    background,
                    case["include_backward"],
                )
                end.record()
                torch.cuda.synchronize()
                timings[case["key"]].append(
                    {
                        "view_index": view_info["index"],
                        "view_order": view_info["order"],
                        "time_ms": start.elapsed_time(end),
                    }
                )

    return timings


def _memory_peak(
    render_fn: Callable,
    views: List[Dict],
    gaussians,
    pipe,
    background,
    warmup: int,
    iters: int,
    include_backward: bool,
) -> float:
    for _ in range(warmup):
        for view_info in views:
            _zero_grad(gaussians)
            _run_once(
                render_fn,
                view_info["view"],
                gaussians,
                pipe,
                background,
                include_backward,
            )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    for _ in range(iters):
        for view_info in views:
            _zero_grad(gaussians)
            _run_once(
                render_fn,
                view_info["view"],
                gaussians,
                pipe,
                background,
                include_backward,
            )
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)


def _profile(
    render_fn: Callable,
    view,
    gaussians,
    pipe,
    background,
    warmup: int,
    iters: int,
    include_backward: bool,
):
    for _ in range(warmup):
        _zero_grad(gaussians)
        _run_once(render_fn, view, gaussians, pipe, background, include_backward)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(iters):
            _zero_grad(gaussians)
            _run_once(render_fn, view, gaussians, pipe, background, include_backward)
            prof.step()
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    return prof, peak_mb


def _rows_from_prof(prof, row_limit: int):
    rows = []
    for item in prof.key_averages():
        cuda_total_us = getattr(item, "cuda_time_total", 0.0)
        if cuda_total_us <= 0:
            continue
        rows.append(
            {
                "name": item.key,
                "calls": item.count,
                "cuda_total_ms": cuda_total_us / 1000.0,
                "cuda_avg_ms": (cuda_total_us / max(item.count, 1)) / 1000.0,
                "self_cuda_total_ms": getattr(item, "self_cuda_time_total", 0.0)
                / 1000.0,
                "cpu_total_ms": getattr(item, "cpu_time_total", 0.0) / 1000.0,
            }
        )
    rows.sort(key=lambda r: r["cuda_total_ms"], reverse=True)
    return rows[:row_limit]


def _write_csv(path: Path, rows: Iterable[Dict]):
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _run_text_command(cmd: List[str], path: Path):
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        path.write_text(proc.stdout)
    except Exception as exc:
        path.write_text(f"Failed to run {' '.join(cmd)}: {exc}\n")


def _write_environment_snapshot(out_dir: Path, suffix: str, gpu_id: Optional[int]):
    _run_text_command(["nvidia-smi"], out_dir / f"nvidia_smi_{suffix}.txt")
    _run_text_command(
        ["nvidia-smi", "-q", "-d", "CLOCK,TEMPERATURE"],
        out_dir / f"clock_temperature_{suffix}.txt",
    )
    if gpu_id is not None:
        _run_text_command(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv",
                "-i",
                str(gpu_id),
            ],
            out_dir / f"compute_apps_gpu{gpu_id}_{suffix}.csv",
        )


def _start_pmon(out_dir: Path, gpu_id: Optional[int], samples: int):
    if samples <= 0:
        return None
    cmd = ["nvidia-smi", "pmon", "-s", "um", "-c", str(samples)]
    if gpu_id is not None:
        cmd.extend(["-i", str(gpu_id)])
    log_path = out_dir / "pmon_during_run.log"
    f = log_path.open("w")
    try:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        return proc, f
    except Exception as exc:
        f.write(f"Failed to start {' '.join(cmd)}: {exc}\n")
        f.close()
        return None


def _stop_pmon(handle):
    if handle is None:
        return
    proc, f = handle
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    finally:
        f.close()


def _markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _format_top_rows(rows: List[Dict]) -> str:
    return _markdown_table(
        ["Rank", "Profiler row", "Calls", "CUDA total ms", "CUDA avg ms", "Self CUDA ms"],
        [
            [
                str(i + 1),
                row["name"].replace("|", "\\|"),
                str(row["calls"]),
                f"{row['cuda_total_ms']:.3f}",
                f"{row['cuda_avg_ms']:.4f}",
                f"{row['self_cuda_total_ms']:.3f}",
            ]
            for i, row in enumerate(rows)
        ],
    )


def _write_svg_chart(path: Path, summary: List[Dict]):
    metric_groups = [
        ("forward", "Forward"),
        ("forward_backward", "Forward + backward"),
    ]
    colors = {
        "diff": "#4C78A8",
        "gsplat_two_pass": "#F58518",
        "gsplat_padded_single": "#B279A2",
        "gsplat_native19": "#B279A2",
        "gsplat_single": "#54A24B",
    }
    labels = {
        "diff": "diff",
        "gsplat_two_pass": "gsplat 2-pass",
        "gsplat_padded_single": "gsplat 19ch",
        "gsplat_native19": "gsplat 19ch",
        "gsplat_single": "gsplat shared",
    }
    rows_by_mode = {
        mode: [row for row in summary if row["mode"] == mode] for mode, _ in metric_groups
    }
    metric_name = "median_ms" if summary and "median_ms" in summary[0] else "mean_ms"
    max_ms = max(row[metric_name] for row in summary)
    width, height = 760, 360
    margin_left, margin_top, margin_bottom = 72, 44, 68
    plot_width = width - margin_left - 34
    plot_height = height - margin_top - margin_bottom
    group_width = plot_width / len(metric_groups)
    bar_width = 42
    scale = plot_height / (max_ms * 1.15)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="26" font-family="Arial" font-size="18" font-weight="700">Render backend profiling</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - 24}" y2="{height - margin_bottom}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#333"/>',
    ]

    for tick in range(0, 5):
        value = max_ms * 1.15 * tick / 4
        y = height - margin_bottom - value * scale
        svg.append(f'<line x1="{margin_left - 5}" y1="{y:.1f}" x2="{width - 24}" y2="{y:.1f}" stroke="#e7e7e7"/>')
        svg.append(f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{value:.0f}</text>')

    for group_idx, (mode, mode_label) in enumerate(metric_groups):
        rows = rows_by_mode.get(mode, [])
        center = margin_left + group_width * (group_idx + 0.5)
        total_bar_width = len(rows) * bar_width + max(len(rows) - 1, 0) * 16
        x0 = center - total_bar_width / 2
        for row_idx, row in enumerate(rows):
            backend = row["backend"]
            x = x0 + row_idx * (bar_width + 16)
            value = row[metric_name]
            bar_h = value * scale
            y = height - margin_bottom - bar_h
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_h:.1f}" fill="{colors.get(backend, "#999")}"/>')
            svg.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{value:.1f}</text>')
            svg.append(f'<text x="{x + bar_width / 2:.1f}" y="{height - margin_bottom + 18}" text-anchor="middle" font-family="Arial" font-size="10">{labels.get(backend, backend)}</text>')
        svg.append(f'<text x="{center:.1f}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="13" font-weight="700">{mode_label}</text>')

    y_axis = "Median time (ms)" if metric_name == "median_ms" else "Mean time (ms)"
    svg.append(f'<text x="18" y="{margin_top + plot_height / 2:.1f}" transform="rotate(-90 18 {margin_top + plot_height / 2:.1f})" text-anchor="middle" font-family="Arial" font-size="12">{y_axis}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg))


def _write_histogram_svg(
    path: Path,
    sample_rows: List[Dict],
    bins: int = 24,
    modes: Optional[List[str]] = None,
    max_views: Optional[int] = 3,
):
    if not sample_rows:
        return

    allowed_modes = set(modes or [])
    selected_view_indices = sorted({int(row["view_index"]) for row in sample_rows})
    if max_views is not None and max_views > 0:
        selected_view_indices = selected_view_indices[:max_views]
    allowed_views = set(selected_view_indices)

    groups = {}
    for row in sample_rows:
        if allowed_modes and row["mode"] not in allowed_modes:
            continue
        if int(row["view_index"]) not in allowed_views:
            continue
        key = (row["backend"], row["mode"], int(row["view_index"]))
        groups.setdefault(key, []).append(float(row["time_ms"]))
    if not groups:
        return

    panel_w = 360
    panel_h = 180
    cols = 2
    rows = math.ceil(len(groups) / cols)
    width = cols * panel_w
    height = rows * panel_h + 42
    margin_l, margin_r, margin_t, margin_b = 42, 18, 34, 30

    colors = {
        "diff": "#4C78A8",
        "gsplat_two_pass": "#F58518",
        "gsplat_padded_single": "#B279A2",
        "gsplat_native19": "#B279A2",
        "gsplat_single": "#54A24B",
    }

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="24" font-family="Arial" font-size="18" font-weight="700">Timing sample distributions</text>',
    ]

    for idx, ((backend, mode, view_index), values) in enumerate(groups.items()):
        col = idx % cols
        row = idx // cols
        x0 = col * panel_w + margin_l
        y0 = 42 + row * panel_h + margin_t
        plot_w = panel_w - margin_l - margin_r
        plot_h = panel_h - margin_t - margin_b
        min_v = min(values)
        max_v = max(values)
        if max_v <= min_v:
            max_v = min_v + 1.0
        bin_w = (max_v - min_v) / bins
        counts = [0] * bins
        for value in values:
            bin_idx = min(int((value - min_v) / bin_w), bins - 1)
            counts[bin_idx] += 1
        max_count = max(max(counts), 1)

        svg.append(f'<text x="{x0}" y="{y0 - 13}" font-family="Arial" font-size="13" font-weight="700">{backend} / {mode} / view {view_index}</text>')
        svg.append(f'<line x1="{x0}" y1="{y0 + plot_h}" x2="{x0 + plot_w}" y2="{y0 + plot_h}" stroke="#333"/>')
        svg.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_h}" stroke="#333"/>')
        for i, count in enumerate(counts):
            bar_h = count / max_count * plot_h
            x = x0 + i * plot_w / bins
            y = y0 + plot_h - bar_h
            svg.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{plot_w / bins - 1:.1f}" height="{bar_h:.1f}" fill="{colors.get(backend, "#999")}"/>'
            )
        svg.append(f'<text x="{x0}" y="{y0 + plot_h + 18}" font-family="Arial" font-size="10">{min_v:.1f} ms</text>')
        svg.append(f'<text x="{x0 + plot_w}" y="{y0 + plot_h + 18}" text-anchor="end" font-family="Arial" font-size="10">{max_v:.1f} ms</text>')

    svg.append("</svg>")
    path.write_text("\n".join(svg))


def _summarize_by_view(cases: List[Dict], timing_samples_by_key: Dict[str, List[Dict]]) -> List[Dict]:
    rows = []
    for case in cases:
        grouped = {}
        for sample in timing_samples_by_key[case["key"]]:
            grouped.setdefault(sample["view_index"], []).append(sample["time_ms"])
        for view_index, values in sorted(grouped.items()):
            stats = _summarize_times(values)
            rows.append(
                {
                    "backend": case["backend"],
                    "mode": case["mode"],
                    "view_index": view_index,
                    **stats,
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Profile render backends on one scene/view.")
    parser.add_argument("-m", "--model_path", default="output/lerf/figurines")
    parser.add_argument("--source_path", default=None)
    parser.add_argument("--images", default=None)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--view_index", type=int, default=0)
    parser.add_argument("--num_views", type=int, default=1)
    parser.add_argument("--view_stride", type=int, default=1)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["diff", "gsplat_two_pass", "gsplat_native19", "gsplat_single"],
    )
    parser.add_argument("--modes", nargs="+", default=["forward", "forward_backward"])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--profile_iters", type=int, default=5)
    parser.add_argument("--row_limit", type=int, default=20)
    parser.add_argument("--out_dir", default="profiling_results")
    parser.add_argument("--convert_SHs_python", action="store_true")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run a separate torch.profiler pass for kernel breakdown. Timing never uses profiler.",
    )
    parser.add_argument(
        "--memory_iters",
        type=int,
        default=3,
        help="Iterations for the separate peak-memory pass.",
    )
    parser.add_argument("--gpu_id", type=int, default=None, help="Physical GPU id for nvidia-smi logs.")
    parser.add_argument("--monitor_gpu", action="store_true", help="Capture nvidia-smi pmon during the run.")
    parser.add_argument("--pmon_samples", type=int, default=999)
    parser.add_argument("--no_env_snapshot", action="store_true")
    parser.add_argument("--shuffle_seed", type=int, default=0)
    parser.add_argument("--histogram_modes", nargs="+", default=["forward_backward"])
    parser.add_argument("--histogram_max_views", type=int, default=3)
    args = parser.parse_args()

    cfg = _load_cfg_args(args.model_path)
    if args.source_path is not None:
        cfg.source_path = args.source_path
    if args.images is not None:
        cfg.images = args.images
    cfg.eval = False
    if not hasattr(cfg, "object_path"):
        cfg.object_path = "object_mask"

    pipe = _make_pipeline(args.convert_SHs_python)
    opt = _make_optimization_args()

    gaussians = GaussianModel(cfg.sh_degree)
    scene = Scene(cfg, gaussians, load_iteration=args.iteration, shuffle=False)
    gaussians.training_setup(opt)
    for _ in range(gaussians.max_sh_degree):
        gaussians.oneupSHdegree()

    train_views = scene.getTrainCameras()
    selected_views = []
    for order in range(args.num_views):
        view_idx = (args.view_index + order * args.view_stride) % len(train_views)
        selected_views.append(
            {
                "index": view_idx,
                "order": order,
                "view": train_views[view_idx],
            }
        )
    selected_view_indices = [view_info["index"] for view_info in selected_views]
    if len(set(selected_view_indices)) != len(selected_view_indices):
        raise ValueError(
            "Selected views contain duplicates after modulo wrapping: "
            f"{selected_view_indices}. Reduce --num_views or adjust --view_stride."
        )
    bg_color = [1, 1, 1] if cfg.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    backend_fns = {
        "diff": _render_diff,
        "gsplat_two_pass": _render_gsplat_two_pass,
        "gsplat_padded_single": _render_gsplat_padded_single,
        "gsplat_native19": _render_gsplat_padded_single,
        "gsplat_single": _render_gsplat,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gpu_id = args.gpu_id
    if gpu_id is None:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        first_visible = cuda_visible.split(",")[0].strip()
        if first_visible.isdigit():
            gpu_id = int(first_visible)

    if "gsplat_native19" in args.backends:
        _assert_native19_supported()

    if not args.no_env_snapshot:
        _write_environment_snapshot(out_dir, "before", gpu_id)
    pmon_handle = _start_pmon(out_dir, gpu_id, args.pmon_samples) if args.monitor_gpu else None

    cases = []
    for backend in args.backends:
        if backend not in backend_fns:
            raise ValueError(f"Unknown backend: {backend}")
        for mode in args.modes:
            include_backward = mode == "forward_backward"
            if mode not in ("forward", "forward_backward"):
                raise ValueError(f"Unknown mode: {mode}")
            cases.append(
                {
                    "backend": backend,
                    "mode": mode,
                    "key": f"{backend}_{mode}",
                    "render_fn": backend_fns[backend],
                    "include_backward": include_backward,
                }
            )

    try:
        print(
            f"[timing] {len(cases)} interleaved cases, {len(selected_views)} views, "
            f"{args.warmup} warmup rounds/view, {args.iters} measured rounds/view"
        )
        timing_samples_by_key = _event_times_interleaved(
            cases,
            selected_views,
            gaussians,
            pipe,
            background,
            args.warmup,
            args.iters,
            args.shuffle_seed,
        )
    finally:
        _stop_pmon(pmon_handle)
        if not args.no_env_snapshot:
            _write_environment_snapshot(out_dir, "after", gpu_id)

    timing_by_key = {
        key: _summarize_times([sample["time_ms"] for sample in samples])
        for key, samples in timing_samples_by_key.items()
    }
    sample_rows = []
    for case in cases:
        for sample_idx, sample in enumerate(timing_samples_by_key[case["key"]]):
            sample_rows.append(
                {
                    "backend": case["backend"],
                    "mode": case["mode"],
                    "view_index": sample["view_index"],
                    "view_order": sample["view_order"],
                    "sample": sample_idx,
                    "time_ms": sample["time_ms"],
                }
            )
    _write_csv(out_dir / "timing_samples.csv", sample_rows)
    _write_csv(out_dir / "per_view_summary.csv", _summarize_by_view(cases, timing_samples_by_key))
    _write_histogram_svg(
        out_dir / "timing_histograms.svg",
        sample_rows,
        modes=args.histogram_modes,
        max_views=args.histogram_max_views,
    )

    summary = []
    details = {}
    for case in cases:
        print(f"[memory] backend={case['backend']} mode={case['mode']}")
        event_peak_mb = _memory_peak(
            case["render_fn"],
            selected_views,
            gaussians,
            pipe,
            background,
            min(args.warmup, 10),
            args.memory_iters,
            case["include_backward"],
        )
        profiler_peak_mb = None
        if args.profile:
            print(f"[profile] backend={case['backend']} mode={case['mode']}")
            prof, profiler_peak_mb = _profile(
                case["render_fn"],
                selected_views[0]["view"],
                gaussians,
                pipe,
                background,
                min(args.warmup, 10),
                args.profile_iters,
                case["include_backward"],
            )
            rows = _rows_from_prof(prof, args.row_limit)
            _write_csv(out_dir / f"{case['key']}_top_cuda.csv", rows)
            details[case["key"]] = rows

        row = {
            "backend": case["backend"],
            "mode": case["mode"],
            **timing_by_key[case["key"]],
            "event_peak_mb": event_peak_mb,
            "profiler_peak_mb": profiler_peak_mb,
        }
        summary.append(row)

    summary_csv = out_dir / "summary.csv"
    _write_csv(summary_csv, summary)
    _write_svg_chart(out_dir / "summary_chart.svg", summary)

    metadata = {
        "model_path": args.model_path,
        "loaded_iteration": scene.loaded_iter,
        "source_path": cfg.source_path,
        "resolution": getattr(cfg, "resolution", None),
        "view_index": args.view_index,
        "num_views": args.num_views,
        "view_stride": args.view_stride,
        "selected_view_indices": [view_info["index"] for view_info in selected_views],
        "image_width": int(selected_views[0]["view"].image_width),
        "image_height": int(selected_views[0]["view"].image_height),
        "num_gaussians": int(gaussians.get_xyz.shape[0]),
        "warmup": args.warmup,
        "iters": args.iters,
        "samples_per_backend_mode": args.iters * len(selected_views),
        "timing_protocol": "interleaved_cuda_events_no_profiler",
        "memory_iters": args.memory_iters,
        "profile": args.profile,
        "profile_iters": args.profile_iters,
        "gpu_id": gpu_id,
        "monitor_gpu": args.monitor_gpu,
        "shuffle_seed": args.shuffle_seed,
        "histogram_modes": args.histogram_modes,
        "histogram_max_views": args.histogram_max_views,
        **_gsplat_metadata(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    report_rows = [
        [
            row["backend"],
            row["mode"],
            str(row["samples"]),
            f"{row['median_ms']:.3f}",
            f"{row['mean_ms']:.3f}",
            f"{row['std_ms']:.3f}",
            f"{row['min_ms']:.3f}",
            f"{row['p05_ms']:.3f}",
            f"{row['p95_ms']:.3f}",
            f"{row['max_ms']:.3f}",
            f"{row['event_peak_mb']:.1f}",
            "" if row["profiler_peak_mb"] is None else f"{row['profiler_peak_mb']:.1f}",
        ]
        for row in summary
    ]
    report = [
        "# Render Backend Profiling",
        "",
        f"- model_path: `{args.model_path}`",
        f"- iteration: `{scene.loaded_iter}`",
        f"- view_index: `{args.view_index}`",
        f"- selected views: `{[view_info['index'] for view_info in selected_views]}`",
        f"- image: `{int(selected_views[0]['view'].image_width)}x{int(selected_views[0]['view'].image_height)}`",
        f"- gaussians: `{int(gaussians.get_xyz.shape[0])}`",
        f"- timing protocol: interleaved CUDA events, profiler disabled",
        f"- forward mode: training-mode forward, autograd graph enabled",
        f"- shuffle seed: `{args.shuffle_seed}`",
        f"- timed samples per backend/mode: `{args.iters * len(selected_views)}` (`{args.iters}` per view) after `{args.warmup}` warmup rounds",
        f"- environment logs: `nvidia_smi_before.txt`, `clock_temperature_before.txt`, `nvidia_smi_after.txt`, `clock_temperature_after.txt`",
        f"- pmon log: `{'pmon_during_run.log' if args.monitor_gpu else 'disabled'}`",
        f"- profiler pass: `{'enabled' if args.profile else 'disabled'}`",
        f"- chart: [`summary_chart.svg`](summary_chart.svg)",
        f"- histograms: [`timing_histograms.svg`](timing_histograms.svg) for modes `{args.histogram_modes}` and first `{args.histogram_max_views}` selected views",
        "",
        "## Summary",
        "",
        _markdown_table(
            [
                "Backend",
                "Mode",
                "Samples",
                "Median ms",
                "Mean ms",
                "Std ms",
                "Min ms",
                "P05 ms",
                "P95 ms",
                "Max ms",
                "Peak MB",
                "Profiler Peak MB",
            ],
            report_rows,
        ),
        "",
        "## Top CUDA Rows",
    ]

    if details:
        for key, rows in details.items():
            report.extend(["", f"### {key}", "", _format_top_rows(rows)])
    else:
        report.extend(["", "Profiler pass disabled. Use `--profile` for kernel breakdown; do not use profiler timings for backend ranking."])

    report_path = out_dir / "profile_report.md"
    report_path.write_text("\n".join(report))
    print(f"Wrote {report_path}")
    print(_markdown_table(["Backend", "Mode", "Median ms", "Std ms", "Peak MB"], [[r["backend"], r["mode"], f"{r['median_ms']:.3f}", f"{r['std_ms']:.3f}", f"{r['event_peak_mb']:.1f}"] for r in summary]))


if __name__ == "__main__":
    main()
