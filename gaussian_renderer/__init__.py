#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# Modified: dual backend support (diff-gaussian-rasterization / gsplat)
#   USE_GSPLAT=1 (default) → gsplat shared-geometry split-raster backend
#   USE_GSPLAT=0           → original diff-gaussian-rasterization backend
#   USE_GSPLAT_NATIVE19=1  → opt-in gsplat native 19-channel single-pass

import inspect
import os
import re
import torch
import math
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

USE_GSPLAT = os.environ.get("USE_GSPLAT", "1") == "1"
USE_GSPLAT_NATIVE19 = os.environ.get("USE_GSPLAT_NATIVE19", "0") == "1"

if USE_GSPLAT_NATIVE19 and not USE_GSPLAT:
    raise RuntimeError("USE_GSPLAT_NATIVE19=1 requires USE_GSPLAT=1.")

if USE_GSPLAT:
    from gsplat.cuda._wrapper import (
        fully_fused_projection,
        isect_offset_encode,
        isect_tiles,
        rasterize_to_pixels,
        spherical_harmonics,
    )
else:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def _native19_supported() -> bool:
    try:
        src = inspect.getsource(rasterize_to_pixels)
    except (NameError, OSError, TypeError):
        return False

    match = re.search(r"if channels not in\s*\((.*?)\):", src, re.DOTALL)
    if match is None:
        return False

    return 19 in {int(value) for value in re.findall(r"\b\d+\b", match.group(1))}


def _assert_native19_supported():
    if _native19_supported():
        return
    raise RuntimeError(
        "USE_GSPLAT_NATIVE19=1 was requested, but the installed gsplat wrapper "
        "does not advertise native 19-channel rasterization. Re-apply the "
        "gsplat native19 patch instead of silently falling back to padded <32u>."
    )


if USE_GSPLAT and USE_GSPLAT_NATIVE19:
    _assert_native19_supported()


def _render_gsplat(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0, override_color=None):
    """Render using gsplat backend with shared projection/tile sorting."""

    # ---- Camera parameters ----
    W = int(viewpoint_camera.image_width)
    H = int(viewpoint_camera.image_height)

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    fx = W / (2.0 * tanfovx)
    fy = H / (2.0 * tanfovy)

    Ks = torch.tensor([
        [fx,  0.0, W / 2.0],
        [0.0, fy,  H / 2.0],
        [0.0, 0.0, 1.0    ],
    ], dtype=torch.float32, device="cuda").unsqueeze(0)

    viewmats = viewpoint_camera.world_view_transform.transpose(0, 1).unsqueeze(0)

    # ---- Gaussian parameters ----
    means = pc.get_xyz
    opacities = pc.get_opacity.squeeze(-1)
    scales = pc.get_scaling
    quats = pc.get_rotation

    if scaling_modifier != 1.0:
        scales = scales * scaling_modifier

    # ---- Projection ----
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

    # ---- Colors / SH ----
    if override_color is not None:
        rgb = override_color.unsqueeze(0)
    elif pipe.convert_SHs_python:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
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

    # ---- Shared geometry, split RGB/object rasterization ----
    N = means.shape[0]
    obj_feat = pc.get_objects.view(N, -1).unsqueeze(0)

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

    if USE_GSPLAT_NATIVE19:
        colors = torch.cat([rgb, obj_feat], dim=-1)
        if colors.shape[-1] != 19:
            raise RuntimeError(
                "USE_GSPLAT_NATIVE19=1 expects RGB plus object features to total "
                f"19 channels, got {colors.shape[-1]}."
            )
        backgrounds = torch.cat(
            [
                bg_color,
                torch.zeros(obj_feat.shape[-1], dtype=bg_color.dtype, device=bg_color.device),
            ],
            dim=0,
        ).unsqueeze(0)
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
    else:
        render_rgb, _ = rasterize_to_pixels(
            means2d,
            conics,
            rgb,
            opacities,
            W,
            H,
            tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=bg_color.unsqueeze(0),
            packed=False,
            absgrad=False,
        )

        render_objs, _ = rasterize_to_pixels(
            means2d,
            conics,
            obj_feat,
            opacities,
            W,
            H,
            tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=None,
            packed=False,
            absgrad=False,
        )
        rendered_image = render_rgb[0].permute(2, 0, 1)
        rendered_objects = render_objs[0].permute(2, 0, 1)

    # ---- Densification info ----
    radii_scalar = radii.squeeze(0).max(dim=-1).values.int()

    screenspace_points = torch.zeros(N, 3, dtype=means.dtype,
                                     device="cuda", requires_grad=True)

    def _capture_grad(grad):
        g = grad.squeeze(0)
        screenspace_points.grad = torch.cat(
            [g, torch.zeros(N, 1, device=g.device, dtype=g.dtype)], dim=-1
        )

    if means2d.requires_grad:
        means2d.register_hook(_capture_grad)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii_scalar > 0,
            "radii": radii_scalar,
            "render_object": rendered_objects}


def _render_original(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0, override_color=None):
    """Render using original diff-gaussian-rasterization backend."""

    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype,
                                          requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    sh_objs = pc.get_objects
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    rendered_image, radii, rendered_objects = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        sh_objs = sh_objs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "render_object": rendered_objects}


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene.

    Backend selected by USE_GSPLAT environment variable:
        USE_GSPLAT=1 (default) → gsplat shared-geometry split raster
        USE_GSPLAT=0           → diff-gaussian-rasterization
        USE_GSPLAT_NATIVE19=1  → gsplat native 19-channel single pass

    Background tensor (bg_color) must be on GPU!
    """
    if USE_GSPLAT:
        return _render_gsplat(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color)
    else:
        return _render_original(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color)
