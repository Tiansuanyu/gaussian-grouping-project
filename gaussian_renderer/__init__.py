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
#   USE_GSPLAT=1 (default) → gsplat backend
#   USE_GSPLAT=0           → original diff-gaussian-rasterization backend

import os
import torch
import math
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

USE_GSPLAT = os.environ.get("USE_GSPLAT", "1") == "1"

if USE_GSPLAT:
    from gsplat import rasterization as gsplat_rasterize
else:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def _render_gsplat(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0, override_color=None):
    """Render using gsplat backend."""

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

    # ---- Colors / SH ----
    if override_color is not None:
        colors = override_color
        sh_degree_to_use = None
    elif pipe.convert_SHs_python:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors = torch.clamp_min(sh2rgb + 0.5, 0.0)
        sh_degree_to_use = None
    else:
        colors = pc.get_features
        sh_degree_to_use = pc.active_sh_degree

    # ---- Pass 1: Render RGB ----
    render_colors, render_alphas, meta = gsplat_rasterize(
        means=means, quats=quats, scales=scales, opacities=opacities,
        colors=colors, viewmats=viewmats, Ks=Ks, width=W, height=H,
        sh_degree=sh_degree_to_use,
        backgrounds=bg_color.unsqueeze(0),
        packed=False,
    )
    rendered_image = render_colors[0].permute(2, 0, 1)

    # ---- Pass 2: Render object identity features ----
    N = means.shape[0]
    obj_feat = pc.get_objects.view(N, -1)

    render_objs, _, _ = gsplat_rasterize(
        means=means, quats=quats, scales=scales, opacities=opacities,
        colors=obj_feat, viewmats=viewmats, Ks=Ks, width=W, height=H,
        sh_degree=None,
        packed=False,
    )
    rendered_objects = render_objs[0].permute(2, 0, 1)

    # ---- Densification info ----
    radii = meta["radii"].squeeze(0)
    radii_scalar = radii.max(dim=-1).values.int()

    screenspace_points = torch.zeros(N, 3, dtype=means.dtype,
                                     device="cuda", requires_grad=True)

    def _capture_grad(grad):
        g = grad.squeeze(0)
        screenspace_points.grad = torch.cat(
            [g, torch.zeros(N, 1, device=g.device, dtype=g.dtype)], dim=-1
        )

    if meta["means2d"].requires_grad:
        meta["means2d"].register_hook(_capture_grad)

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
            sh_objs = pc.get_objects
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
        USE_GSPLAT=1 (default) → gsplat
        USE_GSPLAT=0           → diff-gaussian-rasterization

    Background tensor (bg_color) must be on GPU!
    """
    if USE_GSPLAT:
        return _render_gsplat(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color)
    else:
        return _render_original(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color)