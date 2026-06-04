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

import torch
import math
from gaustudio_diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None,ret_pts=False,onlyrender = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points = torch.zeros((pc.get_xyz.shape[0], 4), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
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

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    dir_pp = (means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1))
    dir_pp = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    xyz = pc.contract_to_unisphere(means3D.clone().detach(),
                                   torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda'))
    enc_xyz = pc.recolor(xyz)

    bottleneck = pc.first_net(enc_xyz)
    reflection = pc.r_net(bottleneck)
    reflection = reflection.float()
    reflection = torch.sigmoid(reflection)
    ref_feature = reflection

    view_dir = pc.direction_encoding(dir_pp)
    v_net_input = torch.cat([view_dir,bottleneck.float()],dim=-1)
    v_net_hidden = pc.v_net[0](v_net_input)
    v_net_hidden = pc.v_net[1](v_net_hidden)
    illumination = pc.v_net[2](v_net_hidden)

    illumination = torch.sigmoid(illumination)

    input_enhancement =torch.detach(torch.cat([v_net_hidden,illumination],dim=-1))
    enhance_x = pc.enhance_net(input_enhancement)

    mid_gamma = pc.gamma_mlp[0](enhance_x)
    mid_gamma = pc.gamma_mlp[1](mid_gamma)
    mid_gamma = torch.cat([mid_gamma,enhance_x],dim=-1)
    mid_gamma = pc.gamma_mlp[2](mid_gamma)
    coeff_gamma = pc.gamma_mlp[3](mid_gamma)

    mid_alpha = pc.alpha_mlp[0](enhance_x)
    mid_alpha = pc.alpha_mlp[1](mid_alpha)
    mid_alpha = torch.cat([mid_alpha,enhance_x],dim=-1)
    mid_alpha = pc.alpha_mlp[2](mid_alpha)
    coeff_alpha = pc.alpha_mlp[3](mid_alpha)


    rgb = reflection * illumination

    L_sg = torch.detach(illumination)
    R_sg = torch.detach(reflection)
    gamma_base = 2.2
    final_gamma = 1 / (coeff_gamma + gamma_base)

    L_enhanced = (L_sg /(coeff_alpha+0.00001)) ** final_gamma

    rgb_enhanced = L_enhanced * R_sg

    enh_illumination = L_enhanced[...,0:1]

    rendered_image = None
    r_image = None
    v_image = None
    v_enh_image = None
    gamma_image = None
    alpha_image = None

    # color_combine = torch.cat([rgb,reflection,illumination,enh_illumination,rgb_enhanced,coeff_gamma,coeff_alpha],dim=-1)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    if onlyrender:
        enhanced_image, radii, rendered_depth, rendered_median_depth, rendered_final_opacity,gs_w = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=rgb_enhanced,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
    else:
        rendered_image, radii, rendered_depth, rendered_median_depth, rendered_final_opacity,gs_w  = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = rgb,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

        r_image,_,_,_,_,_= rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = reflection,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

        input_v = torch.broadcast_to(illumination,reflection.shape)
        v_image,_,_,_,_,_= rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = input_v,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        v_image = v_image[0,:,:]
        v_image = v_image.unsqueeze(0)

        input_enh_v = torch.broadcast_to(enh_illumination,reflection.shape)
        v_enh_image,_,_,_,_,_ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = input_enh_v,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        v_enh_image = v_enh_image[0,:,:]
        v_enh_image = v_enh_image.unsqueeze(0)

        enhanced_image,_,_,_,_,_ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = rgb_enhanced,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

        gamma_image,_,_,_,_,_= rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = coeff_gamma,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

        input_alpha = torch.broadcast_to(coeff_alpha,reflection.shape)
        alpha_image,_,_,_,_,_= rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = input_alpha,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        alpha_image = alpha_image[0,:,:]
        alpha_image = alpha_image.unsqueeze(0)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "reflection": r_image,
            "ref_feature": ref_feature,
            "illumination": v_image,
            "enh_illumination": v_enh_image,
            "enhanced": enhanced_image,
            "gamma": gamma_image,
            "alpha": alpha_image,
            "rendered_depth": rendered_depth,
            "rendered_median_depth": rendered_median_depth,
            "rendered_final_opacity": rendered_final_opacity,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii}
