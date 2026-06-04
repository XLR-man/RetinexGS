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

import os
import numpy as np
import cv2
import diptest
import torch
from random import randint
from utils.loss_utils import l1_loss,ssim,L_TV_abs,Exp_loss,L_color_constancy,L_spatial,ltv_loss,ssim_reg
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from render import render_sets
import math
from collections import defaultdict
import csv

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,prune_sched):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset,opt,pipe)
    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)
        gaussians.update_net_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, r_image, v_image,v_enh_image,enhanced_image,gamma_image,alpha_image, viewspace_point_tensor, visibility_filter, radii = (render_pkg["render"],render_pkg["reflection"],render_pkg["illumination"],render_pkg["enh_illumination"],render_pkg["enhanced"],
                                                                                                                render_pkg["gamma"],render_pkg["alpha"],render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"])
        rendered_depth = render_pkg["rendered_depth"][0]
        rendered_median_depth = render_pkg["rendered_median_depth"][0]
        rendered_median_weight = render_pkg["rendered_median_depth"][1]
        rendered_final_opacity = render_pkg["rendered_final_opacity"][0]
        surface_mask = rendered_final_opacity > 0.5

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        loss1 = l1_loss(r_image*v_image, gt_image)

        if opt.reg_ssim:
            loss_ssim = 1.0 - ssim_reg(r_image*v_image, gt_image)
        else:
            loss_ssim = 1.0 - ssim(r_image*v_image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * loss1 + opt.lambda_dssim * loss_ssim

        if opt.lambda_illu:
            max_values, _ = torch.max(gt_image, dim=0)
            max_values = max_values.unsqueeze(0)
            loss_illu = torch.abs(v_image-max_values**opt.v_gamma).mean()
            loss += opt.lambda_illu * loss_illu

        if opt.lambda_ref2:
            loss_ref2 = L_TV_abs()(r_image)
            loss += opt.lambda_ref2 * loss_ref2

        if opt.lambda_ref3 and iteration > opt.ref3_from:
            max_values, _ = torch.max(gt_image, dim=0)

            img = max_values  # H x W
            H, W = img.shape
            # Scale to 0-255 and convert to integers
            img_int = torch.clamp(torch.round(img * 255), 0, 255).int()

            # Compute histogram
            hist = torch.histc(img_int.float(), bins=256, min=0, max=255)

            # Compute CDF and normalize
            cdf = torch.cumsum(hist, dim=0)
            cdf_min = cdf.min()
            cdf_normalized = (cdf - cdf_min) / (cdf.max() - cdf_min + 1e-8)  # Avoid division by zero

            # Map pixels using CDF
            img_eq_flat = cdf_normalized[img_int.flatten().long()]
            img_eq = img_eq_flat.view(H, W)

            # Rescale to [0, 1]
            img_eq = img_eq / 255.0
            img_eq = img_eq.unsqueeze(0)  # Add channel dim

            max_values2, _ = torch.max(r_image, dim=0)
            max_values2 = max_values2.unsqueeze(0)
            loss_ref3 = torch.mean(torch.abs(max_values2 - (img_eq ** opt.he_exp)))
            loss += opt.lambda_ref3 * loss_ref3

        # depth_distortion loss
        distortion_loss = 0.0
        if opt.lambda_depth_distortion > 0. and opt.depth_from_iter < iteration < opt.depth_until_iter:
            distortion_loss = rendered_median_weight * torch.abs(rendered_depth - rendered_final_opacity.detach() * rendered_median_depth)
            distortion_loss = distortion_loss.mean()
            loss += distortion_loss * opt.lambda_depth_distortion

        if iteration > opt.enh_iteration:
            if opt.lambda_exp:
                loss_exp = Exp_loss(mean_val=opt.exposure)(r_image*v_enh_image)
                loss += opt.lambda_exp * loss_exp
            if opt.lambda_cc:
                loss_cc = L_color_constancy()(r_image*v_enh_image)
                loss += opt.lambda_cc * loss_cc
            if opt.lambda_spa:
                loss_spa = L_spatial(contrast=1)(gt_image,r_image*v_enh_image)
                loss += opt.lambda_spa * loss_spa
            if opt.issm:
                temp_v = torch.detach(v_image)
                gamma_ref = torch.broadcast_to(temp_v,gamma_image.shape)
                loss_gamma_tv = ltv_loss(gamma_image,gamma_ref)
                loss_alpha_tv = ltv_loss(alpha_image,temp_v)
                loss += opt.lambda_gamma_tv * loss_gamma_tv
                loss += opt.lambda_alpha_tv * loss_alpha_tv
            loss_gamma =(gamma_image ** 2).mean()
            loss += opt.lambda_gamma * loss_gamma

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                # progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.set_postfix({"loss": f"{loss.item():.{7}f}","points":gaussians.get_xyz.shape[0]})
                progress_bar.update(10)

            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, loss1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                testing_iterations, scene, render, (pipe, background))

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    if opt.sizeNone:
                        size_threshold = None
                    else:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_grad_abs_threshold,opt.min_opacity, scene.cameras_extent, size_threshold,opt.isAbsGS)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration == opt.iterations:
                print("Final numbers of gaussians:" ,gaussians.get_xyz.shape[0])

            if iteration == 15000:
                render_sets(dataset,iteration,pipe,False,False)

            # Optimizer step
            if iteration < opt.iterations:
                if opt.clip_grad:
                    torch.nn.utils.clip_grad_norm_(gaussians.optimizer_net.param_groups[0]['params'], max_norm=opt.clip_grad_value)
                    torch.nn.utils.clip_grad_value_(gaussians.optimizer_net.param_groups[0]['params'], clip_value=opt.clip_grad_value)
                    # 对梯度进行 nan_to_num 处理
                    for param in gaussians.optimizer_net.param_groups[0]['params']:
                        if param.grad is not None:
                            param.grad = torch.nan_to_num(param.grad)
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                gaussians.optimizer_net.step()
                gaussians.optimizer_net.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args,opt=None, pipe=None):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    with open(os.path.join(args.model_path, "cfg_args_opt"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(opt))))

    with open(os.path.join(args.model_path, "cfg_args_pipe"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(pipe))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, loss_rec, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/loss_rec', loss_rec.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: Loss_rec {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - loss_rec', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            # opacity_values = scene.gaussians.get_opacity
            # if opacity_values.numel() > 0:
            #     tb_writer.add_histogram("scene/opacity_histogram", opacity_values, iteration)
            # else:
            #     print(f"Warning: No opacity values at iteration {iteration}")
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 15_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 15_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--prune_sched", nargs="+", type=int, default=[])
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,args.prune_sched)

    # All done
    print("\nTraining complete.")
