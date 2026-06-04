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
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import math
import torchvision

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

class Exp_loss(nn.Module):

    def __init__(self,patch_size=16,mean_val=0.4):
        super(Exp_loss, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
    def forward(self, x ):
        x = x.unsqueeze(0)
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)

        d = torch.mean((mean- self.mean_val) ** 2)
        return d

class L_spatial(nn.Module):

    def __init__(self,contrast):
        super(L_spatial, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)
        self.contrast = contrast
    def forward(self, org , enhance ):
        org = org.unsqueeze(0)
        enhance = enhance.unsqueeze(0)
        b,c,h,w = org.shape

        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(self.contrast * D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(self.contrast * D_org_right - D_enhance_right,2)
        D_up = torch.pow(self.contrast * D_org_up - D_enhance_up,2)
        D_down = torch.pow(self.contrast * D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)
        loss = torch.mean(E)
        return loss

class L_color_constancy(nn.Module):

    def __init__(self):
        super(L_color_constancy, self).__init__()
    def forward(self, x ):
        mean_rgb = torch.mean(x,[1,2],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=0)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        loss = torch.mean(k)
        return loss

class L_TV_abs(nn.Module):
    def __init__(self,):
        super(L_TV_abs,self).__init__()

    def forward(self,x):

        h_x = x.size()[1]
        w_x = x.size()[2]
        count_h =  (x.size()[1]-1) * x.size()[2]
        count_w = x.size()[1] * (x.size()[2] - 1)
        h_tv = torch.abs(x[:,1:,:]-x[:,:h_x-1,:]).sum()
        w_tv = torch.abs(x[:,:,1:]-x[:,:,:w_x-1]).sum()
        return h_tv/count_h+w_tv/count_w

def ltv_loss(L_e, L,  beta=1.5, alpha=2, eps=1e-4):

    # L = torch.log(L + eps)
    h_x = L.size()[1]
    w_x = L.size()[2]

    dx_L = (L[:, 1:, :] - L[:, :h_x - 1, :])
    dy_L = (L[:,:,1:]-L[:,:,:w_x-1])
    dx_Le = (L_e[:, 1:, :] - L_e[:, :h_x - 1, :])
    dy_Le = (L_e[:, :, 1:] - L_e[:, :, :w_x - 1])

    count_h = L[:, 1:, :].size()[0]*L[ :, 1:, :].size()[1]*L[ :, 1:, :].size()[2]
    count_w = L[:, :, 1:].size()[0]*L[ :, :, 1:].size()[1]*L[ :, :, 1:].size()[2]

    ltv_x = ((beta * dx_Le ** 2) / (dx_L ** alpha + eps)).sum() / count_h
    ltv_y = ((beta * dy_Le ** 2) / (dy_L ** alpha + eps)).sum() / count_w

    return (ltv_x + ltv_y) / 2

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def ssim_reg(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    scaling_grad = 1. / (1e-3 + torch.detach(img1))
    img1 = scaling_grad * img1
    img2 = scaling_grad * img2
    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)