#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM

Modified by 
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@Time: 2020/3/9 9:32 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv2_m = nn.Conv1d(64, args.emb_dims, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn2_m = nn.BatchNorm1d(args.emb_dims)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

        self.pool1 = Pool_FPS(256, 64, 0.2)

    def forward(self, x):
        xyz = copy.deepcopy(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x_t1 = F.relu(self.bn2_m(self.conv2_m(x)))

        node1, node_features_1 = self.pool1(xyz, x)
        node_features_agg = aggregate(xyz, node1, x, 10)
        x = torch.cat((node_features_1, node_features_agg), dim=1)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x_t2 = F.relu(self.bn5(self.conv5(x)))

        x = torch.cat((F.adaptive_max_pool1d(x_t1, 1).squeeze(), F.adaptive_max_pool1d(x_t2, 1).squeeze()), dim=1)
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x, node1


class DGCNN_cls(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN_cls, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn2_m = nn.BatchNorm1d(args.emb_dims)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2_m = nn.Sequential(nn.Conv1d(64 * 2, args.emb_dims, kernel_size=1, bias=False),
                                     self.bn2_m,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(256*2, 256, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(256*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(256*2, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.pool1 = Pool_FPS(args.num_points//4, 128, 0.2)
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        xyz = copy.deepcopy(x)

        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        # pool(sample and aggregate)
        x_t1_ = torch.cat((x1, x2), dim=1)
        x_t1 = self.conv2_m(x_t1_)
        node1, node_features_1 = self.pool1(xyz, x_t1_)
        node_features_agg = aggregate(xyz, node1, x_t1_, self.k)
        x = torch.cat((node_features_1, node_features_agg), dim=1)

        x = get_graph_feature(x, k=self.k//2)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k//2)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat([x3, x4], dim=1)
        x_t2 = self.conv5(x)

        x1 = F.adaptive_max_pool1d(x_t1, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_max_pool1d(x_t2, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*3)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)                                             # (batch_size, 256) -> (batch_size, output_channels)
        
        return x, node1

class DGCNN_scan(nn.Module):
    def __init__(self, args, output_channels=15, seg_num_all=2):
        super(DGCNN_scan, self).__init__()
        self.args = args
        self.k = args.k
        self.seg_num_all = 2

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn2_m = nn.BatchNorm1d(args.emb_dims)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.bn6 = nn.BatchNorm1d(128)
        self.bn7 = nn.BatchNorm1d(128)
        self.bn8 = nn.BatchNorm1d(128)
        self.bn9 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2_m = nn.Sequential(nn.Conv1d(64 * 2, args.emb_dims, kernel_size=1, bias=False),
                                     self.bn2_m,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(256 * 2, 256, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(256 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(256 * 2, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(2*args.emb_dims + 256, 128, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(128 + 256, 128, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(128 + 64, 128, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv9 = nn.Sequential(nn.Conv1d(128 + 64, 128, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp3 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)

        self.pool1 = Pool_FPS(args.num_points // 4, 128, 0.2)
        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        xyz = copy.deepcopy(x)

        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        # pool(sample and aggregate)
        x_t1_ = torch.cat((x1, x2), dim=1)
        x_t1 = self.conv2_m(x_t1_)
        node1, node_features_1 = self.pool1(xyz, x_t1_)
        node_features_agg = aggregate(xyz, node1, x_t1_, self.k)
        x = torch.cat((node_features_1, node_features_agg), dim=1)

        x = get_graph_feature(x, k=self.k // 2)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k // 2)  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat([x3, x4], dim=1)
        x_t2 = self.conv5(x)

        x_t1 = F.adaptive_max_pool1d(x_t1, 1).view(batch_size, -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x_t2 = F.adaptive_max_pool1d(x_t2, 1).view(batch_size, -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        vector = torch.cat((x_t1, x_t2), 1)  # (batch_size, emb_dims*2)

        ## classification
        x = F.leaky_relu(self.bn6(self.linear1(vector)), negative_slope=0.2)  # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        logits_cls = self.linear3(x)  # (batch_size, 256) -> (batch_size, output_channels)

        ## segmentation
        x = vector.unsqueeze(-1).repeat(1, 1, x4.shape[-1])  # (batch_size, 64, num_points//4)
        x = torch.cat((x, x4), dim=1)  # (batch_size, 256+64, num_points//4)
        x = self.conv6(x)  # (batch_size, 256+64, num_points//4) -> (batch_size, 256, num_points//4)

        x = torch.cat((x, x3), dim=1)  # (batch_size, 256+64, num_points//4)
        x = self.conv7(x)  # (batch_size, 256+64, num_points//4) -> (batch_size, 256, num_points//4)

        x = unpool(node1, xyz, x)
        # print('shape of x2: {}'.format(x2.shape))
        # print('shape of x: {}'.format(x.shape))
        x = torch.cat((x, x2), dim=1)  # (batch_size, 256+64, num_points)
        x = self.conv8(x)  # (batch_size, 256+64, num_points) -> (batch_size, 256, num_points)

        x = torch.cat((x, x1), dim=1)  # (batch_size, 256+64, num_points)
        x = self.conv9(x)  # (batch_size, 256+64, num_points) -> (batch_size, 128, num_points)
        x = self.dp3(x)

        logits_seg = self.conv10(x)  # (batch_size, 128, num_points) -> (batch_size, seg_num_all, num_points)

        return logits_cls, logits_seg, node1


class Transform_Net(nn.Module):
    def __init__(self, args):
        super(Transform_Net, self).__init__()
        self.args = args
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x

class Pool_FPS(nn.Module):
    def __init__(self, k, in_dim, p):
        '''
        k: num of kpoints
        in_dim: feature channels
        p: dropout rate
        '''
        super(Pool_FPS, self).__init__()
        self.k = k

    def forward(self, xyz, feature):
        """
        xyz: B, 3, N
        feature: B, C, N
        """
        idx = pointnet2_utils.furthest_point_sample(xyz.permute(0, 2, 1).contiguous(), self.k)
        idx = idx.type('torch.cuda.LongTensor')
        xyz_idx = idx.unsqueeze(2).repeat(1, 1, xyz.shape[1])
        xyz_idx = xyz_idx.permute(0, 2, 1)
        node_static = xyz.gather(2, xyz_idx)  # Bx3xnpoint
        feature_idx = idx.unsqueeze(2).repeat(1, 1, feature.shape[1])
        feature_idx = feature_idx.permute(0, 2, 1)
        node_feature = feature.gather(2, feature_idx)  # Bx3xnpoint
        return node_static, node_feature


def unpool(xyz, unknown_xyz, features):
    """
    idea from three interpolate of PointNet2
    input:
        param xyz: input data Bx3xM tensor
        param unknown_xyz: input node data Bx3xN tensor
        param features: input feature BxCxM tensor
    return:
        unknown_features: BxCxN
    """
    M = xyz.size(-1)
    xyz_expanded = xyz.unsqueeze(2).expand(unknown_xyz.size(0), unknown_xyz.size(1), unknown_xyz.size(2), M) # B, 3, N, M
    unknown_xyz_expanded = unknown_xyz.unsqueeze(3).expand_as(xyz_expanded)

    # calcuate difference between x and each node
    diff = unknown_xyz_expanded - xyz_expanded  # Bx3xNxM
    diff_norm = (diff ** 2).sum(dim=1)  # BxNxM
    nn_dist, nn_idx = torch.topk(diff_norm, k=3, dim=-1, largest=False, sorted=False)  # BxNx3
    weight = F.softmax(nn_dist, dim=-1) # BxNx3

    # gather known features and aggregate unknown features
    nn_idx_fold = nn_idx.reshape(nn_idx.shape[0], -1)  # Bx3N
    nn_idx_fold = nn_idx_fold.unsqueeze(1).expand(
        features.size(0), features.size(1), nn_idx_fold.size(-1)) # B x C x 3N
    feature_grouped = features.gather(dim=2, index=nn_idx_fold)  # B x C x 3N
    feature_unfold = feature_grouped.reshape(features.shape[0], features.shape[1],
                                             nn_idx.shape[1], nn_idx.shape[2])  # B x C x N x 3
    unknown_features = torch.sum(weight.unsqueeze(1) * feature_unfold, dim=-1)
    return unknown_features


def aggregate(xyz, node, features, k):
    """
    :param xyz: input data Bx3xN tensor
    :param node: input node data Bx3xM tensor
    :param features: input feature BxCxN tensor
    :param k: number of neighbors
    return:
    node_features: BxCxM
    """
    M = node.size(-1)
    node = node.unsqueeze(2).expand(xyz.size(0), xyz.size(1), xyz.size(2), M)
    x_expanded = xyz.unsqueeze(3).expand_as(node)

    # calcuate difference between x and each node
    diff = x_expanded - node  # BxCxNxnode_num
    diff_norm = (diff ** 2).sum(dim=1)  # BxNxnode_num

    # find the nearest neighbor
    _, nn_idx = torch.topk(diff_norm, k=k, dim=1, largest=False, sorted=False)  # BxkxM
    nn_idx_fold = nn_idx.reshape(nn_idx.shape[0], -1) #BxkM
    nn_idx_fold = nn_idx_fold.unsqueeze(1).expand(
                   features.size(0), features.size(1), nn_idx_fold.size(-1))
    # B x C x kM
    feature_grouped = features.gather(dim=2, index=nn_idx_fold) # B x C x kM
    feature_unfold = feature_grouped.reshape(features.shape[0], features.shape[1],
                                             nn_idx.shape[1], nn_idx.shape[2]) # B x C x k x M
    feature_max = torch.max(feature_unfold, dim=2)[0]  # BxCxM
    return feature_max



class DGCNN_partseg(nn.Module):
    def __init__(self, args, seg_num_all):
        super(DGCNN_partseg, self).__init__()
        self.args = args
        self.seg_num_all = seg_num_all
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)
        self.bn6_m = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(256)
        self.bn11 = nn.BatchNorm1d(256)
        self.bn12 = nn.BatchNorm1d(128)

        self.pool1 = Pool(self.args.num_points//4, 64, 0.2)
        self.pool2 = Pool(self.args.num_points//16, 64, 0.2)
        self.pool3 = Pool(self.args.num_points//64, 64, 0.2)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6_m = nn.Sequential(nn.Conv1d(64*4, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6_m,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(args.emb_dims+64, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv9 = nn.Sequential(nn.Conv1d(256+64, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv10 = nn.Sequential(nn.Conv1d(256+64, 256, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Sequential(nn.Conv1d(256+64, 256, kernel_size=1, bias=False),
                                    self.bn11,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv12 = nn.Sequential(nn.Conv1d(256+64, 128, kernel_size=1, bias=False),
                                    self.bn12,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.dp = nn.Dropout(p=args.dropout)
        self.conv13 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)
        

    def forward(self, x, l):
        batch_size = x.size(0)

        xyz = copy.deepcopy(x)

        x = get_graph_feature(x, k=self.k)                 # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                                  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                                  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]               # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        node1, node_feature_1, node1_static = self.pool1(xyz, x1)        # (batch_size, 64, num_points) -> (batch_size, 64, num_points//4) 512

        x = get_graph_feature(node_feature_1, k=self.k//2) # (batch_size, 64, num_points//4) -> (batch_size, 64*2, num_points//4, k//2)
        x = self.conv3(x)                                  # (batch_size, 64*2, num_points//4, k//2) -> (batch_size, 64, num_points//4, k//2)
        x = self.conv4(x)                                  # (batch_size, 64, num_points//4, k//2) -> (batch_size, 64, num_points//4, k//2)
        x2 = x.max(dim=-1, keepdim=False)[0]               # (batch_size, 64, num_points//4, k//2) -> (batch_size, 64, num_points//4)

        node2, node_feature_2, node2_static = self.pool2(node1_static, x2)      # (batch_size, 64, num_points//4) -> (batch_size, 64, num_points//16) 128

        x = get_graph_feature(node_feature_2, k=self.k//4) # (batch_size, 64, num_points//16) -> (batch_size, 64*2, num_points//16, k//4)
        x = self.conv5(x)                                  # (batch_size, 64*2, num_points//16, k//4) -> (batch_size, 64, num_points//16, k//4)
        x3 = x.max(dim=-1, keepdim=False)[0]               # (batch_size, 64, num_points//16, k//4) -> (batch_size, 64, num_points//16)

        node3, node_feature_3, node3_static = self.pool3(node2_static, x3)      # (batch_size, 64, num_points//16) -> (batch_size, 64, num_points//64) 32

        x = get_graph_feature(node_feature_3, k=self.k//8)  # (batch_size, 64, num_points//64) -> (batch_size, 64*2, num_points//64, k//8)
        x = self.conv6(x)                                   # (batch_size, 64*2, num_points//64, k//8) -> (batch_size, 64, num_points//64, k//8)
        x4 = x.max(dim=-1, keepdim=False)[0]                # (batch_size, 64, num_points//64, k//8) -> (batch_size, 64, num_points//64)

        x1_t = x1.max(dim=-1, keepdim=True)[0]
        x2_t = x2.max(dim=-1, keepdim=True)[0]
        x3_t = x3.max(dim=-1, keepdim=True)[0]
        x4_t = x4.max(dim=-1, keepdim=True)[0]
        x = torch.cat((x1_t, x2_t, x3_t, x4_t), dim=1)
        x = self.conv6_m(x)                                 # (batch_size, 64*4, 1) -> (batch_size, 1024, 1)

        l = l.view(batch_size, -1, 1)                       # (batch_size, num_categoties, 1)
        l = self.conv7(l)                                   # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        x = torch.cat((x, l), dim=1)                        # (batch_size, 1088, 1)
        x = self.conv8(x)                                   # (batch_size, 1088, 1) -> (batch_size, 256, 1)

        x = x.repeat(1, 1, x4.shape[-1])                    # (batch_size, 64, num_points//64)
        x = torch.cat((x, x4), dim=1)                       # (batch_size, 256+64, num_points//64)
        x = self.conv9(x)                                   # (batch_size, 256+64, num_points//64) -> (batch_size, 256, num_points//64)

        x = unpool(node3_static, node2_static, x)
        x = torch.cat((x, x3), dim=1)                       # (batch_size, 256+64, num_points//16)
        x = self.conv10(x)                                  # (batch_size, 256+64, num_points//16) -> (batch_size, 256, num_points//16)

        x = unpool(node2_static, node1_static, x)
        x = torch.cat((x, x2), dim=1)                       # (batch_size, 256+64, num_points//4)
        x = self.conv11(x)                                  # (batch_size, 256+64, num_points//4) -> (batch_size, 256, num_points//4)

        x = unpool(node1_static, xyz, x)
        x = torch.cat((x, x1), dim=1)                       # (batch_size, 256+64, num_points)
        x = self.conv12(x)                                  # (batch_size, 256+64, num_points) -> (batch_size, 128, num_points)
        x = self.dp(x)

        x = self.conv13(x)                                  # (batch_size, 128, num_points) -> (batch_size, seg_num_all, num_points)
        
        return x, node1, node2, node3, node1_static, node2_static


class DGCNN_semseg(nn.Module):
    def __init__(self, args):
        super(DGCNN_semseg, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.pool1 = Pool(self.args.num_points // 16, 64, 0.2)
        # self.pool2 = Pool(self.args.num_points // 16, 64, 0.2)
        # self.pool3 = Pool(self.args.num_points // 64, 64, 0.2)

        self.conv1 = nn.Sequential(nn.Conv2d(18, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6_m = nn.Sequential(nn.Conv1d(64, args.emb_dims, kernel_size=1, bias=False),
                                     self.bn6,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv8_m = nn.Sequential(nn.Conv1d(args.emb_dims, 256, kernel_size=1, bias=False),
                                     self.bn8,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9_m = nn.Sequential(nn.Conv1d(256 + 64, 256, kernel_size=1, bias=False),
                                     self.bn9,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv10_m = nn.Sequential(nn.Conv1d(256 + 64, 128, kernel_size=1, bias=False),
                                      self.bn10,
                                      nn.LeakyReLU(negative_slope=0.2))
        self.dp3 = nn.Dropout(p=args.dropout)
        self.conv11_m = nn.Conv1d(128 + 64, 13, kernel_size=1, bias=False)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        xyz = x[:, :3, :]                  # use normalized xyz

        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        # node1, node_feature_1 = self.pool1(xyz, x1)  # (batch_size, 64, num_points) -> (batch_size, 64, num_points//4) 512

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points//4) -> (batch_size, 64*2, num_points//4, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points//4, k) -> (batch_size, 64, num_points//4, k)
        x = self.conv4(x)  # (batch_size, 64, num_points//4, k) -> (batch_size, 64, num_points//4, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points//4, k) -> (batch_size, 64, num_points//4)

        node2, node_feature_2 = self.pool1(xyz, x2)  # (batch_size, 64, num_points//4) -> (batch_size, 64, num_points//16) 128

        x = get_graph_feature(node_feature_2, k=self.k//2)  # (batch_size, 64, num_points//16) -> (batch_size, 64*2, num_points//16, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points//16, k) -> (batch_size, 64, num_points//16, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points//16, k) -> (batch_size, 64, num_points//16)

        # x = torch.cat((x1, x2, x3), dim=1)               # (batch_size, 64*3, num_points)
        # node3, node_feature_3 = self.pool3(node2, x3)  # (batch_size, 64, num_points//16) -> (batch_size, 64, num_points//64) 32

        x4 = self.conv6_m(x3)  # (batch_size, 64, num_points//64) -> (batch_size, emb_dims, num_points//64)
        # x = x4.max(dim=-1, keepdim=True)[0]              # (batch_size, emb_dims, num_points//64) -> (batch_size, emb_dims, 1)

        x = self.conv8_m(x4)  # (batch_size, 1024, num_points//64) -> (batch_size, 256, num_points//64)
        x = self.dp1(x)

        # x = unpool(node3, node2, x)
        x = torch.cat((x, x3), dim=1)  # (batch_size, 256+64, num_points//16)
        x = self.conv9_m(x)  # (batch_size, 256+64, num_points//16) -> (batch_size, 256, num_points//16)
        x = self.dp2(x)

        x = unpool(node2, xyz, x)
        x = torch.cat((x, x2), dim=1)  # (batch_size, 256+64, num_points//4)
        x = self.conv10_m(x)  # (batch_size, 256+64, num_points//4) -> (batch_size, 128, num_points//4)
        x = self.dp3(x)

        # x = unpool(node1, xyz, x)
        x = torch.cat((x, x1), dim=1)  # (batch_size, 128+64, num_points)
        x = self.conv11_m(x)  # (batch_size, 128+64, num_points) -> (batch_size, seg_num_all, num_points)

        return x, node2
