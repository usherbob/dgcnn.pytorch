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
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN_cls(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN_cls, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)                       # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)                                             # (batch_size, 256) -> (batch_size, output_channels)
        
        return x


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

class Pool(nn.Module):
    def __init__(self, k, in_dim, p):
        '''
        k: num of kpoints
        in_dim: feature channels
        p: dropout rate
        '''
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        # principal component
        self.proj = nn.Conv1d(in_dim, in_dim, 1) # single_head 8
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, xyz, feature):
        Z = self.drop(feature)
        # adaptive modeling of downsampling
        vector = torch.max(self.proj(Z).squeeze(), dim=-1, keepdim=True)[0] # bs, C, 1
        weights = torch.sum(feature * vector, dim=1) # bs, C, n
        scores = self.sigmoid(weights) # batchsize, 8, n
        values, idx = torch.topk(scores, self.k, dim=-1) # bs, 8, k//8

        xyz_idx = idx.unsqueeze(2).repeat(1, 1, xyz.shape[1])
        xyz_idx = xyz_idx.permute(0, 2, 1)
        node = xyz.gather(2, xyz_idx)  # Bx3xnpoint
        feature_idx = idx.unsqueeze(2).repeat(1, 1, feature.shape[1])
        feature_idx = feature_idx.permute(0, 2, 1)
        node_feature = feature.gather(2, feature_idx)  # Bx3xnpoint
        ## especially important
        values = torch.unsqueeze(values, 1)
        assert values.shape == (feature.shape[0], 1, self.k), "values shape error"
        node_feature = torch.mul(node_feature, values)
        node = torch.mul(node, values)
        return node, node_feature

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
        self.transform_net = Transform_Net(args)
        
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

        self.pool1 = Pool(self.args.num_points//4, 64, 0.2)
        self.pool2 = Pool(self.args.num_points//16, 64, 0.2)
        self.pool3 = Pool(self.args.num_points//64, 64, 0.2)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6_m = nn.Sequential(nn.Conv1d(64*2, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8_m = nn.Sequential(nn.Conv1d(args.emb_dims+64, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9_m = nn.Sequential(nn.Conv1d(256+64, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv10_m = nn.Sequential(nn.Conv1d(256+64, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp3 = nn.Dropout(p=args.dropout)
        self.conv11_m = nn.Conv1d(128+64, self.seg_num_all, kernel_size=1, bias=False)
        

    def forward(self, x, l):
        batch_size = x.size(0)
        num_points = x.size(2)

        x0 = get_graph_feature(x, k=self.k)                # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)                         # (batch_size, 3, 3)
        x = x.transpose(2, 1)                              # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)                                # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        xyz = x.transpose(2, 1)                            # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = get_graph_feature(xyz, k=self.k)               # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                                  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                                  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]               # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        node1, node_feature_1 = self.pool1(xyz, x1)        # (batch_size, 64, num_points) -> (batch_size, 64, num_points//4) 512
        agg_feature_1 = aggregate(xyz, node1, x1, self.k)
        node_feature_1 = torch.cat((node_feature_1, agg_feature_1), dim=1)

        x = get_graph_feature(node_feature_1, k=self.k)    # (batch_size, 64, num_points//4) -> (batch_size, 64*2, num_points//4, k)
        x = self.conv3(x)                                  # (batch_size, 64*2, num_points//4, k) -> (batch_size, 64, num_points//4, k)
        x = self.conv4(x)                                  # (batch_size, 64, num_points//4, k) -> (batch_size, 64, num_points//4, k)
        x2 = x.max(dim=-1, keepdim=False)[0]               # (batch_size, 64, num_points//4, k) -> (batch_size, 64, num_points//4)

        node2, node_feature_2 = self.pool2(node1, x2)      # (batch_size, 64, num_points//4) -> (batch_size, 64, num_points//16) 128
        agg_feature_2 = aggregate(node1, node2, x2, self.k)
        node_feature_2 = torch.cat((node_feature_2, agg_feature_2), dim=1)

        x = get_graph_feature(node_feature_2, k=self.k)    # (batch_size, 64, num_points//16) -> (batch_size, 64*2, num_points//16, k)
        x = self.conv5(x)                                  # (batch_size, 64*2, num_points//16, k) -> (batch_size, 64, num_points//16, k)
        x3 = x.max(dim=-1, keepdim=False)[0]               # (batch_size, 64, num_points//16, k) -> (batch_size, 64, num_points//16)

        # x = torch.cat((x1, x2, x3), dim=1)               # (batch_size, 64*3, num_points)
        node3, node_feature_3 = self.pool3(node2, x3)      # (batch_size, 64, num_points//16) -> (batch_size, 64, num_points//64) 32
        agg_feature_3 = aggregate(node2, node3, x3, self.k)
        node_feature_3 = torch.cat((node_feature_3, agg_feature_3), dim=1)

        x4 = self.conv6_m(node_feature_3)                  # (batch_size, 64, num_points//64) -> (batch_size, emb_dims, num_points//64)
        # x = x4.max(dim=-1, keepdim=True)[0]              # (batch_size, emb_dims, num_points//64) -> (batch_size, emb_dims, 1)

        l = l.view(batch_size, -1, 1)                      # (batch_size, num_categoties, 1)
        l = self.conv7(l)                                  # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        l = l.repeat(1, 1, x4.shape[-1])                   # (batch_size, 64, num_points//64)
        x = torch.cat((x4, l), dim=1)                      # (batch_size, 1088, num_points//64)
        x = self.conv8_m(x)                                # (batch_size, 1088, num_points//64) -> (batch_size, 256, num_points//64)
        x = self.dp1(x)


        x = x.repeat(1, 1, 4)
        x = torch.cat((x, x3), dim=1)                      # (batch_size, 256+64, num_points//16)
        x = self.conv9_m(x)                                # (batch_size, 256+64, num_points//16) -> (batch_size, 256, num_points//16)
        x = self.dp2(x)

        x = x.repeat(1, 1, 4)
        x = torch.cat((x, x2), dim=1)                      # (batch_size, 256+64, num_points//4)
        x = self.conv10_m(x)                               # (batch_size, 256+64, num_points//4) -> (batch_size, 128, num_points//4)
        x = self.dp3(x)

        x = x.repeat(1, 1, 4)
        x = torch.cat((x, x1), dim=1)                      # (batch_size, 128+64, num_points)
        x = self.conv11_m(x)                               # (batch_size, 128+64, num_points) -> (batch_size, seg_num_all, num_points)
        
        return x, xyz, node1, node2, node3


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

        self.pool1 = Pool(self.args.num_points // 4, 64, 0.2)
        self.pool2 = Pool(self.args.num_points // 16, 64, 0.2)
        self.pool3 = Pool(self.args.num_points // 64, 64, 0.2)

        self.conv1 = nn.Sequential(nn.Conv2d(18, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6_m = nn.Sequential(nn.Conv1d(64 * 2, args.emb_dims, kernel_size=1, bias=False),
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

        xyz = copy.deepcopy(x)

        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        node1, node_feature_1 = self.pool1(xyz, x1)  # (batch_size, 64, num_points) -> (batch_size, 64, num_points//4) 512
        agg_feature_1 = aggregate(xyz, node1, x1, self.k)
        node_feature_1 = torch.cat((node_feature_1, agg_feature_1), dim=1)

        x = get_graph_feature(node_feature_1,
                              k=self.k)  # (batch_size, 64, num_points//4) -> (batch_size, 64*2, num_points//4, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points//4, k) -> (batch_size, 64, num_points//4, k)
        x = self.conv4(x)  # (batch_size, 64, num_points//4, k) -> (batch_size, 64, num_points//4, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points//4, k) -> (batch_size, 64, num_points//4)

        node2, node_feature_2 = self.pool2(node1, x2)  # (batch_size, 64, num_points//4) -> (batch_size, 64, num_points//16) 128
        agg_feature_2 = aggregate(node1, node2, x2, self.k)
        node_feature_2 = torch.cat((node_feature_2, agg_feature_2), dim=1)

        x = get_graph_feature(node_feature_2,
                              k=self.k)  # (batch_size, 64, num_points//16) -> (batch_size, 64*2, num_points//16, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points//16, k) -> (batch_size, 64, num_points//16, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points//16, k) -> (batch_size, 64, num_points//16)

        # x = torch.cat((x1, x2, x3), dim=1)               # (batch_size, 64*3, num_points)
        node3, node_feature_3 = self.pool3(node2, x3)  # (batch_size, 64, num_points//16) -> (batch_size, 64, num_points//64) 32
        agg_feature_3 = aggregate(node2, node3, x3, self.k)
        node_feature_3 = torch.cat((node_feature_3, agg_feature_3), dim=1)

        x4 = self.conv6_m(node_feature_3)  # (batch_size, 64, num_points//64) -> (batch_size, emb_dims, num_points//64)
        # x = x4.max(dim=-1, keepdim=True)[0]              # (batch_size, emb_dims, num_points//64) -> (batch_size, emb_dims, 1)

        x = self.conv8_m(x4)  # (batch_size, 1024, num_points//64) -> (batch_size, 256, num_points//64)
        x = self.dp1(x)

        x = x.repeat(1, 1, 4)
        x = torch.cat((x, x3), dim=1)  # (batch_size, 256+64, num_points//16)
        x = self.conv9_m(x)  # (batch_size, 256+64, num_points//16) -> (batch_size, 256, num_points//16)
        x = self.dp2(x)

        x = x.repeat(1, 1, 4)
        x = torch.cat((x, x2), dim=1)  # (batch_size, 256+64, num_points//4)
        x = self.conv10_m(x)  # (batch_size, 256+64, num_points//4) -> (batch_size, 128, num_points//4)
        x = self.dp3(x)

        x = x.repeat(1, 1, 4)
        x = torch.cat((x, x1), dim=1)  # (batch_size, 128+64, num_points)
        x = self.conv11_m(x)  # (batch_size, 128+64, num_points) -> (batch_size, seg_num_all, num_points)

        return x, node1, node2, node3
