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
from util import knn, get_graph_feature, unpool, aggregate, MLP, EdgeConv, RandPool, Transform_Net


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.k = args.k
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv2_m = nn.Conv1d(64, args.emb_dims//2, kernel_size=1, bias=False)
        self.conv1_p = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.conv4_m = nn.Conv1d(128, args.emb_dims//2, kernel_size=1, bias=False)
        self.conv2_p = nn.Conv1d(256, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(256, 256, kernel_size=1, bias=False)
        self.conv6 = nn.Conv1d(256, 256, kernel_size=1, bias=False)
        self.conv6_m = nn.Conv1d(256, args.emb_dims//2, kernel_size=1, bias=False)
        self.conv3_p = nn.Conv1d(512, 512, kernel_size=1, bias=False)
        self.conv7 = nn.Conv1d(512, 512, kernel_size=1, bias=False)
        self.conv8 = nn.Conv1d(512, 512, kernel_size=1, bias=False)
        # self.conv8_m = nn.Conv1d(512, args.emb_dims//2, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn2_m = nn.BatchNorm1d(args.emb_dims//2)
        self.bn1_p = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn4_m = nn.BatchNorm1d(args.emb_dims//2)
        self.bn2_p = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(256)
        self.bn6_m = nn.BatchNorm1d(args.emb_dims//2)
        self.bn3_p = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(512)
        # self.bn8_m = nn.BatchNorm1d(args.emb_dims)

        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.lbn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

        # self.pool1 = IndexSelect(256, 64, neighs=40)
        # self.pool2 = IndexSelect(64, 128, neighs=20)
        # self.pool3 = IndexSelect(16, 256, neighs=10)
        # self.pool4 = IndexSelect(4, 512, neighs=10)
        # self.sigma = nn.Parameter(torch.zeros((2)), requires_grad=True)

    def forward(self, x):
        ret = []
        node = []
        node_static = []
        xyz = copy.deepcopy(x)
        x1 = F.relu(self.bn1(self.conv1(x)))
        x_t1 = F.relu(self.bn2_m(self.conv2_m(x1)))                                     #(batch_size, 512, num_points)

        # node_features, values, idx, ret1, node1_static, node1 = self.pool1(xyz, x1)
        node_features = x1[:, :, :self.args.num_points//4]
        node1 = xyz[:, :, :self.args.num_points//4]
        node_features_agg = aggregate(xyz, node1, x1, self.k)
        x_p1 = torch.cat((node_features, node_features_agg), dim=1)                     #(batch_size, 128, num_points//4)
        x_p1 = F.relu(self.bn1_p(self.conv1_p(x_p1)))

        x2 = F.relu(self.bn3(self.conv3(x_p1)))
        x2 = self.bn4(self.conv4(x2))                                            #(batch_size, 128, num_points//4)
        x2 = F.relu(x2 + x_p1)
        x_t2 = F.relu(self.bn4_m(self.conv4_m(x2)))                                     #(batch_size, 512, num_points//4)

        # node_features, values, idx, ret2, node2_static, node2 = self.pool2(node1_static, x2)
        node_features = x2[:, :, :self.args.num_points // 16]
        node2 = node1[:, :, :self.args.num_points // 16]
        node_features_agg = aggregate(node1, node2, x2, self.k // 2)
        x_p2 = torch.cat((node_features, node_features_agg), dim=1)                     #(batch_size, 256, num_points//16)
        x_p2 = F.relu(self.bn2_p(self.conv2_p(x_p2)))


        x3 = F.relu(self.bn5(self.conv5(x_p2)))
        x3 = self.bn6(self.conv6(x3))                                            #(batch_size, 256, num_points//16)
        x3 = F.relu(x3 + x_p2)
        x_t3 = F.relu(self.bn6_m(self.conv6_m(x3)))                                     #(batch_size, 512, num_points//16)

        # node_features, values, idx, ret3, node3_static, node3 = self.pool3(node2_static, x3)
        node_features = x3[:, :, :self.args.num_points // 64]
        node3 = node2[:, :, :self.args.num_points // 64]
        node_features_agg = aggregate(node2, node3, x3, self.k // 4)
        x_p3 = torch.cat((node_features, node_features_agg), dim=1)                     #(batch_size, 512, num_points//64)
        x_p3 = F.relu(self.bn3_p(self.conv3_p(x_p3)))

        x4 = F.relu(self.bn7(self.conv7(x_p3)))
        x4 = self.bn8(self.conv8(x4))
        x_t4 = F.relu(x4 + x_p3)

        x = torch.cat((F.adaptive_max_pool1d(x_t1, 1).squeeze(), F.adaptive_max_pool1d(x_t2, 1).squeeze(),
                       F.adaptive_max_pool1d(x_t3, 1).squeeze(), F.adaptive_max_pool1d(x_t4, 1).squeeze(),), dim=1)
        x = F.relu(self.lbn1(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)

        return x

class PointNet_scan(nn.Module):
    def __init__(self, args, output_channels=15, seg_num_all=2):
        super(PointNet_scan, self).__init__()
        self.args = args
        self.seg_num_all = seg_num_all

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn2_m = nn.BatchNorm1d(args.emb_dims)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.bn6 = nn.BatchNorm1d(128)
        self.bn7 = nn.BatchNorm1d(128)
        self.bn8 = nn.BatchNorm1d(128)
        self.bn9 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.ReLU())
        self.conv2_m = nn.Sequential(nn.Conv1d(64 * 2, args.emb_dims, kernel_size=1, bias=False),
                                     self.bn2_m,
                                     nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(256 * 2, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv1d(2*args.emb_dims + 256, 128, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv1d(128 + 256, 128, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.ReLU())
        self.conv8 = nn.Sequential(nn.Conv1d(128 + 64, 128, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.ReLU())
        self.conv9 = nn.Sequential(nn.Conv1d(128 + 64, 128, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.ReLU())
        self.dp3 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)

        self.pool1 = Pool(args.num_points // 4, 128, 0.2)
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

        x1 = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = self.conv2(x1)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)

        # pool(sample and aggregate)
        x_t1_ = torch.cat((x1, x2), dim=1)
        x_t1 = self.conv2_m(x_t1_)
        node1, node_features_1, node1_static = self.pool1(xyz, x_t1_)
        node_features_agg = aggregate(xyz, node1, x_t1_, 20)
        x = torch.cat((node_features_1, node_features_agg), dim=1)

        x3 = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x4 = self.conv4(x3)  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)

        x = torch.cat([x3, x4], dim=1)
        x_t2 = self.conv5(x)

        x_t1 = F.adaptive_max_pool1d(x_t1, 1).view(batch_size,
                                                   -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x_t2 = F.adaptive_max_pool1d(x_t2, 1).view(batch_size,
                                                   -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        vector = torch.cat((x_t1, x_t2), 1)  # (batch_size, emb_dims*2)

        ## classification
        x = F.relu(self.bn6(self.linear1(vector)))  # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.relu(self.bn7(self.linear2(x)))  # (batch_size, 512) -> (batch_size, 256)
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

        return logits_cls, logits_seg, node1, node1_static


class DGCNN_cls(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN_cls, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn2_m = nn.BatchNorm1d(args.emb_dims//2)
        self.bn2_p = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn4_m = nn.BatchNorm1d(args.emb_dims//2)
        self.bn4_p = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn6_m = nn.BatchNorm1d(args.emb_dims // 2)
        self.bn6_p = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(512)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2_m = nn.Sequential(nn.Conv1d(64, args.emb_dims//2, kernel_size=1, bias=False),
                                     self.bn2_m,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv2_p = nn.Sequential(nn.Conv1d(128, 128, kernel_size=1, bias=False),
                                     self.bn2_p,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 128, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4_m = nn.Sequential(nn.Conv1d(128, args.emb_dims//2, kernel_size=1, bias=False),
                                     self.bn4_m,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv4_p = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                     self.bn4_p,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(256*2, 256, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(256*2, 256, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6_m = nn.Sequential(nn.Conv1d(256, args.emb_dims//2, kernel_size=1, bias=False),
                                     self.bn6_m,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv6_p = nn.Sequential(nn.Conv1d(512, 512, kernel_size=1, bias=False),
                                     self.bn6_p,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv2d(512*2, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv2d(512*2, 512, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.conv8_m = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
        #                              self.bn8_m,
        #                              nn.LeakyReLU(negative_slope=0.2))
        # self.conv5 = nn.Sequential(nn.Conv1d(256*2, args.emb_dims, kernel_size=1, bias=False),
        #                            self.bn5,
        #                            nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn9 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn10 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        xyz = copy.deepcopy(x)

        ## level 1
        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        # pool(sample and aggregate)
        x_t1 = self.conv2_m(x2)                 #(batch_size, 64, num_points)
        # node_features, values, idx, ret1, node1_static, node1 = self.pool1(xyz, x2)
        node_features = x2[:, :, :self.args.num_points // 4]
        node1 = xyz[:, :, :self.args.num_points // 4]
        node_features_agg = aggregate(xyz, node1, x2, self.k)
        x_p1 = torch.cat((node_features, node_features_agg), dim=1)    #(batch_size, 128, num_points//4)
        x_p1 = self.conv2_p(x_p1)

        ## level2
        x = get_graph_feature(x_p1, k=self.k//2)   # (batch_size, 128, num_points//4) -> (batch_size, 128*2, num_points, k//2)
        x = self.conv3(x)                       # (batch_size, 128*2, num_points//4, k//2) -> (batch_size, 128, num_points, k//2)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points//4, k//2) -> (batch_size, 128, num_points//4)

        x = get_graph_feature(x3, k=self.k//2)  # (batch_size, 128, num_points//4) -> (batch_size, 128*2, num_points//4, k//2)
        x = self.conv4(x)                       # (batch_size, 128*2, num_points//4, k//2) -> (batch_size, 128, num_points//4, k//2)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points//4, k//2) -> (batch_size, 128, num_points//4)
        x4 = F.leaky_relu(x4 + x_p1, negative_slope=0.2)

        # pool(sample and aggregate)
        x_t2 = self.conv4_m(x4)                 #(batch_size, 128, num_points//4)
        # node_features, values, idx, ret2, node2_static, node2 = self.pool2(node1_static, x4)
        node_features = x4[:, :, :self.args.num_points // 16]
        node2 = node1[:, :, :self.args.num_points // 16]
        node_features_agg = aggregate(node1, node2, x4, self.k//2)
        x_p2 = torch.cat((node_features, node_features_agg), dim=1)    #(batch_size, 128*2, num_points//16)
        x_p2 = self.conv4_p(x_p2)

        ## level3
        x = get_graph_feature(x_p2, k=self.k // 4)  # (batch_size, 256, num_points//16) -> (batch_size, 256*2, num_points//16, k//4)
        x = self.conv5(x)                        # (batch_size, 256*2, num_points//16, k//4) -> (batch_size, 256, num_points//16, k//4)
        x5 = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 256, num_points//16, k//4) -> (batch_size, 256, num_points//16)

        x = get_graph_feature(x5, k=self.k // 4) # (batch_size, 256, num_points//16) -> (batch_size, 256*2, num_points//16, k//4)
        x = self.conv6(x)                        # (batch_size, 256*2, num_points//16, k//4) -> (batch_size, 256, num_points//16, k//4)
        x6 = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 256, num_points//16, k//4) -> (batch_size, 256, num_points//16)
        x6 = F.leaky_relu(x6 + x_p2, negative_slope=0.2)

        # pool(sample and aggregate)
        x_t3 = self.conv6_m(x6)                  # (batch_size, 256, num_points//16)
        # node_features, values, idx, ret3, node3_static, node3 = self.pool3(node2_static, x6)
        node_features = x6[:, :, :self.args.num_points // 64]
        node3 = node2[:, :, :self.args.num_points // 64]
        node_features_agg = aggregate(node2, node3, x6, self.k//4)
        x_p3 = torch.cat((node_features, node_features_agg), dim=1)    #(batch_size, 256*2, num_points//64)
        x_p3 = self.conv6_p(x_p3)

        ## level4
        x = get_graph_feature(x_p3, k=self.k // 8)  # (batch_size, 512, num_points//64) -> (batch_size, 512, num_points//64, k//8)
        x = self.conv7(x)                        # (batch_size, 512*2, num_points//64, k//8) -> (batch_size, 512, num_points//64, k//8)
        x7 = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 512, num_points//64, k//8) -> (batch_size, 512, num_points//64)

        x = get_graph_feature(x7, k=self.k // 8) # (batch_size, 512, num_points//64) -> (batch_size, 512*2, num_points//64, k//8)
        x = self.conv8(x)                        # (batch_size, 512*2, num_points//64, k//8) -> (batch_size, 512, num_points//64, k//8)
        x_t4 = x.max(dim=-1, keepdim=False)[0]   # (batch_size, 512, num_points//64, k//8) -> (batch_size, 512, num_points//64)
        x_t4 = F.leaky_relu(x_t4 + x_p3, negative_slope=0.2)

        xt1 = F.adaptive_max_pool1d(x_t1, 1).view(batch_size, -1)           # (batch_size, emb_dims//2, num_points) -> (batch_size, emb_dims//2)
        xt2 = F.adaptive_max_pool1d(x_t2, 1).view(batch_size, -1)           # (batch_size, emb_dims//2, num_points//4) -> (batch_size, emb_dims//2)
        xt3 = F.adaptive_max_pool1d(x_t3, 1).view(batch_size, -1)           # (batch_size, emb_dims//2, num_points//16) -> (batch_size, emb_dims//2)
        xt4 = F.adaptive_max_pool1d(x_t4, 1).view(batch_size, -1)           # (batch_size, emb_dims//2, num_points//64) -> (batch_size, emb_dims//2)
        x = torch.cat((xt1, xt2, xt3, xt4), 1)                              # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn9(self.linear1(x)), negative_slope=0.2)     # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn10(self.linear2(x)), negative_slope=0.2)    # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)                                                 # (batch_size, 256) -> (batch_size, output_channels)

        return x
        # return x, node1, node1_static


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

        self.pool1 = Pool(args.num_points // 4, 128, 0.2)
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
        node1, node_features_1, node1_static = self.pool1(xyz, x_t1_)
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

        x = unpool(node1_static, xyz, x)
        # print('shape of x2: {}'.format(x2.shape))
        # print('shape of x: {}'.format(x.shape))
        x = torch.cat((x, x2), dim=1)  # (batch_size, 256+64, num_points)
        x = self.conv8(x)  # (batch_size, 256+64, num_points) -> (batch_size, 256, num_points)

        x = torch.cat((x, x1), dim=1)  # (batch_size, 256+64, num_points)
        x = self.conv9(x)  # (batch_size, 256+64, num_points) -> (batch_size, 128, num_points)
        x = self.dp3(x)

        logits_seg = self.conv10(x)  # (batch_size, 128, num_points) -> (batch_size, seg_num_all, num_points)

        return logits_cls, logits_seg, node1, node1_static


class DGCNN_partseg(nn.Module):
    def __init__(self, args, seg_num_all):
        super(DGCNN_partseg, self).__init__()
        self.args = args
        self.seg_num_all = seg_num_all
        self.k = args.k
        self.transform_net = Transform_Net(args)

        self.pool1 = RandPool(self.args.num_points//4,  self.k, 64)
        self.pool2 = RandPool(self.args.num_points//16, self.k, 64)
        self.pool3 = RandPool(self.args.num_points//64, self.k, 64)

        self.ec0 = EdgeConv(num_neighs=self.k,    dims=[3, 64, 64])
        self.pn0 = MLP([64, 1024])
        self.ec1 = EdgeConv(num_neighs=self.k,    dims=[64, 64, 64])
        self.pn1 = MLP([64, 1024])
        self.ec2 = EdgeConv(num_neighs=self.k,    dims=[64, 64, 64])
        self.pn2 = MLP([64, 1024])
        self.ec3 = EdgeConv(num_neighs=self.k//2,    dims=[64, 64, 64])
        self.pn3 = MLP([64, 1024])

        # self.pn4 = MLP([256, 1024])
        self.label_conv = MLP([16, 64])

        self.pn4 = MLP([1152, 256])
        self.dp1 = nn.Dropout(p=args.dropout)
        self.pn5 = MLP([256+64, 256])
        self.dp2 = nn.Dropout(p=args.dropout)
        self.pn6 = MLP([256+64, 256])
        self.dp3 = nn.Dropout(p=args.dropout)
        self.pn7 = MLP([256+64, 128])
        self.conv8 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)


    def forward(self, x, l):
        batch_size = x.size(0)

        x_ = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x_)           # (batch_size, 3, 3)
        x = x.transpose(2, 1)                # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)                  # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)                # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        node0 = x

        x0  = self.ec0(x)
        x_t0 = torch.max(self.pn0(x0), dim=-1, keepdim=True)[0]

        if self.args.pool:
            node1, node1_feats = self.pool1(node0, x0)
        else:
            node1 = node0
            node1_feats = x0

        x1  = self.ec1(node1_feats)
        x_t1 = torch.max(self.pn1(x1), dim=-1, keepdim=True)[0]
        if self.args.res:
            x1 = F.relu(x1 + node1_feats)  # (batch_size, 64, num_points//4)

        if self.args.pool:
            node2, node2_feats = self.pool2(node1, x1)
        else:
            node2 = node1
            node2_feats = x1

        x2  = self.ec2(node2_feats)
        x_t2 = torch.max(self.pn2(x2), dim=-1, keepdim=True)[0]
        if self.args.res:
            x2 = F.relu(x2 + node2_feats)  # (batch_size, 64, num_points//16)

        if self.args.pool:
            node3, node3_feats = self.pool3(node2, x2)
        else:
            node3 = node2
            node3_feats = x2

        x3  = self.ec3(node3_feats)
        x_t3 = torch.max(self.pn3(x3), dim=-1, keepdim=True)[0]
        if self.args.res:
            x3 = F.relu(x3 + node3_feats)  # (batch_size, 64, num_points//64)

        x = torch.cat([x_t0, x_t1, x_t2, x_t3], dim=-1)
        x = torch.max(x, dim=-1, keepdim=True)[0]
        l = l.view(batch_size, -1, 1)      # (batch_size, num_categoties, 1)
        l = self.label_conv(l)             # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)
        x = torch.cat([x, l], dim=1)       # (batch_size, 1088, 1)

        x = x.repeat(1, 1, x3.shape[-1])   # (batch_size, 1088, num_points//64)
        x = torch.cat([x, x3], dim=1)      # (batch_size, 1088+64, num_points//64)
        x = self.pn4(x)                    # (batch_size, 1088, num_points//64) -> (batch_size, 256, num_points//64)
        x = self.dp1(x)

        if self.args.pool:
            x = unpool(node3, node2, x)        # (batch_size, 256, num_points//16)
        x = torch.cat((x, x2), dim=1)      # (batch_size, 256+64, num_points//16)
        x = self.pn5(x)                    # (batch_size, 256+64, num_points//16) -> (batch_size, 256, num_points//16)
        x = self.dp2(x)

        if self.args.pool:
            x = unpool(node2, node1, x)        # (batch_size, 256, num_points//4)
        x = torch.cat((x, x1), dim=1)      # (batch_size, 256+64, num_points//4)
        x = self.pn6(x)                    # (batch_size, 256+64, num_points//4) -> (batch_size, 256, num_points//4)
        x = self.dp3(x)

        if self.args.pool:
            x = unpool(node1, node0, x)        # (batch_size, 64, num_points)
        x = torch.cat((x, x0), dim=1)      # (batch_size, 256+64, num_points)
        x = self.pn7(x)                    # (batch_size, 256+64, num_points) -> (batch_size, 128, num_points)

        x = self.conv8(x)                 # (batch_size, 128, num_points) -> (batch_size, seg_num_all, num_points)

        return x


class DGCNN_semseg(nn.Module):
    def __init__(self, args):
        super(DGCNN_semseg, self).__init__()
        self.args = args
        self.k = args.k

        self.pool0 = RandPool(self.args.num_points // 4, self.k, 64)
        self.pool1 = RandPool(self.args.num_points // 16, self.k, 64)
        self.pool2 = RandPool(self.args.num_points // 64, self.k, 64)

        self.ec1 = EdgeConv(num_neighs=self.k, dims=[9, 64, 64])
        self.ec2 = EdgeConv(num_neighs=self.k, dims=[64, 64, 64])
        self.pn3 = MLP([64, 1024])
        self.ec4 = EdgeConv(num_neighs=self.k, dims=[64, 64, 64])
        self.ec5 = EdgeConv(num_neighs=self.k, dims=[64, 64, 64])
        self.pn6 = MLP([64, 1024])
        self.ec7 = EdgeConv(num_neighs=self.k, dims=[64, 64])
        self.ec8 = EdgeConv(num_neighs=self.k, dims=[64, 64])
        self.pn9 = MLP([64, 1024])
        self.ec10 = EdgeConv(num_neighs=self.k//2, dims=[64, 64])
        self.ec11 = EdgeConv(num_neighs=self.k//2, dims=[64, 64])
        self.pn12 = MLP([64, 1024])

        self.pn13 = MLP([1088, 256])
        self.pn14 = MLP([256 + 64, 256])
        self.pn15 = MLP([256 + 64, 256])
        self.pn16 = MLP([256 + 64, 128])
        self.dp = nn.Dropout(p=args.dropout)
        self.conv17 = nn.Conv1d(128, 13, kernel_size=1, bias=False)

    def forward(self, x):
        node0 = x[:, :3, :]                       # use normalized xyz

        x   = self.ec1(x)
        x0  = self.ec2(x)
        x_t0 = torch.max(self.pn3(x0), dim=-1, keepdim=True)[0]
        node1, node1_feats = self.pool0(node0, x0)

        x   = self.ec4(node1_feats)
        x1  = self.ec5(x)
        x_t1 = torch.max(self.pn6(x1), dim=-1, keepdim=True)[0]
        if self.args.res:
            x1 = F.relu(x1 + node1_feats)  # (batch_size, 64, num_points//4)

        node2, node2_feats = self.pool1(node1, x1)

        x   = self.ec7(node2_feats)
        x2  = self.ec8(x)
        x_t2 = torch.max(self.pn9(x2), dim=-1, keepdim=True)[0]
        if self.args.res:
            x2 = F.relu(x2 + node2_feats)  # (batch_size, 64, num_points//16)

        node3, node3_feats = self.pool2(node2, x2)

        x   = self.ec10(node3_feats)
        x3  = self.ec11(x)
        x_t3 = torch.max(self.pn12(x3), dim=-1, keepdim=True)[0]
        if self.args.res:
            x3 = F.relu(x3 + node3_feats)  # (batch_size, 64, num_points//64)

        x = torch.cat([x_t0, x_t1, x_t2, x_t3], dim=-1)
        x = torch.max(x, dim=-1, keepdim=True)[0]

        x = x.repeat(1, 1, x3.shape[-1])
        x = torch.cat([x, x3], dim=1)
        x = self.pn13(x)  # (batch_size, 64, num_points//64) -> (batch_size, 256, num_points//64)

        x = unpool(node3, node2, x)  # (batch_size, 64, num_points//16)
        x = torch.cat((x, x2), dim=1)  # (batch_size, 256+64, num_points//16)
        x = self.pn14(x)  # (batch_size, 256+64, num_points//16) -> (batch_size, 256, num_points//16)

        x = unpool(node2, node1, x)  # (batch_size, 64, num_points//4)
        x = torch.cat((x, x1), dim=1)  # (batch_size, 256+64, num_points//4)
        x = self.pn15(x)  # (batch_size, 256+64, num_points//4) -> (batch_size, 256, num_points//4)

        x = unpool(node1, node0, x)  # (batch_size, 64, num_points)
        x = torch.cat((x, x0), dim=1)  # (batch_size, 256+64, num_points)
        x = self.pn16(x)  # (batch_size, 256+64, num_points) -> (batch_size, 128, num_points)

        x = self.dp(x)
        x = self.conv17(x)  # (batch_size, 128, num_points) -> (batch_size, seg_num_all, num_points)
        # (batch_size, 128, num_points) -> (batch_size, 13, num_points)
        return x, node1, node2, node3
