#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: util
@Time: 4/5/19 3:47 PM
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)


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
    xyz_expanded = xyz.unsqueeze(2).expand(unknown_xyz.size(0), unknown_xyz.size(1), unknown_xyz.size(2),
                                           M)  # B, 3, N, M
    unknown_xyz_expanded = unknown_xyz.unsqueeze(3).expand_as(xyz_expanded)

    # calcuate difference between x and each node
    diff = unknown_xyz_expanded - xyz_expanded  # Bx3xNxM
    diff_norm = (diff ** 2).sum(dim=1)  # BxNxM
    nn_dist, nn_idx = torch.topk(diff_norm, k=3, dim=-1, largest=False, sorted=False)  # BxNx3
    weight = F.softmax(nn_dist, dim=-1)  # BxNx3

    # gather known features and aggregate unknown features
    nn_idx_fold = nn_idx.reshape(nn_idx.shape[0], -1)  # Bx3N
    nn_idx_fold = nn_idx_fold.unsqueeze(1).expand(
        features.size(0), features.size(1), nn_idx_fold.size(-1))  # B x C x 3N
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
    nn_idx_fold = nn_idx.reshape(nn_idx.shape[0], -1)  # BxkM
    nn_idx_fold = nn_idx_fold.unsqueeze(1).expand(
        features.size(0), features.size(1), nn_idx_fold.size(-1))
    # B x C x kM
    feature_grouped = features.gather(dim=2, index=nn_idx_fold)  # B x C x kM
    feature_unfold = feature_grouped.reshape(features.shape[0], features.shape[1],
                                             nn_idx.shape[1], nn_idx.shape[2])  # B x C x k x M
    feature_max = torch.max(feature_unfold, dim=2)[0]  # BxCxM
    return feature_max

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


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super().__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x))
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x))
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), 1).squeeze(-1)
        v = logits.shape[1]
        return logits, logits[:, :v // 2]


class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        last_dim = dims[0]
        self.dims = dims
        for i in range(1, len(dims)):
            setattr(self, "conv_{:d}".format(i),
                    nn.Sequential(nn.Conv1d(last_dim, dims[i], kernel_size=1, bias=False),
                                  nn.BatchNorm1d(dims[i]),
                                  nn.ReLU()))
            last_dim = dims[i]


    def forward(self, x):
        for i in range(1, len(self.dims)):
            x = getattr(self, 'conv_{:d}'.format(i))(x)
        return x

class EdgeConv(nn.Module):
    def __init__(self, num_neighs, dims):
        super().__init__()
        self.num_neighs = num_neighs
        last_dim = dims[0] * 2
        self.dims = dims
        for i in range(1, len(dims)):
            setattr(self, "conv_{:d}".format(i),
                    nn.Sequential(nn.Conv2d(last_dim, dims[i], kernel_size=1, bias=False),
                                      nn.BatchNorm2d(dims[i]),
                                      nn.LeakyReLU(negative_slope=0.2)))
            last_dim = dims[i]

    def forward(self, x):
        x = get_graph_feature(x, k=self.num_neighs)
        for i in range(1, len(self.dims)):
            x = getattr(self, 'conv_{:d}'.format(i))(x)
        x = x.max(dim=-1, keepdim=False)[0]
        return x

class RandPool(nn.Module):
    def __init__(self, num_sample, num_agg, num_channels):
        super().__init__()
        '''
        num_sample: Number of sampled points
        num_agg: Number of aggregated points
        '''
        self.num_sample = num_sample
        self.num_agg = num_agg
        self.conv = nn.Sequential(nn.Conv1d(num_channels * 2, num_channels, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(num_channels),
                                  nn.ReLU())

    def forward(self, input_coords, input_feats):
        pool_coords = input_coords[:, :, :self.num_sample]
        pool_feats = input_feats[:, :, :self.num_sample]
        agg_features = aggregate(input_coords, pool_coords, input_feats, self.num_agg)
        pool_feats = torch.cat((pool_feats, agg_features), dim=1)
        pool_feats = self.conv(pool_feats)
        return pool_coords, pool_feats


class IndexSelect(nn.Module):
    '''
    key module of mutual information pooling
    '''

    def __init__(self, k, n_h, neighs=20):
        super().__init__()
        self.k = k
        self.sigm = nn.Sigmoid()
        self.fc = nn.Sequential(nn.Conv1d(n_h, n_h, kernel_size=1),
                                nn.BatchNorm1d(n_h),
                                nn.ReLU())
        self.disc = Discriminator(n_h)
        self.center = EdgeConv(neighs, n_h, n_h)

    def forward(self, xyz, seq1, samp_bias1=None, samp_bias2=None):
        # seq2 = torch.zeros_like(seq1)
        seq2 = seq1[:, :, torch.randperm(seq1.shape[-1])]  # negative sampling
        h_1 = self.fc(seq1)
        h_2 = self.fc(seq2)
        h_n1 = self.center(h_1)

        X = self.sigm(h_n1)
        ret, ret_true = self.disc(X.permute(0, 2, 1).contiguous(), h_1.permute(0, 2, 1).contiguous(),
                                  h_2.permute(0, 2, 1).contiguous(), samp_bias1, samp_bias2)
        scores = self.sigm(ret_true).squeeze()
        # num_nodes = h_1.shape[1]
        values, idx = torch.topk(scores, self.k, dim=1)

        seq_idx = idx.unsqueeze(2).repeat(1, 1, seq1.shape[1])
        seq_idx = seq_idx.permute(0, 2, 1)
        seq_static = seq1.gather(2, seq_idx)  # BxCxnpoint
        seq = torch.mul(seq_static, values.unsqueeze(dim=1))
        xyz_idx = idx.unsqueeze(2).repeat(1, 1, xyz.shape[1])
        xyz_idx = xyz_idx.permute(0, 2, 1)
        xyz_static = xyz.gather(2, xyz_idx)  # Bx3xnpoint
        xyz = torch.mul(xyz_static, values.unsqueeze(dim=1))
        return seq, values, idx, ret, xyz_static, xyz

def mi_loss(ret):
    N = ret.shape[1] // 2
    device = ret.device
    lbl_t_s1 = torch.ones(ret.shape[0], N)
    lbl_f_s1 = torch.zeros(ret.shape[0], N)
    milbl_s1 = torch.cat((lbl_t_s1, lbl_f_s1), 1).to(device)
    loss = F.binary_cross_entropy_with_logits(ret, milbl_s1)
    return loss

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss

def compute_chamfer_distance(p1, p2):
    '''
    Calculate Chamfer Distance between two point sets
    :param p1: size[b, D, N]
    :param p2: size[b, D, M]
    :param debug: whether need to output debug info
    :return: sum of Chamfer Distance of two point sets
    '''

    diff = p1[:, :, :, None] - p2[:, :, None, :]
    dist = torch.sum(diff*diff,  dim=1) #[B, N, M]
    # dist1 = dist
    # dist2 = torch.transpose(dist, 1, 2)

    dist_min1, _ = torch.min(dist, dim=1)
    dist_min2, _ = torch.min(dist, dim=2)

    return (torch.sum(dist_min1)/dist.shape[1] + torch.sum(dist_min2)/dist.shape[2])
    # return dist_min1, dist_min2

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()
