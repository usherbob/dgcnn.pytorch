#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: main_partseg.py
@Time: 2019/12/31 11:17 AM
"""


from __future__ import print_function
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import ShapeNetPart
from model import DGCNN_partseg
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, mi_loss, compute_chamfer_distance, IOStream
import sklearn.metrics as metrics

seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]

def _init_():
    ckpt_dir = BASE_DIR+'/ckpt/partseg'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(ckpt_dir+'/'+args.exp_name):
        os.makedirs(ckpt_dir+'/'+args.exp_name)
    if not os.path.exists(ckpt_dir+'/'+args.exp_name+'/'+'models'):
        os.makedirs(ckpt_dir+'/'+args.exp_name+'/'+'models')
    if not os.path.exists(ckpt_dir+'/'+args.exp_name+'/'+'visu'):
        os.makedirs(ckpt_dir+'/'+args.exp_name+'/'+'visu')
    if not os.path.exists(ckpt_dir+'/'+args.exp_name+'/'+'pred'):
        os.makedirs(ckpt_dir+'/'+args.exp_name+'/'+'pred')
    os.system('cp main_partseg.py '+ckpt_dir+'/'+args.exp_name+'/'+'main_partseg.py.backup')
    os.system('cp model.py ' + ckpt_dir + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py ' + ckpt_dir + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py ' + ckpt_dir + '/' + args.exp_name + '/' + 'data.py.backup')


def calculate_shape_IoU(pred_np, seg_np, label, class_choice):
    label = label.squeeze()
    shape_ious = []
    for shape_idx in range(seg_np.shape[0]):
        if not class_choice:
            start_index = index_start[label[shape_idx]]
            num = seg_num[label[shape_idx]]
            parts = range(start_index, start_index + num)
        else:
            parts = range(seg_num[label[0]])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious


def train(args, io):
    train_dataset = ShapeNetPart(partition='trainval', num_points=args.num_points, class_choice=args.class_choice, BASE_DIR=BASE_DIR)
    if (len(train_dataset) < 100):
        drop_last = False
    else:
        drop_last = True
    train_loader = DataLoader(train_dataset, num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=drop_last)
    test_loader = DataLoader(ShapeNetPart(partition='test', num_points=args.num_points, class_choice=args.class_choice, BASE_DIR=BASE_DIR),
                            num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    
    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    seg_num_all = train_loader.dataset.seg_num_all
    seg_start_index = train_loader.dataset.seg_start_index
    if args.model == 'dgcnn':
        model = DGCNN_partseg(args, seg_num_all).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    if args.model_path != '':
        ckpt = torch.load(args.model_path)
        model.load_state_dict(ckpt, strict=False)
    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.5)

    criterion = cal_loss

    best_test_iou = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        start = time.time()
        train_loss = 0.0
        train_cd_loss = 0.0
        train_mi_loss = 0.0
        train_cls_loss = 0.0
        loss_cd = 0.0
        loss_mi = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []
        for data, label, seg in train_loader:
            seg = seg - seg_start_index
            label_one_hot = np.zeros((label.shape[0], 16))
            for idx in range(label.shape[0]):
                label_one_hot[idx, label[idx]] = 1
            label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
            data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            seg_pred, ret1, ret2, ret3, node1, node2, node3, node1_static, node2_static, node3_static = model(data, label_one_hot)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss_cls = criterion(seg_pred.view(-1, seg_num_all), seg.view(-1,1).squeeze())
            if args.cd_weights != 0:
                loss_cd = compute_chamfer_distance(node1, data) + compute_chamfer_distance(node2, node1_static) \
                      + compute_chamfer_distance(node3, node2_static)
            if args.mi_weights != 0:
                loss_mi = mi_loss(ret1) + mi_loss(ret2) + mi_loss(ret3)
            loss = loss_cls + args.mi_weights * loss_mi + args.cd_weights * loss_cd
            loss.backward()
            opt.step()
            pred = seg_pred.max(dim=2)[1]               # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item() * batch_size
            train_cls_loss += loss_cls.item() * batch_size
            train_cd_loss += loss_cd * batch_size
            train_mi_loss += loss_mi * batch_size
            seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))       # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))      # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
            train_label_seg.append(label.reshape(-1))
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_label_seg = np.concatenate(train_label_seg)
        train_ious = calculate_shape_IoU(train_pred_seg, train_true_seg, train_label_seg, args.class_choice)
        duration = time.time() - start
        outstr = 'Train %d, loss: %.6f, loss_cls: %.6f, loss_cd: %.6f, loss_mi: %.6f, train acc: %.6f, ' \
                 'train avg acc: %.6f, train iou: %.6f, time usage: %d s'                         % (epoch,
                                                                                                  train_loss*1.0/count,
                                                                                                  train_cls_loss*1.0/count,
                                                                                                  train_cd_loss*1.0/count,
                                                                                                  train_mi_loss*1.0/count,
                                                                                                  train_acc,
                                                                                                  avg_per_class_acc,
                                                                                                  np.mean(train_ious),
                                                                                                  duration)
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        test_cd_loss = 0.0
        test_mi_loss = 0.0
        test_cls_loss = 0.0
        count = 0.0
        model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        test_label_seg = []
        with torch.no_grad():
            for data, label, seg in test_loader:
                seg = seg - seg_start_index
                label_one_hot = np.zeros((label.shape[0], 16))
                for idx in range(label.shape[0]):
                    label_one_hot[idx, label[idx]] = 1
                label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
                data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                seg_pred, ret1, ret2, ret3, node1, node2, node3, node1_static, node2_static, node3_static = model(data, label_one_hot)
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                loss_cls = criterion(seg_pred.view(-1, seg_num_all), seg.view(-1,1).squeeze())
                if args.cd_weights != 0:
                    loss_cd = compute_chamfer_distance(node1, data) + compute_chamfer_distance(node2, node1_static) \
                              + compute_chamfer_distance(node3, node2_static)
                if args.mi_weights != 0:
                    loss_mi = mi_loss(ret1) + mi_loss(ret2) + mi_loss(ret3)
                loss = loss_cls + args.mi_weights * loss_mi + args.cd_weights * loss_cd
                pred = seg_pred.max(dim=2)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                test_cls_loss += loss_cls.item() * batch_size
                test_cd_loss += loss_cd * batch_size
                test_mi_loss += loss_mi * batch_size
                seg_np = seg.cpu().numpy()
                pred_np = pred.detach().cpu().numpy()
                test_true_cls.append(seg_np.reshape(-1))
                test_pred_cls.append(pred_np.reshape(-1))
                test_true_seg.append(seg_np)
                test_pred_seg.append(pred_np)
                test_label_seg.append(label.reshape(-1))
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_label_seg = np.concatenate(test_label_seg)
        test_ious = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg, args.class_choice)
        outstr = 'Test %d, loss: %.6f, loss_cls: %.6f, loss_cd: %.6f, loss_mi: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (epoch,
                                                                                              test_loss*1.0/count,
                                                                                              test_cls_loss*1.0/count,
                                                                                              test_cd_loss*1.0/count,
                                                                                              test_mi_loss*1.0/count,
                                                                                              test_acc,
                                                                                              avg_per_class_acc,
                                                                                              np.mean(test_ious))
        io.cprint(outstr)
        if np.mean(test_ious) >= best_test_iou:
            best_test_iou = np.mean(test_ious)
            torch.save(model.state_dict(), BASE_DIR+'/ckpt/partseg/%s/models/model.t7' % args.exp_name)


def test(args, io):
    test_loader = DataLoader(ShapeNetPart(partition='test', num_points=args.num_points, class_choice=args.class_choice,
                                          BASE_DIR=BASE_DIR), batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    seg_num_all = test_loader.dataset.seg_num_all
    seg_start_index = test_loader.dataset.seg_start_index
    if args.model == 'dgcnn':
        model = DGCNN_partseg(args, seg_num_all).to(device)
    else:
        raise Exception("Not implemented")

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    batch_count = 0
    test_true_cls = []
    test_pred_cls = []
    test_true_seg = []
    test_pred_seg = []
    test_label_seg = []
    for data, label, seg in test_loader:
        batch_count += 1
        seg = seg - seg_start_index
        label_one_hot = np.zeros((label.shape[0], 16))
        for idx in range(label.shape[0]):
            label_one_hot[idx, label[idx]] = 1
        label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
        data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
        data = data.permute(0, 2, 1)
        seg_pred, ret1, ret2, ret3, node1, node2, node3, node1_static, node2_static, node3_static = model(data, label_one_hot)
        seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        pred = seg_pred.max(dim=2)[1]
        seg_np = seg.cpu().numpy()
        pred_np = pred.detach().cpu().numpy() - seg_start_index
        test_true_cls.append(seg_np.reshape(-1))
        test_pred_cls.append(pred_np.reshape(-1))
        test_true_seg.append(seg_np)
        test_pred_seg.append(pred_np)
        test_label_seg.append(label.reshape(-1))
        if args.visu:
            pred_ = torch.cat([data, torch.unsqueeze(pred.to(torch.float), dim=1)], dim=1)
            for i in range(data.shape[0]):
                np.save(BASE_DIR+'/ckpt/partseg/%s/visu/node0_%02d_%04d.npy' % (args.exp_name, label[i], batch_count * args.test_batch_size + i),
                        data[i, :, :].detach().cpu().numpy())
                np.save(BASE_DIR+'/ckpt/partseg/%s/visu/node1_%02d_%04d.npy' % (args.exp_name, label[i], batch_count * args.test_batch_size + i),
                        node1_static[i, :, :].detach().cpu().numpy())
                np.save(BASE_DIR+'/ckpt/partseg/%s/visu/node2_%02d_%04d.npy' % (
                args.exp_name, label[i], batch_count * args.test_batch_size + i),
                        node2_static[i, :, :].detach().cpu().numpy())
                np.save(BASE_DIR+'/ckpt/partseg/%s/visu/node3_%02d_%04d.npy' % (
                args.exp_name, label[i], batch_count * args.test_batch_size + i),
                        node3_static[i, :, :].detach().cpu().numpy())
                np.save(BASE_DIR + '/ckpt/partseg/%s/pred/pred%02d_%04d.npy' % (
                    args.exp_name, label[i], batch_count * args.test_batch_size + i),
                        pred_[i, :, :].detach().cpu().numpy())

    test_true_cls = np.concatenate(test_true_cls)
    test_pred_cls = np.concatenate(test_pred_cls)
    test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
    test_true_seg = np.concatenate(test_true_seg, axis=0)
    test_pred_seg = np.concatenate(test_pred_seg, axis=0)
    test_label_seg = np.concatenate(test_label_seg)
    test_ious = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg, args.class_choice)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (test_acc,
                                                                             avg_per_class_acc,
                                                                             np.mean(test_ious))
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--base_dir', type=str, default='/opt/data/private', metavar='N',
                        help='Path to data and ckpt')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--dataset', type=str, default='shapenetpart', metavar='N',
                        choices=['shapenetpart'])
    parser.add_argument('--class_choice', type=str, default=None, metavar='N',
                        choices=['airplane', 'bag', 'cap', 'car', 'chair',
                                 'earphone', 'guitar', 'knife', 'lamp', 'laptop', 
                                 'motor', 'mug', 'pistol', 'rocket', 'skateboard', 'table'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=40, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--visu', type=bool, default=False,
                        help='visualize atp selection')
    parser.add_argument('--res', type=bool, default=False,
                        help='Turn on residual connection')

    parser.add_argument('--pool', type=str, default=None, metavar='N',
                        choices=['GDP', 'RDP', 'MIP'],
                        help='Pooling method implemented')
    parser.add_argument('--cd_weights', type=float, default=0.0, metavar='CDW',
                        help='weights of cd_loss')
    parser.add_argument('--mi_weights', type=float, default=0.0, metavar='MIW',
                        help='weights of mi_loss')
    args = parser.parse_args()

    BASE_DIR = args.base_dir

    _init_()

    io = IOStream(BASE_DIR+'/ckpt/partseg/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
