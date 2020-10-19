#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main_cls.py
@Time: 2018/10/13 10:39 PM

Modified by 
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@Time: 2019/12/30 9:32 PM
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import ScanObject
from model import DGCNN_scan
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, compute_chamfer_distance, IOStream
import sklearn.metrics as metrics


def _init_():
    ckpt_dir = '/opt/data/private/ckpt/scan'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(ckpt_dir + '/' + args.exp_name):
        os.makedirs(ckpt_dir + '/' + args.exp_name)
    if not os.path.exists(ckpt_dir + '/' + args.exp_name + '/' + 'models'):
        os.makedirs(ckpt_dir + '/' + args.exp_name + '/' + 'models')
    if not os.path.exists(ckpt_dir + '/' + args.exp_name + '/' + 'visu'):
        os.makedirs(ckpt_dir + '/' + args.exp_name + '/' + 'visu')
    os.system('cp main_scan.py ' + ckpt_dir + '/' + args.exp_name + '/' + 'main_scan.py.backup')
    os.system('cp model.py ' + ckpt_dir + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py ' + ckpt_dir + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py ' + ckpt_dir + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args, io):
    train_loader = DataLoader(ScanObject(h5_filename='/opt/data/private/data/ScanObjectNN/main_split/training_objectdataset.h5', num_points=args.num_points), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ScanObject(h5_filename='/opt/data/private/data/ScanObjectNN/main_split/test_objectdataset.h5', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN_scan(args).to(device)
    else:
        raise Exception("Not implemented")

    print(str(model))

    model = nn.DataParallel(model)
    seg_num_all = 2
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
        scheduler = StepLR(opt, step_size=20, gamma=0.7)
    
    criterion = cal_loss

    best_test_acc = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        train_cd_loss = 0.0
        train_cls_loss = 0.0
        train_seg_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label, seg in train_loader:
            data, label, seg = data.to(device), label.to(device).squeeze(), seg.to(device)
            # data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits_cls, logits_seg, node1, node1_static = model(data)
            loss_cls = criterion(logits_cls, label)
            seg_pred = logits_seg.permute(0, 2, 1).contiguous()
            loss_seg = criterion(seg_pred.view(-1, seg_num_all), seg.view(-1, 1).squeeze())
            loss_cd = compute_chamfer_distance(node1, data)
            loss = loss_cls + loss_seg + loss_cd
            loss.backward()
            opt.step()
            preds = logits_cls.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_cls_loss += loss_cls.item() * batch_size
            train_seg_loss += loss_seg.item() * batch_size
            train_cd_loss += loss_cd.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, loss_cls: %.6f, loss_seg: %.6f, loss_cd: %.6f, train acc: %.6f, train avg acc: %.6f' \
                                                                                 % (epoch,
                                                                                    train_loss * 1.0 / count,
                                                                                    train_cls_loss * 1.0 / count,
                                                                                    train_seg_loss * 1.0 / count,
                                                                                    train_cd_loss * 1.0 / count,
                                                                                    metrics.accuracy_score(
                                                                                        train_true, train_pred),
                                                                                    metrics.balanced_accuracy_score(
                                                                                        train_true, train_pred))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        test_cd_loss = 0.0
        test_cls_loss = 0.0
        test_seg_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        with torch.no_grad():
            for data, label, seg in test_loader:
                data, label, seg = data.to(device), label.to(device).squeeze(), seg.to(device)
                # data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                logits_cls, logits_seg, node1, node1_static = model(data)
                loss_cls = criterion(logits_cls, label)
                seg_pred = logits_seg.permute(0, 2, 1).contiguous()
                loss_seg = criterion(seg_pred.view(-1, seg_num_all), seg.view(-1, 1).squeeze())
                loss_cd = compute_chamfer_distance(node1, data)
                loss = loss_cls + loss_seg + loss_cd
                preds = logits_cls.max(dim=1)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                test_cls_loss += loss_cls.item() * batch_size
                test_seg_loss += loss_seg.item() * batch_size
                test_cd_loss += loss_cd.item() * batch_size
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, loss_cls: %.6f, loss_seg: %.6f, loss_cd: %.6f, test acc: %.6f, test avg acc: %.6f' \
                                                                             % (epoch,
                                                                                test_loss * 1.0 / count,
                                                                                test_cls_loss * 1.0 / count,
                                                                                test_seg_loss * 1.0 / count,
                                                                                test_cd_loss * 1.0 / count,
                                                                                test_acc,
                                                                                avg_per_class_acc)
        io.cprint(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), '/opt/data/private/ckpt/scan/%s/models/model.t7' % args.exp_name)


def test(args, io):
    test_loader = DataLoader(ScanObject(h5_filename='/opt/data/private/data/ScanObjectNN/main_split/test_objectdataset.h5',
                                        num_points=args.num_points), batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN_scan(args).to(device)
    else:
        raise Exception("Not implemented")

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_true = []
    test_pred = []
    count = 0
    for data, label in test_loader:
        count += 1
        data, label, seg = data.to(device), label.to(device).squeeze(), seg.to(device)
        data = data.permute(0, 2, 1)
        logits_cls, logits_seg, node1, node1_static = model(data)
        preds = logits_cls.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
        if args.visu and count % 5 == 0:
            for i in range(data.shape[0]):
                np.save('/opt/data/private/ckpt/scan/%s/visu/node0_%04d.npy' % (args.exp_name, count*args.test_batch_size+i), data[i, :, :].detach().cpu().numpy())
                np.save('/opt/data/private/ckpt/scan/%s/visu/node1_%04d.npy' % (args.exp_name, count*args.test_batch_size+i), node1[i, :, :].detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
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
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--visu', type=bool, default=False,
                        help='visualize atp by saving nodes')
    args = parser.parse_args()

    _init_()

    io = IOStream('/opt/data/private/ckpt/scan/' + args.exp_name + '/run.log')
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