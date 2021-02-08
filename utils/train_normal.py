from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset_normal import NormalDataset
from pointnet.model import PointNetDenseNormalPred3DVer, feature_transform_regularizer
from losses.normal_losses import inner_prod_loss, normalization_reg_loss
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=8, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument(
    '--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='normal', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = NormalDataset(
    root=opt.dataset)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

test_dataset = NormalDataset(
    root=opt.dataset,
    split='test',
    data_augmentation=False)
testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))

try:
    os.makedirs(opt.outf)
except OSError:
    pass

blue = lambda x: '\033[94m' + x + '\033[0m'

normal_predictor = PointNetDenseNormalPred3DVer(feature_transform=opt.feature_transform)

if opt.model != '':
    normal_predictor.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(normal_predictor.parameters(), lr=0.0, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
normal_predictor.cuda()

num_batch = len(dataset) / opt.batchSize

for epoch in range(opt.nepoch):
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        points, target = data
        # points: [N, N_pts, 3]

        points = points.transpose(2, 1) # [N, 3, N_pts]
        points, target = points.cuda(), target.cuda()
        
        optimizer.zero_grad()
        normal_predictor = normal_predictor.train()
        
        pred, trans, trans_feat = normal_predictor(points)
        pred = pred.view(-1, 3) # putting batches together, simply for loss, pred: [N*N_pts, 3]
        target = target.view(-1, 3)
        #print(pred.size(), target.size())

        loss_inner_prod, accu_inner_prod = inner_prod_loss(pred, target, accu_thresholds_in_deg=[5, 15, 25])
        loss_norm_reg = normalization_reg_loss(pred)
        loss = loss_inner_prod + loss_norm_reg

        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001

        loss.backward()
        optimizer.step()

        print('[%d: %d/%d] train loss: %6f. loss_inner_prod: %.6f, loss_norm_reg: %.6f, accuracy 5: %6f, 15: %6f, 25: %6f'\
            % (epoch, i, num_batch,
               loss.item(), loss_inner_prod.item(), loss_norm_reg.item(),
               accu_inner_prod[0], accu_inner_prod[1], accu_inner_prod[2]))

        with torch.no_grad():
            if i % 10 == 0:
                j, data = next(enumerate(testdataloader, 0))
                points, target = data
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()

                normal_predictor = normal_predictor.eval()
                pred, _, _ = normal_predictor(points)
                pred = pred.view(-1, 3)
                target = target.view(-1, 3)

                loss_inner_prod, accu_inner_prod = inner_prod_loss(pred, target, accu_thresholds_in_deg=[5, 15, 25])
                loss_norm_reg = normalization_reg_loss(pred)
                loss = loss_inner_prod + loss_norm_reg

                print('[%d: %d/%d] train loss: %6f. loss_inner_prod: %.6f, loss_norm_reg: %.6f, accuracy 5: %6f, 15: %6f, 25: %6f'\
                    % (epoch, i, num_batch,
                    loss.item(), loss_inner_prod.item(), loss_norm_reg.item(),
                    accu_inner_prod[0], accu_inner_prod[1], accu_inner_prod[2]))    

    torch.save(normal_predictor.state_dict(), '%s/normal_model_%d.pth' % (opt.outf, epoch))

"""
## benchmark mIOU
shape_ious = []
for i,data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(2)[1]

    pred_np = pred_choice.cpu().data.numpy()
    target_np = target.cpu().data.numpy() - 1

    for shape_idx in range(target_np.shape[0]):
        parts = range(num_classes)#np.unique(target_np[shape_idx])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            if U == 0:
                iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))

print("mIOU for class {}: {}".format(opt.class_choice, np.mean(shape_ious)))
"""