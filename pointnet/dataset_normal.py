from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
from plyfile import PlyData, PlyElement


class NormalDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.idfile = os.path.join(self.root, 'object_list.txt')
        self.data_augmentation = data_augmentation

        with open(self.idfile, 'r') as f:
            self.objects = f.readlines()
            
    def __getitem__(self, index):
        objname_no_suffix = self.objects[index][:-1]

        point_set = np.loadtxt(
            os.path.join(self.root, objname_no_suffix + '.pts'),
            dtype=np.float32)
        vnormal = np.loadtxt(
            os.path.join(self.root, objname_no_suffix + '.normal'),
            dtype=np.float32)

        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]
        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        vnormal = vnormal[choice]

        # original pointnet augmentations include rotation, we do not implement yet
        if self.data_augmentation:
            #theta = np.random.uniform(0,np.pi*2)
            #rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            #point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        point_set = torch.from_numpy(point_set)
        vnormal = torch.from_numpy(vnormal)
        
        return point_set, vnormal

    def __len__(self):
        return len(self.objects)