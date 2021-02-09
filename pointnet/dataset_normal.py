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
import h5py
from torch_geometric.nn import fps
from open3d import *

class NormalDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=4096,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.data_augmentation = data_augmentation

        if split == 'train':
            self.idfile = os.path.join(self.root, 'train_split.txt')
        else:
            self.idfile = os.path.join(self.root, 'valid_split.txt')

        with open(self.idfile, 'r') as f:
            self.objects = f.readlines()
            
    def __getitem__(self, index):
        objname_no_suffix = self.objects[index][:-1]
        #print('[HERE: In pointnet.pytorch.pointnet.dataset_normal] Getting object %d: %s' % (index, objname_no_suffix))

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

class NormalDatasetAllInOne(data.Dataset):
    def __init__(self,
                 root,
                 npoints=4096,
                 split='train',
                 data_augmentation=True,
                 copy_len=None):
        self.npoints = npoints
        self.root = root
        self.dataset_file = 'thingi10k_all_in_one.hdf5'
        self.data_augmentation = data_augmentation
        self.copy_len = copy_len

        if split == 'train':
            self.idfile = os.path.join(self.root, 'train_split.txt')
        else:
            self.idfile = os.path.join(self.root, 'valid_split.txt')

        with open(self.idfile, 'r') as f:
            self.objects = f.read().splitlines()

        self.data = h5py.File(os.path.join(self.root, self.dataset_file), 'r')
            
    def __getitem__(self, index):
        if self.copy_len is not None:
            index = index % len(self.objects)
        objname = self.objects[index]
        #print('[HERE: In pointnet.pytorch.pointnet.dataset_normal] Getting object %d: %s' % (index, objname_no_suffix))

        #print(f'[HERE In pointnet.pytorch.pointnet.dataset_normal] Trying to access {objname}')
        point_set = self.data[objname + '_pts'][:]
        vnormal = self.data[objname + '_normal'][:]

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
        if self.copy_len is None:
            return len(self.objects)
        return self.copy_len

class NormalDatasetAllInOneFPS(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2048,
                 split='train',
                 data_augmentation=True,
                 copy_len=None):
        self.npoints = npoints
        self.root = root
        self.dataset_file = 'thingi10k_all_in_one.hdf5'
        self.data_augmentation = data_augmentation
        self.copy_len = copy_len

        if split == 'train':
            self.idfile = os.path.join(self.root, 'train_split.txt')
        else:
            self.idfile = os.path.join(self.root, 'valid_split.txt')

        with open(self.idfile, 'r') as f:
            self.objects = f.read().splitlines()

        self.data = h5py.File(os.path.join(self.root, self.dataset_file), 'r')
            
    def __getitem__(self, index):
        if self.copy_len is not None:
            index = index % len(self.objects)
        objname = self.objects[index]
        #print('[HERE: In pointnet.pytorch.pointnet.dataset_normal] Getting object %d: %s' % (index, objname_no_suffix))

        #print(f'[HERE In pointnet.pytorch.pointnet.dataset_normal] Trying to access {objname}')
        point_set = self.data[objname + '_pts'][:]
        vnormal = self.data[objname + '_normal'][:]

        N_pts = point_set.shape[0]
        # maybe change this to offline sampling
        fps_sample_index = fps(torch.from_numpy(point_set), ratio=self.npoints/N_pts)
        if self.npoints/N_pts > 1:
            print(f'[HERE: In pointnet.pytorch.pointnet.dataset_normal.NormalDatasetAllInOneFPS] Sample ratio {self.npoints/N_pts: .4f} = {self.npoints}/{N_pts} larger than 1!')
        if len(fps_sample_index) != self.npoints:
            print(f'[HERE: In pointnet.pytorch.pointnet.dataset_normal.NormalDatasetAllInOneFPS] Resulting sample number {len(fps_sample_index)} is different than designated: {self.npoints}!')

        #choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        #resample
        #point_set = point_set[choice, :]
        point_set = point_set[fps_sample_index]
        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        draw_geometries([point_set])

        #vnormal = vnormal[choice]
        vnormal = vnormal[fps_sample_index]

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
        if self.copy_len is None:
            return len(self.objects)
        return self.copy_len
