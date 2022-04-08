#!/usr/bin/python3
#coding=utf-8

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean 
        self.std  = std
    
    def __call__(self, image, mask=None, edge=None, illumination=None):
        image = (image - self.mean)/self.std
        if mask is None:
            return image
        return image, mask/255, edge/255, illumination/255

class RandomCrop(object):
    def __call__(self, image, mask=None, edge=None, illumination=None):
        H,W,_   = image.shape
        randw   = np.random.randint(W/8)
        randh   = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
        if mask is None:
            return image[p0:p1,p2:p3, :]
        return image[p0:p1,p2:p3, :], mask[p0:p1,p2:p3], edge[p0:p1,p2:p3], illumination[p0:p1,p2:p3]

class RandomFlip(object):
    def __call__(self, image, mask=None, edge=None, illumination=None):
        if np.random.randint(2)==0:
            if mask is None:
                return image[:,::-1,:].copy()
            return image[:,::-1,:].copy(), mask[:, ::-1].copy(), edge[:, ::-1].copy(), illumination[:, ::-1].copy()
        else:
            if mask is None:
                return image
            return image, mask, edge, illumination

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask=None, edge=None, illumination=None):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if mask is None:
            return image
        mask  = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        edge  = cv2.resize( edge, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        illumination  = cv2.resize( illumination, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask, edge, illumination

class ToTensor(object):
    def __call__(self, image, mask=None, edge=None, illumination=None):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        if mask is None:
            return image
        mask  = torch.from_numpy(mask)
        edge  = torch.from_numpy(edge)
        illumination  = torch.from_numpy(illumination)
        return image, mask, edge, illumination


########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean   = np.array([[[124.55, 118.90, 102.94]]])
        self.std    = np.array([[[ 56.77,  55.97,  57.50]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s'%(k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


########################### Dataset Class ###########################
class Data(Dataset):
    def __init__(self, cfg):
        self.cfg        = cfg
        self.normalize  = Normalize(mean=cfg.mean, std=cfg.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize     = Resize(352, 352)
        self.totensor   = ToTensor()

        with open(cfg.datapath+'/'+cfg.mode+'.txt', 'r') as lines:
            self.samples = []
            for line in lines:
                self.samples.append(line.strip())

    def __getitem__(self, idx):
        name  = self.samples[idx]
        #print(name)
        image = cv2.imread(self.cfg.datapath+'/image/'+name+'.jpg')[:,:,::-1].astype(np.float32)

        if self.cfg.mode=='train':
            mask  = cv2.imread(self.cfg.datapath+'/mask/' +name+'.png', 0).astype(np.float32)
            edge  = cv2.imread(self.cfg.datapath+'/edge/' +name+'.png', 0).astype(np.float32)

            illumination =cv2.imread(self.cfg.datapath+'/illumination/' +name+'.jpg', 0).astype(np.float32)

            image, mask, edge, illumination = self.normalize(image, mask, edge, illumination)
            image, mask, edge, illumination = self.randomcrop(image, mask, edge, illumination)
            image, mask, edge, illumination = self.randomflip(image, mask, edge, illumination)            
            return image, mask, edge, illumination
        else:
            shape = image.shape[:2]
            image = self.normalize(image)
            image = self.resize(image)
            image = self.totensor(image)
            return image, shape, name

    def __len__(self):
        return len(self.samples)
    
    def collate(self, batch):
        size = [224, 256, 288, 320, 352][np.random.randint(0, 5)]
        image, mask, edge, illumination = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i]  = cv2.resize(mask[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            edge[i]  = cv2.resize(edge[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            illumination[i]  = cv2.resize(illumination[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        image  = torch.from_numpy(np.stack(image, axis=0)).permute(0,3,1,2)
        mask   = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        edge   = torch.from_numpy(np.stack(edge, axis=0)).unsqueeze(1)
        illumination   = torch.from_numpy(np.stack(illumination, axis=0)).unsqueeze(1)
        return image, mask, edge, illumination
