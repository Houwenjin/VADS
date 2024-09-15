#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 12:03:15 2019

@author: war-machince
"""

import os,sys
pwd = os.getcwd()
sys.path.insert(0,pwd)
#%%
print('-'*30)
print(os.getcwd())
print('-'*30)
#%%
import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
# import torchvision.models.resnet as models
from PIL import Image
import h5py
import numpy as np
import scipy.io as sio
import pickle
from transformers import ViTFeatureExtractor, ViTModel


NFS_path = '/projects/'


#%%
#import pdb
#%%
idx_GPU = 0
is_save = True
#%%
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
#%%
img_dir = os.path.join(NFS_path,'data/CUB/')
file_paths = os.path.join(NFS_path,'data/xlsa17/data/CUB/res101.mat')
save_path = os.path.join(NFS_path,'data/CUB/feature_map_VIT_101_CUB.hdf5')
attribute_path = '/home/shchen/projects/zsl/Baseline1_DAZLE/w2v/CUB_attribute.pkl'
#pdb.set_trace()
model_name = "vit"

# Batch size for training (change depending on how much memory you have)
batch_size = 64

device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")
#%%

model_ref = ViTModel.from_pretrained('google/vit-base-patch16-224')
model_ref.eval()

model_f = nn.Sequential(*list(model_ref.children())[:-2])
model_f.to(device)
model_f.eval()

for param in model_f.parameters():
    param.requires_grad = False
#%%
class CustomedDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, img_dir , file_paths, transform=None):
        self.matcontent = sio.loadmat(file_paths)
        self.image_files = np.squeeze(self.matcontent['image_files'])
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx][0]
        image_file = os.path.join(self.img_dir,
                                  '/'.join(image_file.split('/')[6:]))
        image = Image.open(image_file)
        if image.mode == 'L':
            image=image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

#%%
input_size = 224
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    
CUBDataset = CustomedDataset(img_dir , file_paths, feature_extractor)
dataset_loader = torch.utils.data.DataLoader(CUBDataset,
                                             batch_size=batch_size, shuffle=False,
                                             num_workers=4)
#%%
#with torch.no_grad():
all_features = []
for i_batch, imgs in enumerate(dataset_loader):
    print(i_batch)
    imgs=imgs['pixel_values'][0].to(device)
    features = model_f(imgs).last_hidden_state
    all_features.append(features.cpu().numpy())
all_features = np.concatenate(all_features,axis=0)
#%% get remaining metadata
matcontent = CUBDataset.matcontent
# all_features = matcontent['features'].T
labels = matcontent['labels'].astype(int).squeeze() - 1

split_path = os.path.join(NFS_path,'data/xlsa17/data/CUB/att_splits.mat')
matcontent = sio.loadmat(split_path)
trainval_loc = matcontent['trainval_loc'].squeeze() - 1
#train_loc = matcontent['train_loc'].squeeze() - 1 #--> train_feature = TRAIN SEEN
#val_unseen_loc = matcontent['val_loc'].squeeze() - 1 #--> test_unseen_feature = TEST UNSEEN
test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
att = matcontent['att'].T
original_att = matcontent['original_att'].T
#%% construct attribute w2v
with open(attribute_path,'rb') as f:
    w2v_att = pickle.load(f)
assert w2v_att.shape == (312,300)
print('save w2v_att')
#%%
if is_save:
    f = h5py.File(save_path, "w")
    f.create_dataset('feature_map', data=all_features)
    f.create_dataset('labels', data=labels)
    f.create_dataset('trainval_loc', data=trainval_loc)
    f.create_dataset('test_seen_loc', data=test_seen_loc)
    f.create_dataset('test_unseen_loc', data=test_unseen_loc)
    f.create_dataset('att', data=att)
    f.create_dataset('original_att', data=original_att)
    f.create_dataset('w2v_att', data=w2v_att)
    f.close()
