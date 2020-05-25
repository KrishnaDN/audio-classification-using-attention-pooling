#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:45:06 2019

@author: apple
"""

import torch
import numpy as np


from torch.utils.data import DataLoader   
from SpeechDataGenerator import SpeechDataGenerator
import torch.nn as nn
import os
import numpy as np
from torch import optim

from models.CNN_attention_pooling import AttnPooling
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')
#### Dataset info
num_epochs=100
data_path_train='meta/training.txt'
data_path_test = 'meta/testing.txt'
#### Params

def speech_collate(batch):
    targets = []
    specs = []
    for sample in batch:
        specs.append(sample['spec'])
        targets.append((sample['labels']))
    return specs, targets
 
### Data related
dataset_train = SpeechDataGenerator(manifest=data_path_train,mode='train')
dataloader_train = DataLoader(dataset_train, batch_size=32,shuffle=True,collate_fn=speech_collate) 

dataset_test = SpeechDataGenerator(manifest=data_path_test,mode='test')
dataloader_test = DataLoader(dataset_test, batch_size=32,collate_fn=speech_collate)
## Model related
use_cuda = torch.cuda.is_available()
#device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device("cuda:0")
model = AttnPooling(num_classes=1).to(device)
#model.load_state_dict(torch.load('model_checkpoints/check_point_old')['model'])
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-9)
loss = nn.BCELoss()

all_train_loss_vals=[]
for epoch in range(num_epochs):
    model.train()
    train_loss_list = []
    train_acc_list =[]
    total = 0.
    correct = 0.
    for i_batch, sample_batched in enumerate(dataloader_train):
        
    
        #print(sample_batched)
        features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[0]]))
        labels = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[1]]))
        labels = labels.float()
        features, labels = features.to(device), labels.to(device)
        features.requires_grad = True
        optimizer.zero_grad()
        preds,attn_map = model(features)
        total_loss = loss(preds.squeeze(), labels.squeeze())
        #total_loss = loss(preds, labels.squeeze())
        total_loss.backward()
        
        optimizer.step()
        #prediction = np.argmax(preds.detach().cpu().numpy(),axis=1)
        #print(total_loss.item())
        predictions= preds.detach().cpu().numpy()>=0.5
        pred_list=[]
        for item in predictions:
            if item:
                pred_list.append(1.0)
            else:
                pred_list.append(0.0)
        accuracy = accuracy_score(labels.detach().cpu().numpy(),pred_list)
        train_loss_list.append(total_loss.item())
        train_acc_list.append(accuracy)
        if i_batch%100==0:
            print('Loss {} after {} iteration'.format(np.mean(np.asarray(train_loss_list)),i_batch))
        
        
    mean_loss = np.mean(np.asarray(train_loss_list))
    mean_acc = np.mean(np.asarray(train_acc_list))
    all_train_loss_vals.append(mean_loss)
    print('********* Loss {} and Accuracy {} after {} epoch '.format(mean_loss,mean_acc,epoch))
    
    
    
    model.eval()
    cum_acc=0.0
    test_acc_list=[]
    gt_label=[]
    pred_labels=[]
    test_loss_list=[]
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader_test):
            features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[0]]))
            labels = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[1]])) 
            labels=labels.float()
            features, labels = features.to(device), labels.to(device)
            preds,attn_map = model(features)
            total_loss_test = loss(preds.squeeze(), labels.squeeze())
            
            #prediction = np.argmax(preds.detach().cpu().numpy(),axis=1)
            predictions= preds.detach().cpu().numpy()>=0.5
            pred_list=[]
            for item in predictions:
                if item:
                    pred_list.append(1.0)
                else:
                    pred_list.append(0.0)
            #accuracy = accuracy_score(labels.detach().cpu().numpy(),pred_list)
            for item in labels.detach().cpu().numpy():
                gt_label.append(item)
            pred_labels=pred_labels+pred_list
            #accuracy = accuracy_score(labels.detach().cpu().numpy(),np.argmax(preds.detach().cpu().numpy(),axis=1))
            #test_acc_list.append(accuracy)
            test_loss_list.append(total_loss_test.item())
        mean_test_acc = accuracy_score(gt_label,pred_labels)
        mean_loss_test = np.mean(np.asarray(test_loss_list))
        
        model_save_path = os.path.join('model_checkpoint_cnn_attn_pooling', 'best_check_point_'+str(epoch)+'_'+str(mean_test_acc))
        state_dict = {'model': model.state_dict(),'optimizer': optimizer.state_dict(),'epoch': epoch}
        torch.save(state_dict, model_save_path)
    
        
        print('********* Final test accuracy {} and loss {} after {} '.format(mean_test_acc,mean_loss_test,epoch))
        
    
#np.save('single_scale_wavform_losses.npy',all_train_loss_vals)    
    
