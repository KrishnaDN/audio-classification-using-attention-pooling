#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 19:46:03 2020

@author: Krishna
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules_funs import ConvBlock, LinearAttentionBlock, ProjectorBlock
from initialize import *



class AttnPooling(nn.Module):
    def __init__(self, num_classes=1, attention=True, normalize_attn=True, init='xavierUniform'):
        super(AttnPooling, self).__init__()
        self.attention = attention        
        self.conv_block1 = ConvBlock(513, 256, 1)
        self.conv_block2 = ConvBlock(256, 128, 2)
        self.conv_block3 = ConvBlock(128, 80, 2)
        self.conv_block4 = ConvBlock(80, 64, 2)
        
        self.avgpooling = nn.AvgPool1d(kernel_size=75, stride=1, padding=0)
        self.classify = nn.Linear(in_features=64, out_features=num_classes, bias=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,inputs):
        
        x = self.conv_block1(inputs)
        x = self.conv_block2(x)
        x = F.max_pool1d(self.conv_block3(x), kernel_size=2, stride=2, padding=0)
        x = F.max_pool1d(self.conv_block4(x), kernel_size=2, stride=2, padding=0)
        x = self.avgpooling(x) 
        # classification layer
        x = self.sigmoid(self.classify(x))
        return x


