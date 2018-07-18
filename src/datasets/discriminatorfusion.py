# -*- coding:utf-8 -*_
import torch.utils.data as data
import numpy as np
import ref
import torch
import cv2
from datasets.mpii import MPII
from datasets.h36m import H36M
from datasets.fusion import Fusion

class DiscriminatorFusion(data.Dataset):
    def __init__(self, opt, split):
        super(DiscriminatorFusion,self).__init__()
        self.origin = Fusion(opt, split)
        # self.{heatmap,depth,discriptor}需要实现
        # depth , heatmap需要从generator部分的2d和3d输出拿
        # 修改hourglass3dnet
        # 然后将几个concat到一起成为self.dataset
        # 之后算一个拼合起来的len，定义为nSamples
        self.nSamples = 0
        
    def __getitem__(self,index):
        # if index < self.nSamples:
        #     return self.dataset[index]
        # else:
        #     raise ValueError("index has been over the length of dataset")
        pass
    
    def __len__(self):
        return self.nSamples