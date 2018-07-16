import torch
from torch.autograd import Variable

def discriptor(sample,gt):      #把sample和gt的keypoint一起传进来
    pass 

class Discrminator(torch.nn.Module):
    def __init__(self,data,discriptor): # 预留一部分参数
        self.origin_data = data
        self.discriptor = discriptor

    def forward(self):
        pass