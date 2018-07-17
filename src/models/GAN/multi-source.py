import torch
import torch.nn as nn
import src.models.hg_3d as hg_3d
from torch.autograd import Variable

def discriptor(sample,gt):      #把sample和gt的keypoint一起传进来
    # 把generator生成的结果和3d ground truth比较，生成一个手工描述子
    # 描述子中包含差值和平方，差值引入方向特征，平方引入距离特征
    # sample和gt都是
    delta = pow(sample-gt,2)
    return [sample[0]-gt[0],sample[1]-gt[1],sample[2]-gt[2],delta[0],delta[1],delta[2]]

class Generator(nn.Module):
    def __init__(self,nStack, nModules, nFeats, nRegModules):      # model直接把之前3dhourglass model传进来
        super(Generator,self).__init__()
        # 记得在main中先load pretrain的model
        # 按照文中流程，先pretrain
        self.model = hg_3d.HourglassNet3D(nStack=nStack,nModules=nModules,nFeats=nFeats,nRegModules=nRegModules)
         
    def forward(self,x):
        # 直接出结果
        return self.model(x)

class Discrminator(nn.Module):
    def __init__(self,data,discriptor,heatmaps,depth,batch_size,embedding_dims,nFeats): # 预留一部分参数
        self.origin_data = data
        self.discriptor = discriptor
        self.heatmap = heatmaps
        self.depth = depth
        self.batch_size = batch_size
        self.embedding_dims = embedding_dims
        self.feats = nFeats
        
        # layers定义
        self.fc1 = nn.Linear(self.batch_size,self.embedding_dims)
        self.fc2 = nn.Linear(self.embedding_dims*3,self.embedding_dims*3)
        self.fc3 = nn.Linear(self.embedding_dims*3,self.embedding_dims*3)
        self.fc4 = nn.Linear(self.embedding_dims*3,1)   # 输出0，1判断real和fake

    def forward(self,x):    # x包
        pass



