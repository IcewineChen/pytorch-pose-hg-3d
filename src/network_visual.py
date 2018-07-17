import torch
from torch.autograd import Variable
import models.hg_3d as hg_3d
from visualize import make_dot


x=Variable(torch.randn(6,3,256,256)+1)
model = hg_3d.HourglassNet3D(nFeats=256,nModules=2,nRegModules=2,nStack=2)
y=model(x)
# 测流动方向，如果batch按照(6,3,256,256)放置，就用y[index]，因为y是个list,没有grad_fn
# 否则直接y就可以
g=make_dot(y[0])
g=make_dot(y)
g.view()