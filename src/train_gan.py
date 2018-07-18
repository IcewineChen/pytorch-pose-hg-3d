import torch
import torch.nn
from models.GAN.multi_source import Generator,Discriminator
import models.hg_3d as hg_3d
from datasets.fusion import Fusion
import opts,ref

opt = opts.opts().parse()

def main():
    
    generator = Generator(nStack=opt.nStack,nModules=opt.nModules,nFeats=opt.nFeats,nRegModules=opt.nRegModules).cuda()
    discriminator = Discriminator(embedding_dims=opt.embedding_dims).cuda()
    
    optimizer_g = torch.optim.SGD(generator.parameters(),lr=2e-5,momentum=0.99)
    optimizer_d = torch.optim.SGD(discriminator.parameters(),lr=1e-4,momentum=0.9)

    # 定义adversirial loss
    # 暂时先挂MSE，回头看看文章再改
    ad_loss = torch.nn.MSELoss()

    epochs = opt.epochs
    # generator的输入数据，直接就是Fusion类处理的结果
    gen_dataloader = torch.utils.data.DataLoader(
        Fusion(opt, 'train'),
        batch_size = opt.trainBatch,
        shuffle = True if opt.DEBUG == 0 else False,
        num_workers = int(ref.nThreads)
    )

    dataloader = torch.utils.data.DataLoader(
        Fusion(opt, 'train'),
        batch_size = opt.trainBatch,
        shuffle = True,
        num_workers = int(ref.nThreads) 
    )
    for epoch in range(epochs):
        pass
