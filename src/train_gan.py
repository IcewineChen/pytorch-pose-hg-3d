import torch
import torch.nn
from models.GAN import Generator,Discriminator
import models.hg_3d as hg_3d

import opts,ref

opt = opts.opts.parse()

def main():
    sym_g = Generator()
    sym_d = Discriminator()
    #