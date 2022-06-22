#!/usr/bin/python
# -*- coding: utf-8 -*-

#Libraries
from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from torchvision.utils import save_image

nz=100
ndf=64
ngf=64
nc=3

class Generator(nn.Module):
    def __init__(self,ngpu):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d( nz, ngf * 8, 7, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
           
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 4, 0, bias=False),
           
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
           
        )

    def forward(self, input):
        return self.main(input)




class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential( 

            nn.Conv2d(
                nc,ndf,4,2,1,bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
          
            nn.Conv2d(ndf,ndf * 2,4,4,0,bias=False,
            ),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 2,ndf * 4,4,2, 1,bias=False,
            ),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
          
            nn.Conv2d(ndf * 4, ndf * 8,4,2,1,bias=False,
            ),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
         
            nn.Conv2d(
                ndf * 8,1,7,1,0,bias=False      
            ),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)
