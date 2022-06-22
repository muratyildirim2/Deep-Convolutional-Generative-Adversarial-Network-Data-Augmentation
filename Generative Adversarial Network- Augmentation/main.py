#Libraries 
from __future__ import print_function
import os
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from DCGAN import Generator
from DCGAN import Discriminator
from crossValidation import crossVal
from trainModel import train_model
from datasetTorch import structureData
from datasetLoader import get_images


#Pytorch Device
dev = "cuda:0"  
device = torch.device(dev) 
ngpu=1
trainSetSize=1215  
# 1215 image file      

miniBatchSize = 64      
n_epochs = 300        
learnRate = 0.002
imgSize=224          
        


l = list(range(1215))
trind=l
tsind=l
vlind=l
trmeanb=[]
trmeang=[]
trmeanr=[]

input_images, target_masks,trmeanb,trmeang,trmeanr= get_images(trainSetSize, tsind, trind, vlind)


params = {'batch_size': miniBatchSize, 'shuffle': False}   
training_set = structureData(input_images[trind], target_masks[trind])
# training set içerisinde input images ve onların labelları var.(0-1214)
trainingLoader = DataLoader(training_set, **params)    
validation_set = ""
validationLoader = ""   
test_set = ""
testLoader = ""

Gen = Generator(ngpu).to(device)
Dis = Discriminator(ngpu).to(device)
#Generator ve Discriminator
print(Gen)
# initizalize the conv layers       
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


Gen.apply(init_weights)
Dis.apply(init_weights)                     

# Adam optimizer learning rate = 0.003 beta1=0.5 beta2 = 0.999             
optimG = torch.optim.Adam(Gen.parameters(),learnRate,betas=(0.5, 0.999)) 
optimD = torch.optim.Adam(Dis.parameters(),learnRate,betas=(0.5, 0.999))          
 

pathm = os.getcwd()    

# train model
train_model(n_epochs, Gen, Dis ,trainingLoader, optimG, optimD, imgSize, pathm,trmeanb,trmeang,trmeanr)        

 
