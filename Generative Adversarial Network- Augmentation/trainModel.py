from __future__ import print_function
import os
import torch 
import torch.nn as nn
import numpy as np
from jaccardIndex import Jaccard
from torchvision.utils import save_image
from scipy.io import savemat
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device(dev) 

def train_model(n_epochs, modelG, modelD, trainingLoader, optimG, optimD, imgSize, pathm,trmeanb,trmeang,trmeanr):
    training_losses = []
    nz=100
    criterion = torch.nn.BCELoss()
    #Binary Cross Entropy
    real_label =0.95
    fake_label=0.0
    i=0
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    for epoch in range(n_epochs):
        for trainim,trainmas in trainingLoader :
            modelD.zero_grad()
            real_cpu = trainim.to(device)
            labels_given = trainmas.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = modelD(real_cpu).view(-1) 
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = modelG(noise)
            label.fill_(fake_label)
            output = modelD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()        
            errD = errD_real + errD_fake          
            optimD.step()        
            modelG.zero_grad()
            label.fill_(real_label)  
            output = modelD(fake).view(-1)
            errG = criterion(output, label)         
            errG.backward()
            D_G_z2 = output.mean().item()
         
            optimG.step()
            
            
            if iters % 19 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, n_epochs, i, len(trainingLoader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

         
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            if (iters % 500 == 0) or ((epoch == n_epochs-1) and (i == len(trainingLoader)-1)):
             with torch.no_grad():
                fake = modelG(fixed_noise).detach().cpu()
             img_list.append(fake)

            iters += 1 
        
            
    
    if not os.path.exists(r'filename'):
         os.mkdir(r'filename')
    im_batch_size = 64
    n_images=6000
    for i_batch in range(0, n_images, im_batch_size):
      gen_z = torch.randn(im_batch_size, nz, 1, 1, device=device)
      gen_images = modelG(gen_z)
      images = gen_images.to("cpu").clone().detach()

      images = images.numpy().transpose(0, 2, 3, 1)
      for imm in images :

       np.stack(imm).astype(None)
       mdic = {"inputpatch":imm}
       # sonuçları matlab dosyası olarak kaydediyoruz
       savemat(f'result_{i}.mat',mdic)
       i+=1
      
