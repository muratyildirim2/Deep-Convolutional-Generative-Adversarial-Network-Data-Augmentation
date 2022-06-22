# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:10:19 2022

@author: user
"""

import scipy.io
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt 
import scipy.io
import glob
from PIL import Image
import numpy as np


i=0
plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})


for file in glob.glob(r"matlab files\*.mat") :
 
  images = scipy.io.loadmat(file)
  
  deneme = images['inputpatch']


  


  deneme = deneme.reshape(-1)

  fig =plt.hist(deneme, bins=50)
  plt.gca().set(title='Frequency Histogram1', ylabel='Frequency');
  plt.plot(axes='off')
  
  plt.savefig(r"filename\augmented_data_{}".format(i))
  plt.show()
  i+=1
#%%





