from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


for i in range(10):

    try:
        z = 3/(i-3)
    except:
        continue
    else: 
        print(z)
    print(i)

    
