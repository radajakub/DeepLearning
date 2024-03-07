#%%
import torch
import torch.nn as nn
import numpy as np
from model import FFModel
from template import *
import math

import matplotlib
import matplotlib.pyplot as plt

#!%load_ext autoreload
#!autoreload 2
#!%matplotlib inline


class Predictor():
    """ Wrapper class to convert tensors to numpy and interface with G2Model, e.g. to call
        G2Model.plot_boundary()    
    """

    def __init__(self, name, model):
        self.name = name
        self.model = model

    def score(self, x):
        if isinstance(x,np.ndarray):
            x = torch.tensor(x).to(self.model.w1)
        return self.model.score(x).detach().numpy()

    def classify(self, x_i):
        scores = self.score(x_i)
        return np.sign(scores)


#%%
# make experiment reproducible by fixing the seed for all random generators
torch.manual_seed(1)
# general praparations
smpl_size = 200
vsmpl_size = 1000
tsmpl_size = 1000
hdim = 500
lrate = 1.0e-1
dtype = torch.float64
name = "model_{}".format(hdim) 

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# get training/validation data
# transform them to PyTorch tensors
gmx = G2Model()

def generate_data(sample_size):
    x, t = gmx.generate_sample(sample_size)
    x = torch.tensor(x, dtype=dtype).to(device)
    t = torch.tensor(t, dtype=torch.int).to(device)
    return x,t

x, t = generate_data(smpl_size)
xv, tv = generate_data(vsmpl_size)

# model
model = FFModel(hdim, device, dtype)

#%% Gradient check
niterations = 1000
log_period = 10
print('# Gradient checks', flush=True)

#%% Training
print('# Starting', flush=True)
for count in range(niterations):
    # compute loss
    l = model.mean_loss(x, t)
    # compute gradient
    model.zero_grad()
    l.backward()
    # make a gradinet descent step
    for p in model.parameters:
        p.data -= lrate * p.grad.data
    # evaluate and print
    if (count % log_period == log_period-1) or (count == niterations-1):
        with torch.no_grad():
            vacc = model.mean_accuracy(xv, tv)
        print(f'epoch: {count}  loss: {l.item():.4f} vacc: {vacc:.4f}', flush=True)

#%% 
# Plot predictor
pred = Predictor(name, model)
gmx.plot_boundary((x, t), pred)

#%% 
# Test
# True test risk distribution
#%% Confidence intervals and plots
# Chebyshev
# Hoeffding
# Plot
