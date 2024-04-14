# %%
import os

import torch
import torch.nn as nn

from models import DeepNetwork
from plotting import plot_weights

# %%
OUT_DIR = os.path.join('out', '01_deep')
os.makedirs(OUT_DIR, exist_ok=True)


# %%


def experiment(activation: str, init_name: str, seed: int = 0, num_layers: int = 50, layer_size: int = 512, batch_size: int = 256, dev: str = 'cpu'):
    torch.manual_seed(seed)

    model = DeepNetwork(layer_size=layer_size, num_layers=num_layers, activation=activation, init_name=init_name)
    loss = nn.CrossEntropyLoss()

    x = torch.randn(batch_size, layer_size, requires_grad=True)
    y = torch.empty(batch_size, dtype=torch.long).random_(layer_size)
    prediction = loss(model(x), y)
    prediction.backward()
    model.compute_gradient_statistics()

    plot_weights(model, outname=os.path.join(OUT_DIR, f'{activation}_{init_name}.png'))


# %%

seed = 0
num_layers = 50
layer_size = 512
batch_size = 512

# %%
experiment_list = [
    ('tanh', 'original'),
    ('tanh', 'xavier_uniform'),
    ('tanh', 'xavier_normal'),
    ('relu', 'xavier_uniform'),
    ('relu', 'xavier_normal'),
    ('relu', 'kaiming_uniform'),
    ('relu', 'kaiming_normal'),
]

# %%
for activation, init_name in experiment_list:
    experiment(activation, init_name, seed, num_layers, layer_size, batch_size)
