# %%
import os

import numpy as np

import torch
import torch.nn.functional as F

from helpers import select_device
from models import ShallowNetwork
from progress import Progress
from datasets import CircleDataGenerator
from plotting import plot_decision_boundary, plot_progresses

# %%


def train(data: tuple[torch.tensor], val_data: tuple[torch.tensor], l1_dim: int, l2_dim: int, lr: float, epochs: int, device: torch.device) -> tuple[ShallowNetwork, Progress]:
    x, t = data

    model = ShallowNetwork(in_dim=2, l1_dim=l1_dim, l2_dim=l2_dim, out_dim=2).to(device)

    progress = Progress(epochs, num_layers=3)

    # investigate initial loss
    progress.initial(model, x, t)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for e in range(epochs):
        y = model(x)
        l = F.cross_entropy(y, t)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        progress.log(model, e, l.item(), data, val_data)

    # load the best model based on validation accuracy
    model.load_state_dict(progress.best_model)

    return model, progress


def normalize_data(x: torch.tensor) -> torch.tensor:
    return (x - x.mean(dim=0)) / x.std(dim=0)


# %%
OUT_DIR = os.path.join('out', '02_shallow')
os.makedirs(OUT_DIR, exist_ok=True)

# %%
dev = select_device()

# %%
np_seed = 0
torch_seed = 0

# as suggested in the assignment
first_dim = 6
second_dim = 3

learning_rate = 3e-4
dataset_size = 100
validation_size = 50

epochs = 10000

# %%
np.random.seed(np_seed)
torch.manual_seed(torch_seed)

generator = CircleDataGenerator()
x, t = generator.generate_sample(dataset_size)
x_val, t_val = generator.generate_sample(validation_size)

# the dataset is small so we can move them to dev at once
x = x.to(dev)
t = t.to(dev)
x_val = x_val.to(dev)
t_val = t_val.to(dev)

# %%
model, progress = train((x, t), (x_val, t_val), l1_dim=first_dim, l2_dim=second_dim, lr=learning_rate, epochs=epochs, device=dev)
plot_decision_boundary(x, t, model, dev, outname=os.path.join(OUT_DIR, 'original.png'))

# %%
# normalize the data
norm_x = normalize_data(x)
norm_x_val = normalize_data(x_val)

norm_model, norm_progress = train((norm_x, t), (norm_x_val, t_val), l1_dim=first_dim, l2_dim=second_dim, lr=learning_rate, epochs=epochs, device=dev)
plot_decision_boundary(norm_x, t, norm_model, dev, outname=os.path.join(OUT_DIR, 'norm.png'))

# %%
# multiply_data
c = 0.01
mult_x = c * norm_x
mult_x_val = c * norm_x_val

mult_model, mult_progress = train((mult_x, t), (mult_x_val, t_val), l1_dim=first_dim, l2_dim=second_dim, lr=learning_rate, epochs=epochs, device=dev)
plot_decision_boundary(mult_x, t, mult_model, dev, outname=os.path.join(OUT_DIR, 'mult.png'), scale=0.01)

# %%
# plot progresses for analysis
plot_progresses(progress, norm_progress, mult_progress, OUT_DIR)

# %%
