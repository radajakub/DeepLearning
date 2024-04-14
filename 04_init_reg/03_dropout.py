# %%
import os
from copy import deepcopy
import numpy as np
import torch

import matplotlib.pyplot as plt

from datasets import MNISTData
from models import MNISTNetwork, MNISTEnsemble

from helpers import select_device
# %%
OUT_DIR = os.path.join('out', '03_dropout')
os.makedirs(OUT_DIR, exist_ok=True)

# %%
dev = select_device()

# %%


def evaluate(data: MNISTData, model: MNISTNetwork, device: torch.device, dataset: str = 'train'):
    if dataset == 'train':
        print('train loader')
        loader = data.train_loader
        size = len(data.train_subset)
    elif dataset == 'val':
        print('val loader')
        loader = data.val_loader
        size = len(data.val_subset)
    else:
        print('test loader')
        loader = data.test_loader
        size = len(data.test_set)

    model.eval()

    loss = 0
    error = 0
    for (x, t) in loader:
        x, t = x.to(device), t.to(device)
        x = torch.flatten(x.squeeze(1), start_dim=1)
        with torch.no_grad():
            probs = model(x)
            loss += model.loss(probs, t).item()
            error += model.error(model.classify(probs), t)
    error /= size

    model.train()
    return loss, error


def train(data: MNISTData, p: float, epochs: int, lr=float, device: torch.device = 'cpu') -> MNISTNetwork:
    model = MNISTNetwork(dropout_p=p).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_stats = np.zeros((epochs, 2))
    val_stats = np.zeros((epochs, 2))

    curr_best = np.inf
    best_model = None

    for e in range(epochs):
        # train on every datapoint in loader
        for (x, t) in data.train_loader:
            x, t = x.to(device), t.to(device)
            # flatten x from shape (batch, channels, height, width) to (batch, features)
            x = torch.flatten(x.squeeze(1), start_dim=1)
            optimizer.zero_grad()
            probs = model(x)
            l = model.loss(probs, t)
            l.backward()
            optimizer.step()

        # evaluation
        t_loss, t_err = evaluate(data, model, device, 'train')
        v_loss, v_err = evaluate(data, model, device, 'val')

        train_stats[e, 0] = t_loss
        train_stats[e, 1] = t_err
        val_stats[e, 0] = v_loss
        val_stats[e, 1] = v_err

        if v_err < curr_best:
            curr_best = v_err
            best_model = deepcopy(model.state_dict())

        print(f'e {e} t_loss {t_loss} t_err {t_err} v_loss {v_loss} v_err {v_err}')

    model.load_state_dict(best_model)

    return model, train_stats, val_stats

# %%


learning_rate = 3e-4
epochs = 200
batch_size = 8

# %%
data = MNISTData(batch_size=batch_size)

# %%
dropout_model, dropout_train_stats, dropout_val_stats = train(data, 0.5, epochs, learning_rate, dev)
# %%
dropout_test_loss, dropout_test_err = evaluate(data, dropout_model, dev, 'test')
print(dropout_test_loss)
print(dropout_test_err)

# %%

# %%
model, train_stats, val_stats = train(data, 0, epochs, learning_rate, dev)

# %%
test_loss, test_err = evaluate(data, model, dev, 'test')
print(test_loss)
print(test_err)

# %%
torch.save(dropout_model.state_dict(), 'dropout_model.pth')
torch.save(model.state_dict(), 'model.pth')
np.save('dropout_train_stats.npy', dropout_train_stats)
np.save('dropout_val_stats.npy', dropout_val_stats)
np.save('train_stats.npy', train_stats)
np.save('val_stats.npy', val_stats)

# %%
xs = np.arange(1, epochs + 1)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(xs, dropout_train_stats[:, 0], color='tab:blue', linestyle='-', label='Dropout training loss')
ax.plot(xs, dropout_val_stats[:, 0], color='tab:blue', linestyle='--', label='Dropout validation loss')
ax.plot(xs, train_stats[:, 0], color='tab:orange', linestyle='-', label='No dropout training loss')
ax.plot(xs, val_stats[:, 0], color='tab:orange', linestyle='--', label='No dropout validation loss')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.legend()
ax.grid()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'loss.png'))

# %%

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(xs, dropout_train_stats[:, 1], color='tab:blue', linestyle='-', label='Dropout training error')
ax.plot(xs, dropout_val_stats[:, 1], color='tab:blue', linestyle='--', label='Dropout validation error')
ax.plot(xs, train_stats[:, 1], color='tab:orange', linestyle='-', label='No dropout training error')
ax.plot(xs, val_stats[:, 1], color='tab:orange', linestyle='--', label='No dropout validation error')
ax.set_xlabel('Epochs')
ax.set_ylabel('Error')
ax.legend()
ax.grid()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'error.png'))

# %%

# try ensembles of dropout models
ensemble_model = MNISTEnsemble(0.5).to(dev)
ensemble_model.load_state_dict(deepcopy(dropout_model.state_dict()))
# ensemble_model.load_state_dict(torch.load(os.path.join(OUT_DIR, 'dropout_model.pth')))

ensemble_model.train()

# %%
sizes = [10, 50, 100, 500, 1000, 2000]  # , 5000, 10000]

# %%
for s in sizes:
    # average scores of the network over size evaluations
    error = 0
    for (x, t) in data.test_loader:
        x, t = x.to(dev), t.to(dev)
        x = torch.flatten(x.squeeze(1), start_dim=1)

        scores = torch.zeros((x.shape[0], 10)).to(dev)
        for i in range(s):
            with torch.no_grad():
                scores += ensemble_model(x)
        scores /= s

        # average the scores
        y = scores.softmax(dim=1).argmax(dim=1)
        error += ensemble_model.error(y, t)
    error /= len(data.test_set)
    print(f'{s}: {error}')
