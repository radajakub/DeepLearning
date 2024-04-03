# %%
from typing import Callable
import os
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision.models
from torchvision import transforms, datasets


import matplotlib.pyplot as plt

# !%load_ext autoreload
# !%autoreload 2
# !%matplotlib inline

print(torch.__version__)

# %% constants
STANDARD_FOLDER = '/local/temporary/Datasets/PACS_cartoon'
FEW_SHOT_FOLDER = '/local/temporary/Datasets/PACS_cartoon_few_shot'

OUTFOLDER = 'out'
os.makedirs(OUTFOLDER, exist_ok=True)

LOGFOLDER = 'logs'
os.makedirs(LOGFOLDER, exist_ok=True)

MODELFOLDER = 'best_models'
os.makedirs(MODELFOLDER, exist_ok=True)

# %%
# Helper Functions


def select_device():
    """
    Find the CUDA device with max available memory and set the global dev variable to it
    If less than 4GB memory on all devices, resort to dev='cpu'
    Repeated calls to the function select the same GPU previously selected
    """
    global dev
    global my_gpu
    if 'my_gpu' in globals() and 'cuda' in str(my_gpu):
        dev = my_gpu
    else:
        # find free GPU
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
        memory_used = np.array([int(x.split()[2]) for x in open('tmp', 'r').readlines()])
        print(memory_used)
        ii = np.arange(len(memory_used))
        mask = memory_used < 4000
        print(mask)
        if mask.any():
            mask_index = np.argmin(memory_used[mask])
            index = (ii[mask])[mask_index]
            my_gpu = torch.device(index)
        else:
            my_gpu = torch.device('cpu')
        dev = my_gpu
    print(dev)

# %%


def compute_statictics(data_loader: torch.utils.data.DataLoader):
    # compute mean
    mean_sum = torch.zeros(3)
    N = 0
    for batch_x, _ in data_loader:
        mean_sum += batch_x.sum(axis=(0, 2, 3))
        N += (batch_x.size(0) * batch_x.size(2) * batch_x.size(3))

    mean = mean_sum / N

    # compute std
    std_sum = torch.zeros(3)
    # increase the dimension of the mean to substract it from every batch point
    tensor_mean = mean.unsqueeze(1).unsqueeze(1).unsqueeze(0)
    for batch_x, _ in data_loader:
        std_sum += (batch_x - tensor_mean).pow(2).sum(axis=(0, 2, 3))

    std = torch.sqrt(std_sum / (N - 1))
    return mean, std


def evaluate(model, loader, switch_off_layers=False):
    """ Evaluate the model with the given dataset loader """

    correct = 0
    total_loss = 0
    N = 0

    if switch_off_layers:
        model.eval()

    with torch.no_grad():
        for (x, t) in loader:
            N += x.size(0)
            x = x.to(dev)
            t = t.to(dev)
            scores = model(x)

            prediction = torch.argmax(scores, dim=1)
            correct += (prediction == t).sum()

            probs = torch.log_softmax(scores, -1)
            total_loss += F.nll_loss(probs, t, reduction='sum').item()

    accuracy = correct / N

    if switch_off_layers:
        model.train()

    return accuracy, total_loss


def tensor_to_img(x):
    img = x.detach().cpu().numpy()
    img = np.transpose(img, axes=(1, 2, 0))
    img = img - img.min()
    img /= img.max()
    return img


def analyze_incorrect(model, loader, model_name, k=5):
    model.eval()

    path = os.path.join(OUTFOLDER, model_name)
    os.makedirs(path, exist_ok=True)
    analyzed = 0

    with torch.no_grad():
        for (x, t) in loader:
            x = x.to(dev)
            t = t.to(dev)
            scores = model(x)
            probs = torch.softmax(scores, -1)
            predictions = torch.argmax(probs, dim=1)
            incorrect = predictions != t
            for i in range(incorrect.shape[0]):
                if incorrect[i]:

                    fig, ax = plt.subplots()
                    ax.imshow(tensor_to_img(x[i, :, :, :]))
                    ax.set_axis_off()

                    title = f'Ground truth class: {t[i]}\nTop 3 predictions with probabilities:'
                    iprobs = probs[i, :]
                    indices = iprobs.argsort(descending=True)
                    for j in range(3):
                        title += f' {indices[j]} ({round(iprobs[indices[j]].item() * 100, 2)}%)'
                    ax.set_title(title)

                    fig.tight_layout()
                    fig.savefig(os.path.join(path, f'error_{analyzed}.png'))

                    analyzed += 1
                    if analyzed == k:
                        return

# %%


class LearningResults:
    def __init__(self, epochs: int, lrs: list[float], dataset_type: str, model_preparation: Callable):
        self.epochs = epochs + 1
        self.lrs = lrs
        self.model_name = f'{dataset_type}_{model_preparation.__name__}'

        grid_shape = (len(lrs), self.epochs)

        self.train_losses = np.zeros(grid_shape)
        self.train_accuracies = np.zeros(grid_shape)
        self.val_losses = np.zeros(grid_shape)
        self.val_accuracies = np.zeros(grid_shape)

        self.best_val_acc = -1
        self.best_epoch = None
        self.best_lr = None

    # return true if model improved
    def record(self, epoch: int, li: int, train_loss: float, train_acc: float, val_loss: float, val_acc: float) -> bool:
        print(f'lr={self.lrs[li]} e={epoch}: train acc {train_acc} | val acc {val_acc}')

        self.train_losses[li, epoch] = train_loss
        self.train_accuracies[li, epoch] = train_acc
        self.val_losses[li, epoch] = val_loss
        self.val_accuracies[li, epoch] = val_acc

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
            self.best_lr = self.lrs[li]
            return True

        return False

    def print_res(self):
        print(f'best validation accuracy: {self.best_val_acc}')
        print(f'best learning rate: {self.best_lr}')
        print(f'best stopping epoch: {self.best_epoch}')

    def plot_stat(self, ax, values, ylabel, title):
        xs = np.arange(self.epochs, dtype=int)
        for li, lr in enumerate(self.lrs):
            ax.plot(xs, values[li, :], label=f'lr={lr}')
        ax.set_xticks([x for x in xs if x % 5 == 0])
        ax.set_xlabel('Epochs')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.set_title(title)

    def plot(self):
        fig, axes = plt.subplots(2, 2)
        self.plot_stat(axes[0, 0], self.train_accuracies, ylabel='Accuracy', title='Training Accuracy')
        self.plot_stat(axes[0, 1], self.val_accuracies, ylabel='Accuracy', title='Validation Accuracy')
        self.plot_stat(axes[1, 0], self.train_losses, ylabel='Loss', title='Training Loss')
        self.plot_stat(axes[1, 1], self.val_losses, ylabel='Loss', title='Validation Loss')
        fig.set_size_inches(18.5, 10.5)
        fig.tight_layout()
        fig.savefig(f'{OUTFOLDER}/{self.model_name}.png', dpi=100)


# %% ######### DATA LOADING AND PREPROCESSING #########


def preporcess_data(folder, batch_size: int = 8):
    # load the traning data to compute statistics
    train_data = datasets.ImageFolder(f'{folder}/train', transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

    print(f'training dataset size: {len(train_data)}')

    # calculate the statistics of the whole set incrementally by minibatches
    mean, std = compute_statictics(train_loader)
    print(f'mean: {mean}')
    print(f'std: {std}')

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    train_data = datasets.ImageFolder(f'{folder}/train', transform)
    test_data = datasets.ImageFolder(f'{folder}/test', transform)
    return train_data, test_data


def split_data(data, validation_ratio=0.2, batch_size=8, seed=42):
    torch.manual_seed(seed)

    m = len(data)
    shuffled_indices = torch.randperm(m).detach().numpy()
    split_idx = int(m * validation_ratio)

    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, num_workers=0,
                                               sampler=torch.utils.data.SubsetRandomSampler(shuffled_indices[split_idx:]))
    val_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, num_workers=0,
                                             sampler=torch.utils.data.SubsetRandomSampler(shuffled_indices[:split_idx]))

    return train_loader, val_loader

# %% model preparation functions


def full_model():
    model = torchvision.models.resnet18(weights=False)
    # remove old classifier head
    del model.fc

    # create new linear layer and intialize it
    linear_layer = nn.Linear(in_features=512, out_features=7)
    nn.init.xavier_normal(linear_layer.weight)

    # set it as a new classifier head
    model.fc = linear_layer

    model.train()

    return model


def fine_tuning():
    model = torchvision.models.resnet18(weights=True)

    # freeze all weights
    for param in model.parameters():
        param.requires_grad = False

    # freeze batch-norm -> DO NOT TURN IT BACK ON IN TRAINING
    model.eval()

    del model.fc

    # create new linear layer and intialize it
    linear_layer = nn.Linear(in_features=512, out_features=7)
    nn.init.xavier_normal(linear_layer.weight)

    # set it as a new classifier head
    model.fc = linear_layer

    # new linear layer is trainable by default

    return model


def full_fine_tuning():
    model = torchvision.models.resnet18(weights=True)

    del model.fc

    # create new linear layer and intialize it
    linear_layer = nn.Linear(in_features=512, out_features=7)
    nn.init.xavier_normal(linear_layer.weight)

    # set it as a new classifier head
    model.fc = linear_layer

    model.train()

    return model

# %%


def cross_validate(dataset_type: str, model_preparation: Callable, epochs: int, lrs: list[float], freeze_layers_for_eval=False, transform=None):
    # preapre containers for data for plotting
    results = LearningResults(epochs=epochs, lrs=lrs, dataset_type=dataset_type, model_preparation=model_preparation)

    # crossvalidate over learning rates
    for li, lr in enumerate(lrs):
        # we should prepare model anew so that we do not train it from scratch
        model = model_preparation()
        model.to(dev)

        # evaluate in epoch zero
        train_accuracy, train_loss = evaluate(model, train_loader, switch_off_layers=freeze_layers_for_eval)
        val_accuracy, val_loss = evaluate(model, val_loader, switch_off_layers=freeze_layers_for_eval)
        results.record(0, li, train_loss, train_accuracy, val_loss, val_accuracy)

        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        # train for full epochs
        for epoch in range(1, epochs + 1):
            # go over the data once in each epoch
            for (x, t) in train_loader:
                x = x.to(dev)
                t = t.to(dev)
                if transform is not None and epoch > 1:
                    x = transform(x)
                score = model(x)
                log_p = torch.log_softmax(score, -1)
                l = F.nll_loss(log_p, t, reduction='sum')
                optimizer.zero_grad()
                l.backward()
                optimizer.step()

            # evaluate both on training and validation sets
            train_accuracy, train_loss = evaluate(model, train_loader, switch_off_layers=freeze_layers_for_eval)
            val_accuracy, val_loss = evaluate(model, val_loader, switch_off_layers=freeze_layers_for_eval)

            # save them for further processing and get improvement
            improved = results.record(epoch, li, train_loss, train_accuracy, val_loss, val_accuracy)

            if improved:
                # save best model to folder
                torch.save(model.state_dict(), f'{MODELFOLDER}/{results.model_name}.pt')

    return results


# %% set training constants
epochs = 50
lrs = [0.01, 0.03, 0.001, 0.003, 0.0001]
errors_to_show = 5
validation_ratio = 0.2
batch_size = 8
seed = 42

# %% select GPU
select_device()

# %% STANDARD

# %% ######### TRAINING FROM SCRATCH #########

print('STANDARD FROM SCRATCH')

# %% prepare data
# load and preprocess data
train_data, test_data = preporcess_data(STANDARD_FOLDER, batch_size=batch_size)

# split training data into train and validation sets
train_loader, val_loader = split_data(train_data, validation_ratio=validation_ratio, batch_size=batch_size, seed=seed)

# %% perform cross validation

results = cross_validate(dataset_type='standard', model_preparation=full_model, epochs=epochs, lrs=lrs, freeze_layers_for_eval=True)

# %% print summary
results.print_res()
results.plot()

# %% load the best saved model
best_full_model = full_model()

best_full_model.load_state_dict(torch.load(f'{MODELFOLDER}/{results.model_name}.pt'))
best_full_model.to(dev)
best_full_model.eval()

# %% evaluate test error of the best model

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_accuracy, test_loss = evaluate(best_full_model, test_loader, switch_off_layers=True)
print(f'test accuracy: {test_accuracy}')
print(f'test loss: {test_loss}')

# %% analyze few cases of incorrectly classified images

analyze_incorrect(best_full_model, test_loader, results.model_name, k=errors_to_show)

# %% ######### FINE TUNING #########

print('STANDARD FINE TUNING')

# %% prepare data
# load and preprocess data
train_data, test_data = preporcess_data(STANDARD_FOLDER, batch_size=batch_size)

# split training data into train and validation sets with the same seed as before
train_loader, val_loader = split_data(train_data, validation_ratio=validation_ratio, batch_size=batch_size, seed=seed)

# %%

results = cross_validate(dataset_type='standard', model_preparation=fine_tuning, epochs=epochs, lrs=lrs, freeze_layers_for_eval=False)

# %% print summary
results.print_res()
results.plot()

# %% load the best saved model
best_fine_model = fine_tuning()
best_fine_model.load_state_dict(torch.load(f'{MODELFOLDER}/{results.model_name}.pt'))
best_fine_model.to(dev)

# %% evaluate test error of the best model

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)
# the layers are switched off by default in this model
test_accuracy, test_loss = evaluate(best_fine_model, test_loader, switch_off_layers=False)
print(f'test accuracy: {test_accuracy}')
print(f'test loss: {test_loss}')

# %% analyze few cases of incorrectly classified images

analyze_incorrect(best_fine_model, test_loader, results.model_name, k=errors_to_show)

# %% ######### FULL FINE TUNING #########

print('STANDARD FULL FINE TUNING')

# %% prepare data
# load and preprocess data
train_data, test_data = preporcess_data(STANDARD_FOLDER, batch_size=batch_size)

# split training data into train and validation sets with the same seed as before
train_loader, val_loader = split_data(train_data, validation_ratio=validation_ratio, batch_size=batch_size, seed=seed)

# %%

results = cross_validate(dataset_type='standard', model_preparation=full_fine_tuning, epochs=epochs, lrs=lrs, freeze_layers_for_eval=True)

# %% print summary
results.print_res()
results.plot()

# %% load the best saved model
best_full_fine_model = full_fine_tuning()
best_full_fine_model.load_state_dict(torch.load(f'{MODELFOLDER}/{results.model_name}.pt'))
best_full_fine_model.to(dev)

# %% evaluate test error of the best model

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)
# the layers are switched off by default in this model
test_accuracy, test_loss = evaluate(best_full_fine_model, test_loader, switch_off_layers=True)
print(f'test accuracy: {test_accuracy}')
print(f'test loss: {test_loss}')

# %% analyze few cases of incorrectly classified images

analyze_incorrect(best_fine_model, test_loader, results.model_name, k=errors_to_show)

# %% #############################################################

# %% FEW SHOT DATASET

# %% ######### TRAINING FROM SCRATCH #########

print('FEW SHOT FROM SCRATCH')

# %% prepare data
# load and preprocess data
train_data, test_data = preporcess_data(FEW_SHOT_FOLDER, batch_size=batch_size)

# split training data into train and validation sets
train_loader, val_loader = split_data(train_data, validation_ratio=validation_ratio, batch_size=batch_size, seed=seed)

# %% perform cross validation

results = cross_validate(dataset_type='few_shot', model_preparation=full_model, epochs=epochs, lrs=lrs, freeze_layers_for_eval=True)

# %% print summary
results.print_res()
results.plot()

# %% load the best saved model
best_full_model = full_model()

best_full_model.load_state_dict(torch.load(f'{MODELFOLDER}/{results.model_name}.pt'))
best_full_model.to(dev)
best_full_model.eval()

# %% evaluate test error of the best model

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_accuracy, test_loss = evaluate(best_full_model, test_loader, switch_off_layers=True)
print(f'test accuracy: {test_accuracy}')
print(f'test loss: {test_loss}')

# %% analyze few cases of incorrectly classified images

analyze_incorrect(best_full_model, test_loader, results.model_name, k=errors_to_show)

# %% ######### FINE TUNING #########

print('FEW SHOT FINE TUNING')

# %% prepare data
# load and preprocess data
train_data, test_data = preporcess_data(FEW_SHOT_FOLDER, batch_size=batch_size)

# split training data into train and validation sets with the same seed as before
train_loader, val_loader = split_data(train_data, validation_ratio=validation_ratio, batch_size=batch_size, seed=seed)

# %%

results = cross_validate(dataset_type='few_shot', model_preparation=fine_tuning, epochs=epochs, lrs=lrs, freeze_layers_for_eval=False)

# %% print summary
results.print_res()
results.plot()

# %% load the best saved model
best_fine_model = fine_tuning()
best_fine_model.load_state_dict(torch.load(f'{MODELFOLDER}/{results.model_name}.pt'))
best_fine_model.to(dev)

# %% evaluate test error of the best model

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)
# the layers are switched off by default in this model
test_accuracy, test_loss = evaluate(best_fine_model, test_loader, switch_off_layers=False)
print(f'test accuracy: {test_accuracy}')
print(f'test loss: {test_loss}')

# %% analyze few cases of incorrectly classified images

analyze_incorrect(best_fine_model, test_loader, results.model_name, k=errors_to_show)

# %% ######### FULL FINE TUNING #########

print('FEW SHOT FULL FINE TUNING')

# %% prepare data
# load and preprocess data
train_data, test_data = preporcess_data(FEW_SHOT_FOLDER, batch_size=batch_size)

# split training data into train and validation sets with the same seed as before
train_loader, val_loader = split_data(train_data, validation_ratio=validation_ratio, batch_size=batch_size, seed=seed)

# %%

results = cross_validate(dataset_type='few_shot', model_preparation=full_fine_tuning, epochs=epochs, lrs=lrs, freeze_layers_for_eval=True)

# %% print summary
results.print_res()
results.plot()

# %% load the best saved model
best_full_fine_model = full_fine_tuning()
best_full_fine_model.load_state_dict(torch.load(f'{MODELFOLDER}/{results.model_name}.pt'))
best_full_fine_model.to(dev)

# %% evaluate test error of the best model

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)
# the layers are switched off by default in this model
test_accuracy, test_loss = evaluate(best_full_fine_model, test_loader, switch_off_layers=True)
print(f'test accuracy: {test_accuracy}')
print(f'test loss: {test_loss}')

# %% analyze few cases of incorrectly classified images

analyze_incorrect(best_fine_model, test_loader, results.model_name, k=errors_to_show)


# %% FEW SHOT AUGMENTED FULL FINE TUNING

print('FEW SHOT AUGMENTED FULL FINE TUNING')

# %% prepare data
# load and preprocess data
train_data, test_data = preporcess_data(FEW_SHOT_FOLDER, batch_size=batch_size)

# split training data into train and validation sets with the same seed as before
train_loader, val_loader = split_data(train_data, validation_ratio=0.3, batch_size=1, seed=seed)

# %% prepare transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.RandomAffine(10),
    transforms.RandomCrop(200),
    # transforms.RandomPerspective(p=0.2),
    transforms.RandomAdjustSharpness(2, p=0.1),
    # transforms.ColorJitter(hue=0.3, brightness=0.2),
    transforms.RandomGrayscale(p=0.1),
    # transforms.RandomInvert(p=0.1),
])

# %%

results = cross_validate(dataset_type='few_shot_augmented', model_preparation=full_fine_tuning,
                         epochs=epochs, lrs=lrs, freeze_layers_for_eval=True, transform=transform)

# %% print summary
results.print_res()
results.plot()

# %% load the best saved model
best_full_fine_model = full_fine_tuning()
best_full_fine_model.load_state_dict(torch.load(f'{MODELFOLDER}/{results.model_name}.pt'))
best_full_fine_model.to(dev)

# %% evaluate test error of the best model

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)
# the layers are switched off by default in this model
test_accuracy, test_loss = evaluate(best_full_fine_model, test_loader, switch_off_layers=True)
print(f'test accuracy: {test_accuracy}')
print(f'test loss: {test_loss}')
