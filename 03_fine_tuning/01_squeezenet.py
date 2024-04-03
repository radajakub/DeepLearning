# %%
import os
import numpy as np

import torch
import torch.utils.data
import torchvision.models
from torchvision import transforms
from PIL import Image


import matplotlib.pyplot as plt

# !%load_ext autoreload
# !%autoreload 2
# !%matplotlib inline

print(torch.__version__)

dog_file = 'data/dog.jpg'
imagenet_classes_file = 'data/imagenet_classes.txt'

outfolder = 'out'
os.makedirs(outfolder, exist_ok=True)


# %% (1) load squeezenet model
model = torchvision.models.squeezenet1_0(weights=torchvision.models.SqueezeNet1_0_Weights.DEFAULT)
print(model)
dev = model.features[0].weight.device
print(dev)

# %% (2) load and transform the image

img = Image.open(dog_file)
plt.figure()
plt.imshow(img)
plt.show()

# reshape as [C x H x W], normalize
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(224, antialias=True),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# reshape as [B C H W], move to device
x = transform(img).unsqueeze(0).to(dev)


# %% (3) compute class predictive probabilities and report top 5 classes and their probabilities


with open(imagenet_classes_file, 'r') as f:
    classes = [s.strip() for s in f.readlines()]

# set model to evaluation mode
model.eval()
# evaluate the model on image x and compute probabilities of classes
y = model(x).squeeze(0).softmax(0)

# take top 5 most probable elements with indices
probs, indices = torch.topk(y, 5, dim=0)

# print the results
print('Top 5 probable classes:')
for prob, idx in zip(probs, indices):
    print(f'{classes[idx]}: {prob}')

# %%

# %% (4) display weights of the first convolutional layer as images in a grid of 8x12

weight_tensor = model.features[0].weight.detach().numpy()

rows, cols = 8, 12
fig, axes = plt.subplots(rows, cols, layout='constrained')
for r in range(rows):
    for c in range(cols):
        weight = weight_tensor[r * cols + c, :, :, :]
        # transpose from (3, 7, 7) to (7, 7, 3) and preserve the order of the first two
        w = np.transpose(weight, (1, 2, 0))
        # normalize each color to be in [0, 1] range
        min_w = w.min(axis=(0, 1), keepdims=True)
        max_w = w.max(axis=(0, 1), keepdims=True)
        w = (w - min_w) / (max_w - min_w)
        # plot the image
        axes[r, c].set_axis_off()
        axes[r, c].imshow(w)
fig.savefig(os.path.join(outfolder, '01_first_layer_weights.png'))

# %% (5) Apply the First linear layer of the network to the input image and display the resulting activation maps for the first 16 channels

# print(model)
# there is no linear layer in the network so I assume the first convolutional layer is meant

first_layer = model.features[0]
first_two_layers = model.features[0:2]

submodels = [first_layer, first_two_layers]
names = ['first_layer', 'first_layer_activated']

for submodel, name in zip(submodels, names):
    rows, cols = 4, 4
    y = submodel(x).detach().numpy().squeeze(0)[:(rows * cols), :, :]
    fig, axes = plt.subplots(rows, cols, layout='constrained')
    for r in range(rows):
        for c in range(cols):
            axes[r, c].set_axis_off()
            axes[r, c].imshow(y[r * cols + c, :, :])
    fig.savefig(os.path.join(outfolder, f'01_pass_{name}.png'))
