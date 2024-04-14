import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset


class CircleDataGenerator():
    def one_class_generate(self, radius, y, n):
        angle = np.random.uniform(0, 2 * np.pi, n)
        noise = np.random.uniform(-1, 1, n)
        r = radius + noise
        x1 = np.cos(angle) * r + 10
        x2 = np.sin(angle) * r + 10
        x = np.stack([x1, x2], axis=1)
        t = np.ones((n,), dtype=np.int64) * int(y)
        return x, t

    def generate_sample(self, n):
        x1, t1 = self.one_class_generate(4, 1, n // 2)
        x0, t0 = self.one_class_generate(1, 0, n // 2)

        x = np.concatenate((x1, x0), axis=0)
        t = np.concatenate((t1, t0), axis=0)

        x = torch.tensor(x, dtype=torch.float32)
        t = torch.tensor(t, dtype=torch.long)

        return x, t


class MNISTData():
    def __init__(self, batch_size):
        # transforms
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        # transforms.Lambda(torch.flatten)])
        transform = transforms.ToTensor()
        self.train_set = torchvision.datasets.MNIST('./data', download=True, train=True, transform=transform)
        self.test_set = torchvision.datasets.MNIST('./data', download=True, train=False, transform=transform)

        # split train_set into train_subset and val_subset
        self.train_subset = Subset(self.train_set, list(range(5000)))
        self.val_subset = Subset(self.train_set, list(range(5000, 15000)))

        # dataloaders
        self.train_loader = torch.utils.data.DataLoader(self.train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.val_loader = torch.utils.data.DataLoader(self.val_subset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=batch_size, shuffle=False, num_workers=0)
