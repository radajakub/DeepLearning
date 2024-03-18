import torch
import torch.nn as nn
import copy
import numpy as np

class FFModel():
    def __init__(self, hdim, device, dtype):
        """ hdim -- hidden layer size """
        # hidden layer
        self.w1 = torch.empty([2, hdim], dtype=dtype, device=device).uniform_(-1.0, 1.0) # [2 hdim]
        self.w1.requires_grad_()
        # ...
        self.parameters = [self.w1, ]
    
    def score(self, x):
        """ Compute scores for inputs x 
        x : [N x d] 
        output:
        s : [N] - scores
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x).to(self.w1)
        # raise NotImplementedError()
        # implement score here
        s = (x @ self.w1).sum(dim=-1)
        return s

    def classify(self, x):
        scores = self.score(x)
        return scores.sign()

    def mean_loss(self, x, targets):
        """               
        Compute the mean_loss of the training data = average negative log likelihood
        *
        :param train_data: tuple(x,y)
        x [N x d]
        y [N], encoded +-1 classes
        :return: mean negative log likelihood
        """
        raise NotImplementedError()

    def mean_accuracy(self, x, targets):
        y = self.classify(x)
        acc = (y == targets).float().mean()
        return acc

    def zero_grad(self):
        # set .grad to None (or zeroes) for all parameters
        for p in self.parameters:
            p.grad = None
  
    def check_gradient(self, x, targ, pname):
        p = getattr(self, pname)
        epsilon = 1.e-5
        # compute gradients
        raise NotImplementedError()
        print("# Grad error in {}: {:.4}".format(pname, torch.abs(error)))
