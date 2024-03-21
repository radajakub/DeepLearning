from typing import Callable

import torch
import numpy as np


class FFModel():
    def __init__(self, input_size: int, hidden_size: int, activation: Callable, device: str, dtype: torch.dtype):
        """ hdim -- hidden layer size """
        # hidden layer
        self.W1 = torch.empty([input_size, hidden_size], dtype=dtype, device=device).uniform_(-1.0, 1.0)
        self.W1.requires_grad_()
        self.b1 = torch.empty([hidden_size], dtype=dtype, device=device).uniform_(-1.0, 1.0)
        self.b1.requires_grad_()

        # logistic regression layer
        self.w = torch.empty([hidden_size], dtype=dtype, device=device).uniform_(-1.0, 1.0)
        self.w.requires_grad_()
        self.b = torch.empty(1, dtype=dtype, device=device).uniform_(-1.0, 1.0)
        self.b.requires_grad_()

        self.activation = activation

        self.parameters = [self.W1, self.b1, self.w, self.b]

    def score(self, x: torch.tensor) -> torch.tensor:
        """ Compute scores for inputs x
        x : [N x d]
        output:
        s : [N] - scores
        """
        if isinstance(x, np.ndarray):
            # turns the numpy array into a tensor with the same dtype and device as self.W1
            x = torch.tensor(x).to(self.W1)

        # phi = torch.tanh(x @ self.W1 + self.b1)
        phi = self.activation(x @ self.W1 + self.b1)

        s = phi @ self.w + self.b

        return s

    def classify(self, x: torch.tensor) -> torch.tensor:
        scores = self.score(x)
        return scores.sign()

    def mean_loss(self, x: torch.tensor, targets: torch.tensor) -> torch.float:
        """
        Compute the mean_loss of the training data = average negative log likelihood
        *
        :param train_data: tuple(x,y)
        x [N x d]
        y [N], encoded +-1 classes
        :return: mean negative log likelihood
        """
        s = self.score(x)
        return - torch.mean(torch.log(torch.sigmoid(targets * s)))

    def mean_accuracy(self, x, targets):
        y = self.classify(x)
        acc = (y == targets).float().mean()
        return acc

    def empirical_test_error(self, x, targets):
        y = self.classify(x)
        errs = (y != targets).float()
        return torch.mean(errs).item(), torch.var(errs).item()

    def zero_grad(self):
        # set .grad to None (or zeroes) for all parameters
        for p in self.parameters:
            p.grad = None

    def check_gradient(self, x: torch.tensor, targ: torch.tensor, param_name: str, epsilon: torch.float32 = 1.e-5) -> None:
        param = getattr(self, param_name)
        # clone weights to reuse them afterwards
        original_weights = param.data

        # compute numerical gradient
        # (1) sample random unit direction
        u = torch.empty_like(param).uniform_(-1.0, 1.0)
        u = u / torch.norm(u)

        # (2) compute L(w + epsilon * u)
        param.data = original_weights + epsilon * u
        with torch.no_grad():
            L_plus = self.mean_loss(x, targ)

        # (3) compute L(w - epsilon * u)
        param.data = original_weights - epsilon * u
        with torch.no_grad():
            L_minus = self.mean_loss(x, targ)

        # (4) compute numerical gradient
        g = (L_plus - L_minus) / (2 * epsilon)

        # return original weights
        param.data = original_weights

        # compute auto gradient
        self.zero_grad()
        loss = self.mean_loss(x, targ)
        loss.backward()
        nabla_L = param.grad

        # compute error
        error = torch.abs(g - nabla_L.flatten().dot(u.flatten()))
        print("Grad error in {}: {:.4}".format(param_name, error))
