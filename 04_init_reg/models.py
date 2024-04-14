import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepNetwork(nn.Module):
    def __init__(self, layer_size: int, num_layers: int, activation: str, init_name: str):
        super().__init__()

        self.activation = self.get_activation(activation)
        self.init_name = init_name

        self.layer_size = layer_size
        self.num_layers = num_layers

        linear_layers = [nn.Linear(in_features=layer_size, out_features=layer_size) for _ in range(num_layers)]

        self.layers = nn.Sequential(*linear_layers)

        # intialize the created layers
        self.apply(self.initialize_layer)

        self.forward_statistics = np.zeros((self.num_layers, 2))
        self.backward_statistics = np.zeros((self.num_layers, 2))

    def forward(self, x):
        self.forward_statistics = np.zeros((self.num_layers, 2))
        for i in range(self.num_layers):
            x = self.layers[i](x)
            # TODO: maybe don't use the last activation?
            x = self.activation(x)
            self.forward_statistics[i, 0] = x.mean().item()
            self.forward_statistics[i, 1] = x.std().item()

        return x

    def compute_gradient_statistics(self):
        for i, layer in enumerate(self.layers):
            self.backward_statistics[i, 0] = layer.weight.grad.mean().item()
            self.backward_statistics[i, 1] = layer.weight.grad.std().item()

    def initialize_layer(self, layer: nn.Module) -> None:
        if isinstance(layer, nn.Linear):
            # ignore bias
            nn.init.zeros_(layer.bias.data)
            # initialize weights
            if self.init_name == 'original':
                a = 1 / np.sqrt(self.layer_size)
                nn.init.uniform_(layer.weight.data, -a, a)
            elif self.init_name == 'xavier_uniform':
                nn.init.xavier_uniform_(layer.weight.data)
            elif self.init_name == 'xavier_normal':
                nn.init.xavier_normal_(layer.weight.data)
            elif self.init_name == 'kaiming_uniform':
                nn.init.kaiming_uniform_(layer.weight.data, nonlinearity='relu')
            elif self.init_name == 'kaiming_normal':
                nn.init.kaiming_normal_(layer.weight.data, nonlinearity='relu')
            else:
                raise ValueError(f'Unsupported initialization: {self.init_name}')

    def get_activation(self, name: str) -> nn.Module:
        if name == 'relu':
            return nn.ReLU()
        elif name == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f'Unsupported activation: {name}')


class ShallowNetwork(nn.Module):
    def __init__(self, in_dim: int, l1_dim: int, l2_dim: int, out_dim: int):
        super().__init__()

        self.dim = in_dim
        self.classes = out_dim

        # first hidden linear layer
        self.l1 = nn.Linear(in_dim, l1_dim, bias=True)
        self.init_layer(self.l1)

        # second hidden linear layer
        self.l2 = nn.Linear(l1_dim, l2_dim, bias=True)
        self.init_layer(self.l2)

        # output layer to reduce to the number of classes
        self.output = nn.Linear(l2_dim, out_dim, bias=True)
        self.init_layer(self.output)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.softmax(self.output(x), dim=-1)
        return x

    def classify(self, x: torch.tensor) -> torch.tensor:
        self.eval()
        with torch.no_grad():
            scores = self.forward(x)
            y = scores.argmax(dim=-1)
        self.train()
        return y

    def accuracy(self, x: torch.tensor, t: torch.tensor) -> float:
        y = self.classify(x).detach()
        return (y == t).sum().item() / t.shape[0]

    def init_layer(self, layer: nn.Module) -> None:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight.data, nonlinearity='relu')
            # some way to initialize bias which I found
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
            a = 1 / np.sqrt(fan_in)
            nn.init.uniform_(layer.bias.data, -a, a)


class MNISTNetwork(nn.Module):
    def __init__(self, dropout_p: float):
        super().__init__()
        self.p = dropout_p

        self.layers = nn.Sequential(
            nn.Dropout(p=self.p),
            nn.Linear(784, 800),
            nn.Dropout(p=self.p),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.Dropout(p=self.p),
            nn.ReLU(),
            nn.Linear(800, 10),
        )

        self.apply(self.init_layer)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.layers(x)
        x = x.softmax(dim=1)
        return x

    def classify(self, probs: torch.tensor) -> torch.tensor:
        return probs.argmax(dim=1)

    def loss(self, probs: torch.tensor, t: torch.tensor) -> float:
        return F.cross_entropy(probs, t)

    def error(self, y: torch.tensor, t: torch.tensor) -> float:
        return (y != t).sum().item()

    def init_layer(self, layer: nn.Module) -> None:
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight.data, mean=0, std=0.1)
            nn.init.constant_(layer.bias.data, 0)


# helper class for the last assignment
class MNISTEnsemble(nn.Module):
    def __init__(self, dropout_p: float):
        super().__init__()
        self.p = dropout_p

        self.layers = nn.Sequential(
            nn.Dropout(p=self.p),
            nn.Linear(784, 800),
            nn.Dropout(p=self.p),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.Dropout(p=self.p),
            nn.ReLU(),
            nn.Linear(800, 10),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.layers(x)
        return x

    def error(self, y: torch.tensor, t: torch.tensor) -> float:
        return (y != t).sum().item()
