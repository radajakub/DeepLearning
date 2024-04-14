from copy import deepcopy

import torch
import torch.nn.functional as F
import numpy as np

from models import ShallowNetwork


class Progress:
    def __init__(self, epochs: int, num_layers: int) -> None:
        self.training_l = np.zeros(epochs)
        self.training_acc = np.zeros(epochs)
        self.validation_acc = np.zeros(epochs)
        self.gradient_norm = np.zeros((epochs, num_layers))

        self.epochs = epochs

        self.best_v_acc = 0
        self.best_model = None

    def initial(self, model: ShallowNetwork, x: torch.tensor, t: torch.tensor) -> None:
        model.eval()
        with torch.no_grad():
            self.initial_loss = F.cross_entropy(model(x), t).item()
        model.train()
        self.initial_target_loss = - np.log(1 / model.classes)
        print(f'initial loss is {self.initial_loss}')
        print(f'it should be around {self.initial_target_loss}')

    def log(self, model: ShallowNetwork, e: int, t_l: float, data: tuple[torch.tensor, torch.tensor], val_data: tuple[torch.tensor, torch.tensor]) -> None:
        self.training_l[e] = t_l

        x, t = data
        t_acc = model.accuracy(x, t)
        self.training_acc[e] = t_acc

        x_val, t_val = val_data
        v_acc = model.accuracy(x_val, t_val)
        self.validation_acc[e] = v_acc

        print(f'epoch {e + 1} training loss {t_l} training accuracy {t_acc} validation accuracy {v_acc}')

        if v_acc >= self.best_v_acc:
            self.best_v_acc = v_acc
            # state_dict() is just a reference!!
            self.best_model = deepcopy(model.state_dict())

        # logg gradients
        self.gradient_norm[e, 0] = model.l1.weight.grad.norm().item()
        self.gradient_norm[e, 1] = model.l2.weight.grad.norm().item()
        self.gradient_norm[e, 2] = model.output.weight.grad.norm().item()
