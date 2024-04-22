# %%
from typing import Callable
import torch
import torch.nn as nn
import numpy as np
import scipy as sp
from toy_model import G2Model
from model import FFModel
import os

import matplotlib.pyplot as plt

# !%load_ext autoreload
# !autoreload 2
# !%matplotlib inline

# %%


def get_torch_fun_name(function: Callable):
    return function.__name__.split('.')[-1]

# %%


def get_device(dtype: torch.dtype):
    # tried to use mps on macos but the results were considerably worse (loss was increasing)
    # if torch.backends.mps.is_available() and dtype == torch.float32:
    #     device = torch.device('mps')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'using {device} for computations')
    return device

# %%


class Predictor():
    """ Wrapper class to convert tensors to numpy and interface with G2Model, e.g. to call
        G2Model.plot_boundary()
    """

    def __init__(self, name, model):
        self.name = name
        self.model = model

    def score(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x).to(self.model.W1)
        return self.model.score(x).detach().cpu().numpy()

    def classify(self, x_i):
        scores = self.score(x_i)
        return np.sign(scores)


def generate_data(gmx: G2Model, sample_size: int, dtype: torch.dtype, device: torch.device):
    x, t = gmx.generate_sample(sample_size)
    x = torch.tensor(x, dtype=dtype).to(device)
    t = torch.tensor(t, dtype=torch.int).to(device)
    return x, t

# %% gradient check function


def gradient_check(epsilon, dtype):
    torch.manual_seed(1)
    smpl_size = 200
    hdim = 500

    device = get_device(dtype)

    # get training/validation data
    # transform them to PyTorch tensors
    gmx = G2Model()

    x, t = generate_data(gmx, smpl_size, dtype, device)

    # model
    # model = FFModel(2, hdim, torch.tanh, device, dtype)
    model = FFModel(2, hdim, torch.tanh, device, dtype)

    # check gradient
    print(f'Gradient checks {dtype} {epsilon}', flush=True)
    model.check_gradient(x, t, 'W1', epsilon=epsilon)
    model.check_gradient(x, t, 'b1', epsilon=epsilon)
    model.check_gradient(x, t, 'w', epsilon=epsilon)
    model.check_gradient(x, t, 'b', epsilon=epsilon)
    print()


# %%


def train(hidden_size, learning_rate=0.1, activation=torch.tanh, epochs=1000, log_period=10, dtype=torch.float64):
    torch.manual_seed(1)
    smpl_size = 200
    vsmpl_size = 1000
    tsmpl_size = 30000
    name = "model_{}_{}".format(hidden_size, get_torch_fun_name(activation))

    device = get_device(dtype)

    gmx = G2Model()

    x, t = generate_data(gmx, smpl_size, dtype, device)
    xv, tv = generate_data(gmx, vsmpl_size, dtype, device)
    xt, tt = generate_data(gmx, tsmpl_size, dtype, device)

    # model
    model = FFModel(2, hidden_size, activation, device, dtype)

    print('# Starting', flush=True)
    for count in range(epochs):
        # compute loss
        l = model.mean_loss(x, t)
        # compute gradient
        model.zero_grad()
        l.backward()
        # make a gradinet descent step
        for p in model.parameters:
            p.data -= learning_rate * p.grad.data
        # evaluate and print
        if (count % log_period == log_period - 1) or (count == epochs - 1):
            with torch.no_grad():
                vacc = model.mean_accuracy(xv, tv)
            print(f'epoch: {count}  loss: {l.item():.4f} vacc: {vacc:.4f}', flush=True)

    tacc = model.mean_accuracy(xt, tt)
    terr = 1 - tacc
    print(f'test error on {tsmpl_size} samples: {terr:.4f}', flush=True)

    training_error = 1 - model.mean_accuracy(x, t)
    print(f'training error {training_error:.4f}', flush=True)

    generalization_gap = abs(training_error - terr)
    print(f'generalization gap: {generalization_gap:.4f}', flush=True)

    pred = Predictor(name, model)
    plt.clf()
    gmx.plot_boundary((x.cpu().numpy(), t.cpu().numpy()), pred, title=name)
    plt.savefig(os.path.join('boundaries', name + '.png'))
    return model, terr, gmx

# %%


def compute_test_errors(model: FFModel, gmx: G2Model, device, m: int = 1000, repetitions: int = 10000):
    err = np.empty(repetitions, np.float64)
    var = np.empty(repetitions, np.float64)

    for i in range(repetitions):
        x, t = generate_data(gmx, m, torch.float64, device)
        e, v = model.empirical_test_error(x, t)
        err[i] = e
        var[i] = v

    return err, var

# %%


class Plotter:
    def __init__(self, m: int):
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 5))
        self.bins = np.array([(0.5 / m) + i * (1 / m) for i in range(500)])
        self.ax.set_xlabel('Error rate, %')
        self.ax.set_ylabel('Frequency')

    def plot_distribution(self, errors: np.array, center: np.float64, color: str, name: str, mean_name: str):
        self.ax.hist(errors, bins=self.bins, color=color, alpha=0.3, label=name)
        self.ax.axvline(center, color=color, linestyle='dashed', label=mean_name)

    def plot_confidence(self, low: np.float64, high: np.float64, center: np.float64, y: np.float64, color: str, name: str):
        self.ax.errorbar(center, y, xerr=[[center - low], [high - center]], fmt='o', color=color, markersize=3, capsize=5, label=name)

    def show(self):
        self.ax.legend()
        return self.fig

    def save(self, name: str):
        self.ax.legend()
        self.ax.set_title(name)
        self.fig.tight_layout()
        self.fig.savefig(os.path.join('errors', name + '.png'))

# %%


def hoeffding(m: int, alpha: float, delta_l: float = 1.0):
    return np.sqrt((np.power(delta_l, 2) * (np.log(2) - np.log(1 - alpha))) / (2 * m))


def chebyshev(errors: np.array, alpha: float):
    m = errors.size
    return np.sqrt((np.var(errors)) / (m * (1 - alpha)))


def true_chebyshev(var: np.float64, m: int, alpha: float):
    return np.sqrt(var / (m * (1 - alpha)))


def bernstein(errors: np.array, alpha: float):
    m = errors.size
    logs = (np.log(2) - np.log(1 - alpha))
    A = m
    B = - logs * (2 / 3) * 1
    C = - logs * 2 * np.var(errors)
    D = np.sqrt(np.power(B, 2) - 4 * A * C)
    t1 = (-B + D) / (2 * A)
    t2 = (-B - D) / (2 * A)
    return np.maximum(t1, t2)

# %%


def predictor_error_analysis(model: FFModel, gmx: G2Model, model_name: str, alpha: float, device, m: int, repetitions: int):

    print(f'Error analysis of {model_name}', flush=True)

    empirical_errors, empirical_vars = compute_test_errors(test_model, gmx, device, m=m, repetitions=repetitions)
    mean_empirical_error = np.mean(empirical_errors)
    mean_empirical_variance = np.mean(empirical_vars)

    # compute vector of erros on a single training set
    tx, tt = generate_data(gmx, m, torch.float64, device)
    errors = (model.classify(tx) != tt).detach().cpu().numpy().astype(int)
    Rt = np.mean(errors)

    # compute bootstrap distribution and confidence intervals
    res = sp.stats.bootstrap((errors,), np.mean, confidence_level=0.9, method='BCa')
    bootstrap_errors = res.bootstrap_distribution

    # compute hoeffding and chebyshev
    hoeffding_e = hoeffding(m, alpha)
    chebyshev_e = chebyshev(errors, alpha)
    true_chebyshev_e = true_chebyshev(mean_empirical_variance, m, alpha)
    bernstein_e = bernstein(errors, alpha)

    print(f'Hoeffding confidence: {Rt} \u00B1 {hoeffding_e}')
    print(f'Chebyshev confidence: {Rt} \u00B1 {chebyshev_e}')
    print(f'Chebyshev confidence (true V): {Rt} \u00B1 {true_chebyshev_e}')
    print(f'Bernstein confidence: {Rt} \u00B1 {bernstein_e}')
    print()

    plotter = Plotter(m)
    plotter.plot_distribution(empirical_errors, mean_empirical_error, 'tab:blue', name='Test error distribution', mean_name='True (expected) error rate')
    plotter.plot_distribution(bootstrap_errors, Rt, 'tab:orange', name='Bootstrap distribution', mean_name='Empirical error rate RT')
    plotter.plot_confidence(res.confidence_interval.low, res.confidence_interval.high, Rt, 200, 'magenta', name='90.0% - Bootstrap BCa')
    plotter.plot_confidence(Rt - hoeffding_e, Rt + hoeffding_e, Rt, 50, 'red', name='90.0% - Hoeffding')
    plotter.plot_confidence(Rt - chebyshev_e, Rt + chebyshev_e, Rt, 100, 'blue', name='90.0% - Chebyshev estimated V')
    plotter.plot_confidence(Rt - true_chebyshev_e, Rt + true_chebyshev_e, Rt, 150, 'black', name='90.0% - Chebyshev true V')
    plotter.plot_confidence(Rt - bernstein_e, Rt + bernstein_e, Rt, 250, 'green', name='90.0% - Bernstein estimated V')
    plotter.save(name)


# %% gradient checks
for dtype in [torch.float32, torch.float64]:
    for epsilon in [10e-2, 10e-3, 10e-4, 10e-5,]:
        gradient_check(epsilon, dtype)

# %% learning experiments
hidden_dims = [5, 10, 100, 500]
activations = [torch.tanh, nn.functional.relu]

test_errors = {}
models = {}
# save generators so we can evalute the error on them to avoid bias by sampling from the same seed for testing!
generators = {}

for hidden_size in hidden_dims:
    for activation in activations:
        model, test_error, generator = train(hidden_size, activation=activation, dtype=torch.float64)
        test_errors[(hidden_size, activation)] = test_error
        models[(hidden_size, activation)] = model
        generators[(hidden_size, activation)] = generator


# %% print test errors nicely
print(' ' * 4 + ' | ', end='')
for hd in hidden_dims:
    print(f'{hd:6d}', end=' | ')
print()
for activation in activations:
    activation_name = get_torch_fun_name(activation)
    print(f'{activation_name}', end=' | ')
    for hd in hidden_dims:
        print(f'{test_errors[(hd, activation)]:.4f}', end=' | ')
    print()

# %%
# fixed test set size
m = 1000
# number of samples to estimate distribution of empirical errors
repetitions = 10000
alpha = 0.9
# create a ground truth model for testing
device = get_device(dtype)

# %%
# select model to test on and compute its empirical error distribution (repetitions samples)
for hidden_size in hidden_dims:
    for activation in activations:
        test_model = models[(hidden_size, activation)]
        generator = generators[(hidden_size, activation)]
        name = "model_{}_{}".format(hidden_size, get_torch_fun_name(activation))
        predictor_error_analysis(test_model, generator, name, alpha, device, m, repetitions)

# %%
# investigate 5 relu
model = models[(5, nn.functional.relu)]
generator = generators[(5, nn.functional.relu)]
name = "model_{}_{}".format(5, get_torch_fun_name(nn.functional.relu))
predictor_error_analysis(model, generator, name, alpha, device, m, repetitions)
