import os

import numpy as np
import matplotlib.pyplot as plt
import torch

from models import DeepNetwork
from progress import Progress


def plot_stats(ax, xs: np.ndarray, mu: np.ndarray, std: np.ndarray, title: str):
    ax.plot(xs, mu, color='tab:blue', label='mean')
    ax.errorbar(xs, mu, std, fmt='o', linewidth=1, capsize=3, markersize=3, color='tab:blue', label='standard deviation')
    ax.set_xticks([i for i in range(0, xs.shape[0], 5)])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Activation value')
    ax.set_title(title)
    ax.grid()
    ax.legend()


def plot_weights(model: DeepNetwork, outname: str):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), squeeze=False)

    xs = np.arange(model.num_layers)
    plot_stats(axes[0, 0], xs, model.forward_statistics[:, 0], model.forward_statistics[:, 1], title='Forward activations')
    plot_stats(axes[0, 1], xs, model.backward_statistics[:, 0], model.backward_statistics[:, 1], title='Backward activations')

    fig.tight_layout()
    fig.savefig(outname)


def plot_val_comparison(outname: str, xs: np.ndarray, vals: np.ndarray, norm_vals: np.ndarray, mult_vals: np.ndarray, ylabel: str, title: str):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(xs, vals, color='tab:red', label='Sampled data')
    ax.plot(xs, norm_vals, color='tab:green', label='Normalized data')
    ax.plot(xs, mult_vals, color='tab:blue', label='Normalized data multiplied by 0.01')
    ax.set_xticks([i for i in range(0, xs.shape[0] + 1, 1000)])
    ax.set_title(title)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.legend()
    fig.tight_layout()
    fig.savefig(outname)


def plot_progresses(p: Progress, norm_p: Progress, mult_p: Progress, outdir: str):
    xs = np.arange(1, p.epochs + 1)

    # plot learning losses
    outname = os.path.join(outdir, 'training_loss.png')
    plot_val_comparison(outname, xs, p.training_l, norm_p.training_l, mult_p.training_l, ylabel='Loss', title='Training loss')

    # plot accuracies
    plot_val_comparison(os.path.join(outdir, 'training_acc.png'), xs, p.training_acc, norm_p.training_acc,
                        mult_p.training_acc, ylabel='Accuracy', title='Training accuracy')
    plot_val_comparison(os.path.join(outdir, 'validation_acc.png'), xs, p.validation_acc, norm_p.validation_acc,
                        mult_p.validation_acc, ylabel='Accuracy', title='Validation accuracy')

    # plot norms of the gradient
    plot_val_comparison(os.path.join(outdir, 'grad_1.png'), xs, p.gradient_norm[:, 0], norm_p.gradient_norm[:, 0],
                        mult_p.gradient_norm[:, 0], ylabel='Norm of gradient', title='First layer (6 neurons)')
    plot_val_comparison(os.path.join(outdir, 'grad_2.png'), xs, p.gradient_norm[:, 1], norm_p.gradient_norm[:, 1],
                        mult_p.gradient_norm[:, 1], ylabel='Norm of gradient', title='Second layer (3 neurons)')
    plot_val_comparison(os.path.join(outdir, 'grad_out.png'), xs, p.gradient_norm[:, 2], norm_p.gradient_norm[:, 2],
                        mult_p.gradient_norm[:, 2], ylabel='Norm of gradient', title='Output layer')


# add scale parameter for the case when we multiply x by 0.01
def plot_decision_boundary(gt_data, gt_target, model, device, outname, scale=1):
    step_size = 0.1 * scale
    padding = scale
    xmin = gt_data[:, 0].min().item() - padding
    xmax = gt_data[:, 0].max().item() + padding
    ymin = gt_data[:, 1].min().item() - padding
    ymax = gt_data[:, 1].max().item() + padding
    xx, yy = torch.meshgrid(torch.arange(xmin, xmax + step_size, step_size),
                            torch.arange(ymin, ymax + step_size, step_size))
    grid_data = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    model.eval()
    with torch.no_grad():
        y = model.forward(grid_data.to(device))
        y = y.detach().cpu().numpy()
    model.train()

    prob = y[:, 0].reshape(xx.shape)

    data = gt_data.detach().cpu().numpy()
    t = gt_target.detach().cpu().numpy()

    fig, ax = plt.subplots(1, 1)
    ax.imshow(prob.T, origin='lower', extent=(xmin, xmax, ymin, ymax), cmap='RdBu')
    ax.contour(xx, yy, prob, [0.5], origin='lower', colors='k')
    ax.plot(data[t == 0, 0], data[t == 0, 1], 'o', color='orange')
    ax.plot(data[t == 1, 0], data[t == 1, 1], 'o', color='lightgreen')
    fig.tight_layout()
    fig.savefig(outname)
