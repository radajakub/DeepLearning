import os

import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from models import VAE


def plot_elbo(vaes: list[VAE]):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=200)
    cmap = plt.cm.get_cmap('tab10', len(vaes))
    for i, vae in enumerate(vaes):
        elbos, val_elbos = vae.load_elbos('outputs')
        n = min(elbos.shape[0], 100)
        xs = np.arange(n)
        ax.plot(xs, elbos[:n], label=f'ELBO ({vae.name} VAE)', color=cmap(i))
        ax.plot(xs, val_elbos[:n], label=f'Validation ELBO ({vae.name} VAE)', color=cmap(i), linestyle='dashed')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('ELBO')
        ax.legend()


def show_reconstructed(images: torch.tensor, vaes: list[VAE]):
    n_images = images.shape[0]
    n_vaes = len(vaes)

    unit = 2
    figsize = ((n_vaes + 1) * unit, n_images * unit)

    fig, axes = plt.subplots(n_images, n_vaes + 1, gridspec_kw={'wspace': 0, 'hspace': 0}, sharex=True, sharey=True, figsize=figsize, dpi=50)

    for ax in axes.flatten():
        ax.axis('off')

    axes[0, 0].set_title('Original')
    for ax, vae in zip(axes[0, 1:], vaes):
        ax.set_title(vae.name)

    for i, image in enumerate(images):
        img = image.numpy().transpose(1, 2, 0)
        axes[i, 0].imshow(img)

    for i, image in enumerate(images):
        x = image.flatten(start_dim=0)
        for j, vae in enumerate(vaes):
            with torch.no_grad():
                img = vae.reconstruct(x).numpy()
            img = img.reshape(1, 28, 28).transpose(1, 2, 0)
            axes[i, j + 1].imshow(img)

    fig.tight_layout()


def show_collapse(images: torch.tensor, vaes: list[VAE], dev: torch.device):
    for i, vae in enumerate(vaes):
        with torch.no_grad():
            kl_divergences = vae.collapse(images, dev).numpy()
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        # ax.bar(np.arange(kl_divergences.shape[0]), kl_divergences)
        ax.hist(kl_divergences, bins=16, range=(0.0, kl_divergences.max()), edgecolor='black')
        ax.set_title(f'Latent avg. KL Divergence ({vae.name} VAE)')
        ax.set_xlabel('KL Divergence')
        ax.set_ylabel('Occurences')


def show_decoder(vaes: list[VAE]):
    for vae in vaes:
        fig, axes = plt.subplots(8, 8, figsize=(12, 12))
        fig.suptitle(f'{vae.name} VAE decoder samples')
        with torch.no_grad():
            mus = vae.random_codes(64).numpy()
        for i, mu in enumerate(mus):
            img = mu.reshape(1, 28, 28).transpose(1, 2, 0)
            axes.flat[i].imshow(img)
            axes.flat[i].axis('off')
        fig.tight_layout()


def show_animation(vaes: list[VAE]):
    for vae in vaes:
        k = 100
        images = vae.limiting_distribution(64, k)

        iters_images = []

        for iter in images:
            imgs = []
            for img in iter:
                imgs.append(img.numpy().reshape((1, 28, 28)).transpose(1, 2, 0))
            iters_images.append(imgs)

        def update_plot(i, ims):
            for j, im in enumerate(ims):
                im.set_data(iters_images[i][j])
            return ims

        fig, axes = plt.subplots(8, 8, figsize=(4, 4))
        fig.suptitle(f'Limiting distribution of\n{vae.name} VAE')

        ims = []
        for i, ax in enumerate(axes.flat):
            ax.axis('off')
            im = ax.imshow(iters_images[0][i])
            ims.append(im)

        fig.tight_layout()

        ani = animation.FuncAnimation(fig, update_plot, frames=k, fargs=(ims,), interval=100)

        os.makedirs('images', exist_ok=True)
        ani.save(f'ld_{vae.name}.gif')
