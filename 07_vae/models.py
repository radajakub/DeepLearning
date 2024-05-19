import os
import torch
import torch.nn as nn
import torch.distributions as dstr
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from helpers import flatten_batch, select_device

import numpy as np

MNIST_DIM = 784


class Encoder(nn.Module):
    def __init__(self, zdim: int, input_dim: int = MNIST_DIM, layer_sizes: list[int] = []):
        super().__init__()

        # construct the network
        self.zdim = zdim
        self.net = nn.Sequential()
        if len(layer_sizes) > 0:
            self.net.append(nn.Linear(input_dim, layer_sizes[0]))
            self.net.append(nn.ReLU())
            for i in range(len(layer_sizes) - 1):
                self.net.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                self.net.append(nn.ReLU())
            self.net.append(nn.Linear(layer_sizes[-1], self.zdim * 2))
        else:
            self.net.append(nn.Linear(input_dim, self.zdim * 2))

    def forward(self, x):
        scores = self.net(x)
        # split the scores tensor into chunks of size zdim
        mu, sigma = torch.split(scores, self.zdim, dim=1)
        # sigma = torch.clamp(sigma, -20, 2)
        sigma = torch.exp(sigma)
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, zdim: int, output_dim: int = MNIST_DIM, layer_sizes: list[int] = []):
        super().__init__()

        # construct the network
        self.zdim = zdim
        self.net = nn.Sequential()
        if len(layer_sizes) > 0:
            self.net.append(nn.Linear(self.zdim, layer_sizes[0]))
            self.net.append(nn.ReLU())
            for i in range(len(layer_sizes) - 1):
                self.net.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                self.net.append(nn.ReLU())
            self.net.append(nn.Linear(layer_sizes[-1], output_dim))
        else:
            self.net.append(nn.Linear(self.zdim, output_dim))

        # if you learn the sigma of the decoder
        self.logsigma = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        mu = self.net(x)
        return mu


class VAE(nn.Module):
    def __init__(self, name: str, zdim: int, lr: float, input_dim: int = MNIST_DIM, encoder_hidden: list[int] = [], decoder_hidden: list[int] = []):
        super().__init__()

        self.name = name

        self.decoder = Decoder(zdim, output_dim=input_dim, layer_sizes=decoder_hidden)
        self.encoder = Encoder(zdim, input_dim=input_dim, layer_sizes=encoder_hidden)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def compute_nelbo(self, x: torch.tensor) -> torch.tensor:
        # apply encoder q(z|x)
        z_mu, z_sigma = self.encoder(x)
        qz = dstr.Normal(z_mu, z_sigma)

        # sample with re-parametrization
        z = qz.rsample()

        # apply decoder p(x|z)
        x_mu = self.decoder(z)
        px = dstr.Normal(x_mu, torch.exp(self.decoder.logsigma))

        # prior p(z)
        pz = dstr.Normal(torch.zeros_like(z_mu), torch.ones_like(z_mu))

        # learn
        logx = px.log_prob(x)
        logx = logx.mean(0).sum()

        # KL-Div term
        kl_div = dstr.kl_divergence(qz, pz).mean(0).sum()
        nelbo = kl_div - logx
        return nelbo

    def learn_step(self, x: torch.tensor):
        self.optimizer.zero_grad()

        nelbo = self.compute_nelbo(x)
        nelbo.backward()

        self.optimizer.step()

        return nelbo.detach()

    def train(self, train_loader: data.DataLoader, val_loader: data.DataLoader, epochs: int, dev: torch.device, save_dir: str = '') -> np.ndarray:
        if save_dir != '':
            save_dir = 'outputs'

        os.makedirs(save_dir, exist_ok=True)

        train_elbos = np.zeros(epochs)
        val_elbos = np.zeros(epochs)

        best_val_elbo = -np.inf

        for e in range(epochs):
            # train on the training loader
            elbo = 0
            n = 0
            for (x, _) in train_loader:
                x = x.to(dev)
                x = flatten_batch(x)
                nelbo = self.learn_step(x)
                # learning step returns negative ELBO -> subtract it
                elbo -= nelbo.item()
                n += x.shape[0]
            train_elbos[e] = elbo / n

            # validate on a validation set
            val_elbo = 0
            n = 0
            for (x, _) in val_loader:
                x = x.to(dev)
                x = flatten_batch(x)
                with torch.no_grad():
                    nelbo = self.compute_nelbo(x)
                    val_elbo -= nelbo.item()
                    n += x.shape[0]
            val_elbos[e] = val_elbo / n

            print(f"Epoch {e}, ELBO: {train_elbos[e]}, val ELBO: {val_elbos[e]}")

            if val_elbos[e] > best_val_elbo:
                best_val_elbo = val_elbos[e]
                self.save(save_dir)

        np.save(os.path.join(save_dir, f'{self.name}_elbo'), train_elbos)
        np.save(os.path.join(save_dir, f'{self.name}_valelbo'), val_elbos)

        return train_elbos

    def loader_elbo(self, loader: data.DataLoader, dev: torch.device) -> float:
        total_elbo = 0
        n = 0
        for (x, _) in loader:
            x = x.to(dev)
            x = flatten_batch(x)
            with torch.no_grad():
                total_elbo -= self.compute_nelbo(x).item()
            n += x.shape[0]
        return total_elbo / n

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        # reshape from (784) to (1, 784)
        x = x.unsqueeze(0)
        with torch.no_grad():
            mu, sigma = self.encoder(x)
        mu = mu.detach().squeeze(0)
        sigma = sigma.detach().squeeze(0)
        N_dist = dstr.MultivariateNormal(loc=mu, covariance_matrix=torch.diag(sigma ** 2))
        z = N_dist.sample()

        with torch.no_grad():
            gen_mu = self.decoder(z)

        return gen_mu

    def collapse(self, x: torch.tensor, dev: torch.device) -> torch.tensor:
        x = x.to(dev)
        x = flatten_batch(x)

        z_mu, z_sigma = self.encoder(x)
        qz = dstr.Normal(z_mu, z_sigma)

        pz = dstr.Normal(torch.zeros_like(z_mu), torch.ones_like(z_mu))

        # KL-Div term per latent variables
        kl_div = dstr.kl_divergence(qz, pz).mean(0)
        return kl_div.detach()

    def random_codes(self, n: int) -> torch.tensor:
        N_dist = dstr.Normal(torch.zeros(self.decoder.zdim), torch.ones(self.decoder.zdim))
        samples = N_dist.sample((n,))

        with torch.no_grad():
            mu = self.decoder(samples)

        return mu

    def limiting_distribution(self, n: int = 64, k: int = 100) -> torch.tensor:
        images = []
        N_dist = dstr.Normal(torch.zeros(self.decoder.zdim), torch.ones(self.decoder.zdim))
        z = N_dist.sample((n,))

        for _ in range(k):
            # decode
            with torch.no_grad():
                mu_x = self.decoder(z)
            images.append(mu_x)

            # sample
            qx_dist = dstr.Normal(mu_x, torch.exp(self.decoder.logsigma))
            x = qx_dist.rsample()

            # encode
            with torch.no_grad():
                mu_z, sigma_z = self.encoder(x)

            # sample new sample
            qz = dstr.Normal(mu_z, sigma_z)
            z = qz.rsample()

        images = torch.stack(images)

        return images

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, f'{self.name}.pt'))

    def load(self, save_dir: str) -> None:
        self.load_state_dict(torch.load(os.path.join(save_dir, f'{self.name}.pt'), map_location=torch.device('cpu')))

    def load_elbos(self, save_dir: str) -> tuple[np.ndarray, np.ndarray]:
        return np.load(os.path.join(save_dir, f'{self.name}_elbo.npy')), np.load(os.path.join(save_dir, f'{self.name}_valelbo.npy'))


if __name__ == '__main__':
    dev = select_device()
    print(f'training on {dev}')

    batch_size = 16
    epochs = 100
    lr = 3e-4
    zdim = 64

    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())

    # split the training data into training and validation sets
    m = len(train_data)
    shuffled_indices = torch.randperm(m).detach().numpy()
    split_idx = int(m * 0.2)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=0,
                                               sampler=torch.utils.data.SubsetRandomSampler(shuffled_indices[split_idx:]))
    val_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=0,
                                             sampler=torch.utils.data.SubsetRandomSampler(shuffled_indices[:split_idx]))

    os.makedirs('outputs', exist_ok=True)

    print('## train vanilla')
    vanilla = VAE(name='vanilla', zdim=zdim, lr=lr).to(dev)
    print(vanilla)
    vanilla_elbos = vanilla.train(train_loader, val_loader, epochs=epochs, dev=dev, save_dir='outputs')

    layer_sizes = [
        [512], [256], [128],
        [512, 256], [256, 128],
        [512, 256, 128],
    ]

    for layers in layer_sizes:
        code = '_'.join(str(l) for l in layers)
        encoder_hidden = layers
        decoder_hidden = layers[::-1]
        print(f'## train {code} (encoder {encoder_hidden}) (decoder {decoder_hidden})')
        deep = VAE(name=f'deep_{code}', zdim=zdim, lr=lr, encoder_hidden=encoder_hidden, decoder_hidden=decoder_hidden).to(dev)
        print(deep)
        deep_elbos = deep.train(train_loader, val_loader, epochs=epochs, dev=dev, save_dir='outputs')
