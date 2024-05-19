# %%
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models import VAE
from view import plot_elbo, show_reconstructed, show_collapse, show_decoder, show_animation
from helpers import select_device

# %%
# vanilla = VAE(name='vanilla', zdim=64, lr=3e-4)
# vanilla.load('outputs')
# deep128 = VAE(name='deep_128', zdim=64, lr=3e-4, encoder_hidden=[128], decoder_hidden=[128])
# deep128.load('outputs')
# deep256 = VAE(name='deep_256', zdim=64, lr=3e-4, encoder_hidden=[256], decoder_hidden=[256])
# deep256.load('outputs')
# deep512 = VAE(name='deep_512', zdim=64, lr=3e-4, encoder_hidden=[512], decoder_hidden=[512])
# deep512.load('outputs')
# deep256_128 = VAE(name='deep_256_128', zdim=64, lr=3e-4, encoder_hidden=[256, 128], decoder_hidden=[128, 256])
# deep256_128.load('outputs')
# deep512_256 = VAE(name='deep_512_256', zdim=64, lr=3e-4, encoder_hidden=[512, 256], decoder_hidden=[256, 512])
# deep512_256.load('outputs')
# deep512_256_128 = VAE(name='deep_512_256_128', zdim=64, lr=3e-4, encoder_hidden=[512, 256, 128], decoder_hidden=[128, 256, 512])
# deep512_256_128.load('outputs')


# list_all = [vanilla, deep128, deep256, deep512, deep256_128, deep512_256, deep512_256_128]


deep = VAE(name='deep_128_64_32_16', zdim=10, lr=3e-4, encoder_hidden=[128, 64, 32, 16], decoder_hidden=[16, 32, 64, 128])
deep.load('outputs_small')
list_all = [deep]
# %%
# print('Vanilla VAE parameters:', vanilla.get_param_count())
# print('Deep 128 VAE parameters:', deep128.get_param_count())
# print('Deep 256 VAE parameters:', deep256.get_param_count())
# print('Deep 512 VAE parameters:', deep512.get_param_count())
# print('Deep 256_128 VAE parameters:', deep256_128.get_param_count())
# print('Deep 512_256 VAE parameters:', deep512_256.get_param_count())
# print('Deep 512_256_128 VAE parameters:', deep512_256_128.get_param_count())

print('Deep 128_64_32_16 VAE parameters:', deep.get_param_count())

# %%
plot_elbo(list_all)

# %%

test_data = datasets.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())

# %% reconstruct images
test_loader = DataLoader(test_data, batch_size=16, shuffle=True, num_workers=0)
test_images, _ = next(iter(test_loader))

show_reconstructed(test_images, list_all)

# %% evaluate test errors
dev = select_device()

test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=0)
# print('Vanilla VAE test elbo:', vanilla.loader_elbo(test_loader, dev))
# print('Deep 128 test elbo:', deep128.loader_elbo(test_loader, dev))
# print('Deep 256 test elbo:', deep256.loader_elbo(test_loader, dev))
# print('Deep 512 test elbo:', deep512.loader_elbo(test_loader, dev))
# print('Deep 256_128 test elbo:', deep256_128.loader_elbo(test_loader, dev))
# print('Deep 512_256 test elbo:', deep512_256.loader_elbo(test_loader, dev))
# print('Deep 512_256_128 test elbo:', deep512_256_128.loader_elbo(test_loader, dev))
print('Deep 128_64_32_16 test elbo:', deep.loader_elbo(test_loader, dev))

# %% posterior collapse
test_loader = DataLoader(test_data, batch_size=256, shuffle=True, num_workers=0)
test_images, _ = next(iter(test_loader))

show_collapse(test_images, list_all, dev)

# %% evaluating the decoder
show_decoder(list_all)

# %% limiting distribution
show_animation(list_all)
