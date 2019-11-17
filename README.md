# deepgen

**Deep convolutional neural networks for generative modeling (VAE, GANs, etc.)**

## Requirements
Tested with:
- `python==3.7`
- `torch==1.3`
- `torchvision===0.4`
- `numpy==1.16`
- `matplotlib==3.1`
- `pandas==0.25`

## Installation
1. Install dependencies: `conda env create -f environment.yml`
2. Install with setup.py: `python setup.py install` (replace `install` with `develop` to have code changes take immediate effect)

## Simple usage
To train a ResNet-style Variational Autoencoder on the CelebA dataset: `python cnn/train_vae.py` (add `-h` for available arguments)
