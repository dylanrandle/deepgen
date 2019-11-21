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

Note: set `--report_freq` and `--batch_size` properly according to your needs. The former specifies how often to save examples and report losses, while the latter must be appropriate for the computational and memory resources you are running on (i.e. locally, use batch size 1, but on a GPU maybe 32, 64, etc.). 

- To train a ResNet-style (Conditional) Variational Autoencoder on the CelebA dataset: `python cnn/run_vae.py --train --save_examples` (add `-h` for available arguments)
- To test a trained model: `python cnn/run_vae.py --test --save_examples --model_path /path/to/you/fancy/model`
