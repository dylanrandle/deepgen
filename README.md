# deepgen

*Deep convolutional neural networks for generative modeling (i.e. VAEs, GANs, etc.)*

![vae_gif](examples/Gif-2019-52-26-09-52-24.gif)

## Requirements
Tested with:
- python==3.7
- torch==1.3
- torchvision==0.4
- numpy==1.16
- matplotlib==3.1
- pandas==0.25

All dependencies are listed in `environment.yml`.

## Installation
1. Install dependencies: `conda env create -f environment.yml`
2. Install with setup.py: `python setup.py install` (replace `install` with `develop` to have code changes take immediate effect)

## Usage
Run `python cnn/run_vae.py -h` to see a list of available arguments.

### Train ResNet-style (Conditional) Variational Autoencoder:
- `python cnn/run_vae.py --train --save_examples`

### Test Trained Model:
- `python cnn/run_vae.py --test --save_examples --model_path /path/to/model`

*Note:* be careful to set `--report_freq` and `--batch_size` properly. The former specifies how often to save examples and report losses (i.e. controls how spammy the output is), while the latter specifies how many examples will be batched together and must be appropriate for your computational and memory resources (I like batch size 1 for local development and use batch size 32 on a GPU,  but for performance the largest possible batch size is best).

## References
- [Deep Conditional Generative Models](https://pdfs.semanticscholar.org/3f25/e17eb717e5894e0404ea634451332f85d287.pdf)
