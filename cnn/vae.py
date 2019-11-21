import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import utils

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEFAULT_MODEL_PATH = 'vae_resnet_celebA.pt'

# Follows elements of https://github.com/pytorch/examples/blob/master/vae/main.py
class AutoEncoder(nn.Module):
    """
    Implements a (conditional) variational auto-encoder
    Uses ResNet-style blocks of convolutions
    Concatenates CelebA attributes to latent dimension
    """
    def __init__(self, latent_dim = 128, condition_dim = 40):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1, dilation = 1, ceil_mode = False)

        # first encoding layer (similar to first layer of original ResNet)
        self.encode1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            self.relu,
            self.maxpool,
        )

        # second encoding block
        self.encode2 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
        )

        # third encoding block
        self.encode3 = nn.Sequential(
            ResidualBlock(64, 128,
                residual_op = nn.Sequential(conv3x3(64, 128), nn.BatchNorm2d(128))
            ),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            self.maxpool,
        )

        # projecting the last feature map into latent space (linear)
        self.flat = nn.Flatten()
        self.mu_latent = nn.Linear(128 * 32 * 32, self.latent_dim)
        self.sig_latent = nn.Linear(128 * 32 * 32, self.latent_dim)

        # projecting out of latent space (non-linear)
        self.project_z = nn.Sequential(
            nn.Linear(self.latent_dim + self.condition_dim, 32 * 32),
            nn.ReLU(inplace=True),
        )

        # first decoding block
        self.decode1 = nn.Sequential(
            nn.Upsample(scale_factor = (2,2)),
            ResidualBlock(1, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
        )

        # second decoding block
        self.decode2 = nn.Sequential(
            nn.Upsample(scale_factor = (2,2)),
            ResidualBlock(128, 64,
                residual_op = nn.Sequential(conv3x3(128, 64), nn.BatchNorm2d(64))
            ),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
        )

        # second decoding block
        self.decode3 = nn.Sequential(
            nn.Upsample(scale_factor = (2,2)),
            ResidualBlock(64, 32,
                residual_op = nn.Sequential(conv3x3(64, 32), nn.BatchNorm2d(32))
            ),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
        )

        # final decoding block
        self.decode4 = nn.Sequential(
            conv3x3(32, 3),
            nn.Sigmoid(),  # sigmoid to make pixels lie in (0, 1)
        )

    def reparameterize(self, mu, std):
        """
        # previous reference to `sig` is really log variance, used for numerical stability
        here make conversion and apply the reparameterization trick, i.e. N(mu, sig) = mu + N(0, 1) * sig
        """
        # std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        out = self.encode1(x)     # 2x stride 2 => reduce dim by 4x
        out = self.encode2(out)   # 1x stride 2 => reduce dim by 2x
        out = self.encode3(out)   # 1x stride 2 => reduce dim by 2x
        out = self.flat(out)
        mu = self.mu_latent(out)
        sig = self.sig_latent(out)
        return mu, sig

    def decode(self, z):
        # latent space operations: project + reshape
        zout = self.project_z(z)
        batch_size, new_dim = int(zout.shape[0]), int(np.sqrt(zout.shape[1]))
        zout = zout.view(batch_size, 1, new_dim, new_dim) # reshape to be "image-like"
        # decoding layers
        out = self.decode1(zout)
        out = self.decode2(out)
        out = self.decode3(out)
        out = self.decode4(out)
        return out

    def forward(self, x, attr):
        # encoding
        mu, sig = self.encode(x)
        # operations in latent space: sample + concat attributes
        z = self.reparameterize(mu, sig)
        z_cond = torch.cat((z, attr), 1) # add attributes to z
        # decoding
        out = self.decode(z_cond)
        return out, mu, sig

# 3x3 convolution: from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

# Residual block: from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, residual_op=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.residual_op = residual_op

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.residual_op:
            residual = self.residual_op(x)
        out += residual
        out = self.relu(out)
        return out

MSE_LOSS = torch.nn.MSELoss().to(DEVICE)
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, sig):
    RECON = MSE_LOSS(recon_x, x)
    # see Appendix B from VAE paper:
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + torch.log(sig.pow(2)) - mu.pow(2) - sig.pow(2))
    return RECON + KLD

def train(train_loader, model_path = None, num_epochs = 10, seed = 42, report_freq = 100, save_examples = False):
    """ runs training for AutoEncoder model
        if model_path not provided, training starts from scratch """
    torch.manual_seed(seed)
    if model_path:
        ae = load_model(model_path = model_path)
    else:
        # model_path not provided => will save to default path
        # check if default path exists and ask user what to do
        if os.path.exists(DEFAULT_MODEL_PATH):
            print(f'Detected a model already saved at {DEFAULT_MODEL_PATH} (the default model path)')
            print('Would you like to resume (r) training of this model or start from scratch and overwrite (o) it?')
            answer = input('Resume (r) / Overwrite (o): ').strip().lower()
            if answer == 'r':
                ae = load_model(model_path = DEFAULT_MODEL_PATH)
            elif answer == 'o':
                ae = create_model()
            else:
                raise ValueError(f'You response "{answer}" was not understood.')
        else: # no concern of overwriting
            ae = create_model()

    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)

    # Train the model
    total_step = len(train_loader)
    losses = []
    for epoch in range(num_epochs):
        for i, (img, attr) in enumerate(train_loader):
            img = img.to(DEVICE)
            attr = attr.to(DEVICE)

            # Forward pass
            gen_img, mu, sig = ae(img, attr)
            loss = loss_function(gen_img, img, mu, sig)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if (i+1) % report_freq == 0:
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                if save_examples:
                    save_to = f'TrainExamples_Epoch{epoch+1}_Step{i+1}.png'
                    utils.make_examples(img, gen_img, save_to = save_to)

        # Save model after each epoch
        save_to = model_path if model_path else DEFAULT_MODEL_PATH # use this default path if None provided
        torch.save(ae.state_dict(), save_to)
        print(f'Saved model to {save_to}')

def test(test_loader, model_path, report_freq = 100, save_examples = False):
    """ runs a saved model over test data. user must provide model_path. """
    ae = load_model(model_path = model_path)
    ae.eval()

    total_step = len(test_loader)
    test_loss = 0
    with torch.no_grad():
        for i, (img, attr) in enumerate(test_loader):
            img = img.to(DEVICE)
            gen_img, mu, sig = ae(img, attr)
            loss = loss_function(gen_img, img, mu, sig)
            test_loss += loss.item()
            if (i + 1) % report_freq == 0:
                print("Step [{}/{}] Test Loss: {:.4f}".format(i+1, total_step, loss.item()))
                if save_examples:
                    save_to = f'TestExamples_Step{i+1}.png'
                    utils.make_examples(img, gen_img, save_to = save_to)

    test_loss /= total_step
    print('====> Test set loss: {:.4f}'.format(test_loss))

def load_model(model_path):
    """ helper function to load model if given path exists
        otherwise creates new object and returns it """
    if not os.path.exists(model_path):
        raise RuntimeError(f'Could not find provided model_path: {model_path}')

    ae = AutoEncoder()
    ae.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f'Loaded existing model from {model_path}')

    total_params = utils.count_parameters(ae)
    print(f'Model has {total_params} parameters')
    ae = ae.to(DEVICE)
    return ae

def create_model():
    """ helper function to instantiate new model """
    ae = AutoEncoder()
    print('Loaded new AutoEncoder model')
    total_params = utils.count_parameters(ae)
    print(f'Model has {total_params} parameters')
    ae = ae.to(DEVICE)
    return ae
