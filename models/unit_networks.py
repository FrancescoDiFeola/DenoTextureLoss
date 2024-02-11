import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def ada_in(content_feature, mean_s, std_s, epsilon=1e-5):
    # Calculate mean and standard deviation of content feature
    mean_c = torch.mean(content_feature, dim=(2, 3), keepdim=True)
    std_c = torch.std(content_feature, dim=(2, 3), keepdim=True) + epsilon

    # Broadcast mean_s and std_s to match content_feature's batch size
    mean_s = mean_s.unsqueeze(2).unsqueeze(3)
    std_s = std_s.unsqueeze(2).unsqueeze(3)

    # Apply AdaIN formula
    print("Mean_s:")
    print(mean_s.shape)
    print("content:")
    print(content_feature.shape)
    normalized_content = std_s * (content_feature - mean_c) / (std_c + 1e-5) + mean_s
    print("normalized:")
    print(normalized_content.shape)
    return normalized_content


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = int(n_epochs)
        self.offset = int(offset)
        self.decay_start_epoch = int(decay_start_epoch)

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Encoder(nn.Module):
    def __init__(self, in_channels=3, dim=64, n_downsample=2, shared_block=None):
        super(Encoder, self).__init__()

        # Initial convolution block
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, dim, 7),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Downsampling
        for _ in range(n_downsample):
            layers += [
                nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim * 2),
                nn.ReLU(inplace=True),
            ]
            dim *= 2

        # Residual blocks
        for _ in range(3):
            layers += [ResidualBlock(dim)]

        self.model_blocks = nn.Sequential(*layers)
        self.shared_block = shared_block

    def reparameterization(self, mu):
        Tensor = torch.cuda.FloatTensor if mu.is_cuda else torch.FloatTensor
        z = Variable(Tensor(np.random.normal(0, 1, mu.shape)))
        return z + mu

    def forward(self, x):
        x = self.model_blocks(x)
        mu = self.shared_block(x)
        z = self.reparameterization(mu)
        return mu, z


class Generator(nn.Module):
    def __init__(self, out_channels=3, dim=64, n_upsample=2, shared_block=None):
        super(Generator, self).__init__()

        self.shared_block = shared_block

        layers = []
        dim = dim * 2 ** n_upsample
        # Residual blocks
        for _ in range(3):
            layers += [ResidualBlock(dim)]

        # Upsampling
        for _ in range(n_upsample):
            layers += [
                nn.ConvTranspose2d(dim, dim // 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim // 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            dim = dim // 2

        # Output layer
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(dim, out_channels, 7), nn.Tanh()]

        self.model_blocks = nn.Sequential(*layers)

    def forward(self, x):
        x = self.shared_block(x)
        x = self.model_blocks(x)
        return x


class Generator_AdaIn(nn.Module):
    def __init__(self, out_channels=3, dim=64, n_upsample=2, shared_block=None):
        super(Generator_AdaIn, self).__init__()

        self.shared_block = shared_block

        layers = []
        dim = dim * 2 ** n_upsample
        # Residual blocks
        for _ in range(3):
            layers += [ResidualBlock(dim)]

        # Upsampling
        for _ in range(n_upsample):
            layers += [
                nn.ConvTranspose2d(dim, dim // 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim // 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            dim = dim // 2

        # Output layer
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(dim, out_channels, 7), nn.Tanh()]

        self.model_blocks = nn.Sequential(*layers)

    def forward(self, x, mean_s, std_s):
        x = ada_in(x, mean_s, std_s)
        x = self.shared_block(x)
        x = self.model_blocks(x)
        return x


#################################
#           STYLE ENCODER
#################################
class StyleEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=256, num_conv=8):
        super(StyleEncoder, self).__init__()

        self.p_gamma = torch.tensor(0.0)
        self.p_beta = torch.tensor(0.0)

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        ]

        for _ in range(num_conv):
            layers += [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ]

        layers += [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, 2 * out_channels),  # Output gamma and beta
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        output = self.model(x)

        self.p_gamma = 0.95 * self.p_gamma + 0.05 * output[..., :output.size(-1) // 2].detach()
        self.p_beta = 0.95 * self.p_beta + 0.05 * output[..., output.size(-1) // 2:].detach()

        return self.p_gamma, self.p_beta  # beta and gamma, respectively in the paper


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        channels, height, width = input_shape
        # Calculate output of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 3, padding=1)
        )

    def forward(self, img):
        return self.model(img)


if __name__ == "__main__":
    d = Encoder()
    t = torch.ones((1, 3, 256, 256))
    print(t.size(0))
    z, t = d(t)
