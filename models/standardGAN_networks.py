import torch.nn as nn
import torch.nn.functional as F
import torch


def ada_in(content_feature, mean_s, std_s, epsilon=1e-5):
    # Calculate mean and standard deviation of content feature
    mean_c = torch.mean(content_feature, dim=(2, 3), keepdim=True)
    std_c = torch.std(content_feature, dim=(2, 3), keepdim=True) + epsilon

    # Apply AdaIN formula
    normalized_content = std_s.unsqueeze(2).unsqueeze(3) * (content_feature - mean_c) / (std_c + 1e-5) + mean_s.unsqueeze(2).unsqueeze(3)

    return normalized_content


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


###############################
#           RESNET
###############################
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(in_features, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(in_features, affine=True, track_running_stats=True),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


#################################
#           STYLE ENCODER
#################################
class StyleEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=256, num_conv=6):
        super(StyleEncoder, self).__init__()

        self.p_gamma = torch.tensor(0.0, requires_grad=False)
        self.p_beta = torch.tensor(0.0, requires_grad=False)

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


#################################
#           CONTENT ENCODER
#################################
class ContentEncoder(nn.Module):
    def __init__(self, img_shape=(3, 128, 128), res_blocks=3, c_dim=5):
        super(ContentEncoder, self).__init__()
        channels, img_size, _ = img_shape

        # Initial convolution block
        model = [
            nn.Conv2d(channels + c_dim, 64, 7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        curr_dim = 64
        for _ in range(2):
            model += [
                nn.Conv2d(curr_dim, curr_dim * 2, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ]
            curr_dim *= 2

        # Residual blocks
        for _ in range(res_blocks):
            model += [ResidualBlock(curr_dim)]

        self.model = nn.Sequential(*model)

    def forward(self, x, c1, c2, train=True):

        if train:
            c1 = c1.view(c1.size(0), c1.size(1), 1, 1)
            c1 = c1.repeat(1, 1, x.size(2), x.size(3))  # it ensures that the conditioning information c has the same shape as the input image tensor x along the batch, channel, height, and width dimensions
            x1 = torch.cat((x, c1), 1)
            c2 = c2.view(c2.size(0), c2.size(1), 1, 1)
            c2 = c2.repeat(1, 1, x.size(2), x.size(3))
            x2 = torch.cat((x, c2), 1)
            return self.model(x1), self.model(x2)
        else:
            return self.model(x)


#################################
#           DECODER
#################################
class ContentDecoder(nn.Module):
    def __init__(self, channels, curr_dim):
        super(ContentDecoder, self).__init__()

        # AdaIN parameters initialization
        self.mean_s = 0
        self.std_s = 0
        model = []

        # First deconvolution
        model += [
            nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        ]
        curr_dim = curr_dim // 2

        # Second deconvolution
        model += [
            nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        ]
        curr_dim = curr_dim // 2

        # Final convolution layer
        model += [
            nn.Conv2d(curr_dim, channels, 7, stride=1, padding=3),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x, mean_s, std_s):

        normalized_features = ada_in(x, mean_s, std_s)

        return self.model(normalized_features)


##############################
#        DISCRIMINATOR
##############################
class Discriminator(nn.Module):
    def __init__(self, img_shape=(3, 128, 128), c_dim=5, n_strided=6):
        super(Discriminator, self).__init__()
        channels, img_size, _ = img_shape

        def discriminator_block(in_filters, out_filters):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1), nn.LeakyReLU(0.01)]
            return layers

        layers = discriminator_block(channels, 64)
        curr_dim = 64
        for _ in range(n_strided - 1):
            layers.extend(discriminator_block(curr_dim, curr_dim * 2))
            curr_dim *= 2

        self.model = nn.Sequential(*layers)

        # Output 1: PatchGAN
        self.out1 = nn.Conv2d(curr_dim, 1, 3, padding=1, bias=False)
        # Output 2: Class prediction
        kernel_size = img_size // 2 ** n_strided
        self.out2 = nn.Conv2d(curr_dim, c_dim, kernel_size, bias=False)

    def forward(self, img):
        feature_repr = self.model(img)
        out_adv = self.out1(feature_repr)
        out_cls = self.out2(feature_repr)
        return out_adv, out_cls.view(out_cls.size(0), -1)

if __name__ == "__main__":
    import numpy as np
    d = Discriminator(img_shape=(3, 256, 256), c_dim=2, n_strided=6)
    t = torch.ones((1, 3, 256, 256))


