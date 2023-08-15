import matplotlib.pyplot as plt
from torchvision.models import inception_v3
import pickle
import scipy.linalg
import torch
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from util.util import save_ordered_dict_as_csv

def list_iterator(data, batch_size):
    for i in tqdm(range(0, len(data), batch_size)):
        yield data[i:i + batch_size]  # When a function contains the yield keyword, it becomes a generator function.


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,  # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=False,
                 requires_grad=False,
                 device="cpu"):

        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        inception = models.inception_v3(pretrained=True).to(device)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp


class WrapperInceptionV3(nn.Module):

    def __init__(self, fid_incv3):
        super().__init__()
        self.fid_incv3 = fid_incv3

    def prepare_img(self, img, standardize=False):
        # Rescale in range [0 1]
        img = (img + 1) / 2
        img = img.clamp(0, 1)
        # Make tree channel.
        if img.shape[1] == 1:
            img = img.repeat([1, 3, 1, 1])  # make a three channel tensor
        # Normalize
        if standardize:
            img = (img - self.mean) / self.std  # Normalize img

        return img

    @torch.no_grad()
    def forward(self, x):
        x = self.prepare_img(x)
        y = self.fid_incv3(x)
        y = y[0]
        y = y[:, :, 0, 0]
        return y


# ----------------------------------------------------------------------------

class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64  # @ applies matrix multiplication.

    def append_torch(self, x):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = pickle.load(f)
        obj = FeatureStats(capture_all=s['capture_all'], max_items=s['max_items'])
        obj.__dict__.update(s)
        return obj


# ----------------------------------------------------------------------------

class GANMetrics:

    def __init__(self, device, detector_name='inceptionv3', batch_size=64):

        self.device = device
        self.detector_name = detector_name
        self.batch_size = batch_size

        if self.detector_name == 'inceptionv3':
            dims = 2048
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
            model = InceptionV3([block_idx]).to(self.device)

            # wrapper model to pytorch_fid model
            wrapper_model = WrapperInceptionV3(model)
            self.detector = wrapper_model.eval()
        else:
            raise NotImplementedError

    def compute_feature(self, imgs, max_items, **stats_kwargs):

        # Initialize.
        stats = FeatureStats(max_items=max_items, **stats_kwargs)
        assert stats.max_items is not None

        imgs_iterator = list_iterator(imgs, self.batch_size)

        # Main loop.
        for img in imgs_iterator:
            img = img.to(self.device)
            feat = self.detector(img)
            stats.append_torch(feat)

        return stats

    def compute_fid(self, imgs0, imgs1, n_samples):
        """
        Frechet Inception Distance (FID) from the paper
        "GANs trained by a two time-scale update rule converge to a local Nash
        equilibrium". Matches the original implementation by Heusel et al. at
        https://github.com/bioinf-jku/TTUR/blob/master/fid.py
        """

        stats0 = self.compute_feature(
            imgs=imgs0, max_items=n_samples, capture_mean_cov=True
        )
        mu0, sigma0 = stats0.get_mean_cov()

        stats1 = self.compute_feature(
            imgs=imgs1, max_items=n_samples, capture_mean_cov=True
        )

        mu1, sigma1 = stats1.get_mean_cov()

        m = np.square(mu1 - mu0).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma1, sigma0), disp=False)  # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma1 + sigma0 - s * 2))

        return float(fid)


if __name__ == '__main__':
    weight = np.load("/Users/francescodifeola/Desktop/pix2pix_results/losses_ep50/loss_pix2pix_texture_att_window_13/weight.npy")
    plt.plot(range(0,16800), weight)
    plt.show()
    print(weight.shape)

    fake_buffer = torch.load('/Volumes/Untitled/test_pix2pix_perceptual_window_5/fake_buffer_test_1_epoch50.pth', map_location=torch.device('cpu'))
    real_buffer = torch.load('/Volumes/Untitled/test_pix2pix_perceptual_window_5/real_buffer_test_1_epoch50.pth',  map_location=torch.device('cpu'))
    print(fake_buffer.shape)
    print(real_buffer.shape)

    '''for i in range(5):
        plt.imshow(real_buffer[i, 0, :, :], cmap='gray')
        plt.show()
        plt.imshow(fake_buffer[i, 0, :, :], cmap='gray')
        plt.show()'''

    metric_obj = GANMetrics('cpu', detector_name='inceptionv3', batch_size=64)
    fid = metric_obj.compute_fid(fake_buffer, real_buffer, 560)
    fid = {'fid': fid}
    save_ordered_dict_as_csv(fid, "/Volumes/Untitled/test_pix2pix_perceptual_window_5/fid_test_1.csv")

