import numpy as np
from math import log10

import piq
from scipy import fftpack
from skimage.metrics import structural_similarity
from sewar.full_ref import vifp
import os
from piq import SSIMLoss

def mean_squared_error(x, y):
    # MSE
    return np.square(np.subtract(x, y)).mean()


def peak_signal_to_noise_ratio(x, y):
    mse = mean_squared_error(x, y)

    if mse == 0:  # MSE is zero means no noise is present in the signal. Therefore, PSNR have no importance.
        return 100

    max_pixel = 1

    # PSNR
    return 10 * log10((max_pixel ** 2) / mse)


def structural_similarity_index(x, y):
    # SSIM
    return structural_similarity(x, y, data_range=2)


def vif(x, y):
    # VIF
    return vifp(x, y)


# COMPUTE THE AZIMUTHALLY AVERAGED RADIAL PRIFILE
def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fracitonal pixels).
    """
    # compute the 2D discrete transform
    F_img = fftpack.fftshift(fftpack.fft2(image))
    # power spectral density
    psd2D_img = np.abs(F_img) ** 2

    # Calculate the indices from the image (y: rows, x: columns)
    y, x = np.indices(psd2D_img.shape)

    # calculate the center of the image
    if not center:
        center = np.array([(x.max() - x.min()) / 2.0, (x.max() - x.min()) / 2.0])

    # calculate the hypotenusa for each position respect to the center
    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)  # .flat is a 1-D iterator over the array.
    r_sorted = r.flat[ind]
    i_sorted = psd2D_img.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]  # consider the location of changed radius
    nr = rind[1:] - rind[:-1]  # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof


def element_wise_average(lists):
    list_lengths = [len(sublist) for sublist in lists[0]]

    if any(length != list_lengths[0] for length in list_lengths):
        raise ValueError("All lists must have the same length")

    result = []
    for sublists in zip(*lists):
        if any(len(sublist) != len(sublists[0]) for sublist in sublists):
            raise ValueError("Sublists must have the same length")

        avg_sublist = [sum(values) / len(values) for values in zip(*sublists)]
        result.append(avg_sublist)

    return result


def compute_average_profiles(folder_path, **kwargs):

    path_list = []
    for _, path in kwargs.items():
        path_list.append(os.join(folder_path, path))

    profiles = []
    for i in path_list:
        experiment_profile = open(i)
        profiles.append(experiment_profile)

    return element_wise_average(profiles)





if __name__ == "__main__":
    import torch
    from data.storage import load_from_json
    from skimage.feature import graycomatrix, graycoprops
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import skimage as ski

    x = torch.rand(4, 3, 256, 256, requires_grad=True)

    y = torch.rand(4, 3, 256, 256, requires_grad=True)

    ssim_loss = SSIMLoss(data_range=1)
    print(ssim_loss(x, y))

    """import torch
    import matplotlib.pyplot as plt
    import pylab as py
    import json
    # fake_buffer = torch.load('/Volumes/Untitled/test_pix2pix_perceptual_window_5/fake_buffer_test_1_epoch50.pth', map_location=torch.device('cpu'))
    # real_buffer = torch.load('/Volumes/Untitled/test_pix2pix_perceptual_window_5/real_buffer_test_1_epoch50.pth', map_location=torch.device('cpu'))

    raps_ld = open(f'/Users/francescodifeola/PycharmProjects/DenoTextureLoss/raps_ELCAP_ld.json')
    raps_ld = json.load(raps_ld)

    result = []
    for sublist1, sublist2, sublist3 in zip(raps_ld, raps_ld, raps_ld):
        if len(sublist1) != len(sublist2) or len(sublist1) != len(sublist3):
            raise ValueError("Sublists must have the same length")


        avg_sublist = [(x + y + z) / 3 for x, y, z in zip(sublist1, sublist2, sublist3)]
        result.append(avg_sublist)


    mean_profile_ld = [np.mean(k) for k in zip(*result)]

    # deno_img = azimuthalAverage(np.squeeze(fake_buffer[200, 0, :, :].cpu().detach().numpy()))
    # hd_img = azimuthalAverage(np.squeeze(real_buffer[200, 0, :, :].cpu().detach().numpy()))
    py.semilogy(mean_profile_ld, color='red')
    # py.semilogy(hd_img, color='blue')
    py.show()"""
