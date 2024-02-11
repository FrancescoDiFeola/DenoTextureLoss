import os
from options.test_options import TestOptions
from data import create_dataset, BaseDataset
from models import create_model
from util.visualizer import save_images
from util import html
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer_offline import Visualizer
from util.util import save_ordered_dict_as_csv
from tqdm import tqdm
import torch
from metrics.FID import *
import matplotlib as mpl
import importlib
from data.storage import load_from_json
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # cycleGAN (time it takes to compute the loss function and perform the backpropagation)
    print("CycleGAN")
    time_baseline = np.load("/Volumes/sandisk/times_denoising/metrics_cycleGAN_baseline_time/time.npy")
    print(len(time_baseline))
    print(f"Time baseline:{np.mean(time_baseline[5:])}")
    time_texture_max = np.load("/Volumes/sandisk/times_denoising/metrics_cycleGAN_texture_max_time/time.npy")
    print(f"Time texture max:{np.mean(time_texture_max[5:])}")
    time_texture_avg = np.load("/Volumes/sandisk/times_denoising/metrics_cycleGAN_texture_avg_time/time.npy")
    print(f"Time texture avg:{np.mean(time_texture_avg[5:])}")
    time_texture_Frob = np.load("/Volumes/sandisk/times_denoising/metrics_cycleGAN_texture_Frob_time/time.npy")
    print(f"Time texture Frob:{np.mean(time_texture_Frob[5:])}")
    time_texture_att = np.load("/Volumes/sandisk/times_denoising/metrics_cycleGAN_texture_att_time/time.npy")
    print(f"Time texture att:{np.mean(time_texture_att[5:])}")
    time_perceptual = np.load("/Volumes/sandisk/times_denoising/metrics_cycleGAN_perceptual_time/time.npy")
    print(f"Time perceptual:{np.mean(time_perceptual[5:])}")
    time_ssim = np.load("/Volumes/sandisk/times_denoising/metrics_cycleGAN_ssim_time/time.npy")
    print(f"Time ssim:{np.mean(time_ssim)}")
    time_edge = np.load("/Volumes/sandisk/times_denoising/metrics_cycleGAN_edge_time/time.npy")
    print(f"Time EDGE:{np.mean(time_edge)}")
    time_ae_CT = np.load("/Volumes/sandisk/times_denoising/metrics_cycleGAN_autoencoder_time/time.npy")
    print(f"Time AE-CT:{np.mean(time_ae_CT)}")
    print(f"###########################################")
    print("Pix2Pix")
    time_baseline = np.load("/Volumes/sandisk/times_denoising/metrics_pix2pix_baseline_time/time.npy")
    print(f"Time baseline:{np.mean(time_baseline)}")
    time_texture_max = np.load("/Volumes/sandisk/times_denoising/metrics_pix2pix_texture_max_time/time.npy")
    print(f"Time texture max:{np.mean(time_texture_max)}")
    time_texture_avg = np.load("/Volumes/sandisk/times_denoising/metrics_pix2pix_texture_avg_time/time.npy")
    print(f"Time texture avg:{np.mean(time_texture_avg)}")
    time_texture_Frob = np.load("/Volumes/sandisk/times_denoising/metrics_pix2pix_texture_Frob_time/time.npy")
    print(f"Time texture Frob:{np.mean(time_texture_Frob)}")
    time_texture_att = np.load("/Volumes/sandisk/times_denoising/metrics_pix2pix_texture_att_time/time.npy")
    print(f"Time texture att:{np.mean(time_texture_att)}")
    time_perceptual = np.load("/Volumes/sandisk/times_denoising/metrics_pix2pix_perceptual_time/time.npy")
    print(f"Time perceptual:{np.mean(time_perceptual)}")
    time_ssim = np.load("/Volumes/sandisk/times_denoising/metrics_pix2pix_ssim_time/time.npy")
    print(f"Time ssim:{np.mean(time_ssim)}")
    time_edge = np.load("/Volumes/sandisk/times_denoising/metrics_pix2pix_edge_time/time.npy")
    print(f"Time EDGE:{np.mean(time_edge)}")
    time_ae_CT = np.load("/Volumes/sandisk/times_denoising/metrics_pix2pix_autoencoder_time/time.npy")
    print(f"Time AE-CT:{np.mean(time_ae_CT)}")
    print(f"###########################################")
    print("UNIT")
    time_baseline = np.load("/Volumes/sandisk/times_denoising/metrics_UNIT_baseline_time/time.npy")
    print(f"Time baseline:{np.mean(time_baseline)}")
    time_texture_max = np.load("/Volumes/sandisk/times_denoising/metrics_UNIT_texture_max_time/time.npy")
    print(f"Time texture max:{np.mean(time_texture_max)}")
    time_texture_avg = np.load("/Volumes/sandisk/times_denoising/metrics_UNIT_texture_avg_time/time.npy")
    print(f"Time texture avg:{np.mean(time_texture_avg)}")
    time_texture_Frob = np.load("/Volumes/sandisk/times_denoising/metrics_UNIT_texture_Frob_time/time.npy")
    print(f"Time texture Frob:{np.mean(time_texture_Frob)}")
    time_texture_att = np.load("/Volumes/sandisk/times_denoising/metrics_UNIT_texture_att_time/time.npy")
    print(f"Time texture att:{np.mean(time_texture_att)}")
    time_perceptual = np.load("/Volumes/sandisk/times_denoising/metrics_UNIT_perceptual_time/time.npy")
    print(f"Time perceptual:{np.mean(time_perceptual)}")
    time_ssim = np.load("/Volumes/sandisk/times_denoising/metrics_UNIT_ssim_time/time.npy")
    print(f"Time ssim:{np.mean(time_ssim)}")
    time_edge = np.load("/Volumes/sandisk/times_denoising/metrics_UNIT_edge_time/time.npy")
    print(f"Time EDGE:{np.mean(time_edge)}")
    time_ae_CT = np.load("/Volumes/sandisk/times_denoising/metrics_UNIT_autoencoder_time/time.npy")
    print(f"Time AE-CT:{np.mean(time_ae_CT)}")
    print(f"###########################################")






    """haralicks_A = load_from_json(f"which_haralick_A")
    haralicks_B = load_from_json(f"which_haralick_B")
    keys = list(haralicks_A.keys())



    A_contrast_1 = [(i + j + k + v)/4 for i, j, k, v in zip(haralicks_A["contrast_1-0"], haralicks_A["contrast_1-45"], haralicks_A["contrast_1-90"], haralicks_A["contrast_1-135"])]  # /len(haralicks["contrast_1-0"])
    A_contrast_3 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_A["contrast_3-0"], haralicks_A["contrast_3-45"], haralicks_A["contrast_3-90"], haralicks_A["contrast_3-135"])]  # / len(haralicks["contrast_3-0"])
    A_contrast_5 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_A["contrast_5-0"], haralicks_A["contrast_5-45"], haralicks_A["contrast_5-90"], haralicks_A["contrast_5-135"])] # / len(haralicks["contrast_5-0"])
    A_contrast_7 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_A["contrast_7-0"], haralicks_A["contrast_7-45"], haralicks_A["contrast_7-90"], haralicks_A["contrast_7-135"])]  # / len(haralicks["contrast_7-0"])
    B_contrast_1 = [(i + j + k + v)/4 for i, j, k, v in zip(haralicks_B["contrast_1-0"], haralicks_B["contrast_1-45"], haralicks_B["contrast_1-90"], haralicks_B["contrast_1-135"])]  # /len(haralicks["contrast_1-0"])
    B_contrast_3 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_B["contrast_3-0"], haralicks_B["contrast_3-45"], haralicks_B["contrast_3-90"], haralicks_B["contrast_3-135"])]  # / len(haralicks["contrast_3-0"])
    B_contrast_5 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_B["contrast_5-0"], haralicks_B["contrast_5-45"], haralicks_B["contrast_5-90"], haralicks_B["contrast_5-135"])] # / len(haralicks["contrast_5-0"])
    B_contrast_7 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_B["contrast_7-0"], haralicks_B["contrast_7-45"], haralicks_B["contrast_7-90"], haralicks_B["contrast_7-135"])]  # / len(haralicks["contrast_7-0"])

    diff_contrast_1 = [(i-j) for i, j in zip(A_contrast_1, B_contrast_1)]
    diff_contrast_3 = [(i-j) for i, j in zip(A_contrast_3, B_contrast_3)]
    diff_contrast_5 = [(i-j) for i, j in zip(A_contrast_5, B_contrast_5)]
    diff_contrast_7 = [(i-j) for i, j in zip(A_contrast_7, B_contrast_7)]
    mean_diff_contrast = (sum(diff_contrast_1)/len(diff_contrast_1) + sum(diff_contrast_3)/len(diff_contrast_3) + sum(diff_contrast_5)/len(diff_contrast_5) + sum(diff_contrast_7)/len(diff_contrast_7))/4
    print(mean_diff_contrast)

    # contrast_avg = (contrast_1 + contrast_3 + contrast_5 + contrast_7)/4
    # plt.plot(sorted(contrast_1))
    # plt.plot(sorted(contrast_3))
    # plt.plot(sorted(contrast_5))
    # plt.plot(sorted(contrast_7))
    # plt.show()
    # print(contrast_avg)
    A_dissimilarity_1 = [(i + j + k + v)/4 for i, j, k, v in zip(haralicks_A["dissimilarity_1-0"], haralicks_A["dissimilarity_1-45"], haralicks_A["dissimilarity_1-90"], haralicks_A["dissimilarity_1-135"])]  # )/len(haralicks["dissimilarity_1-0"])
    A_dissimilarity_3 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_A["dissimilarity_3-0"], haralicks_A["dissimilarity_3-45"], haralicks_A["dissimilarity_3-90"], haralicks_A["dissimilarity_3-135"])]  # ) / len(haralicks["dissimilarity_3-0"])
    A_dissimilarity_5 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_A["dissimilarity_5-0"], haralicks_A["dissimilarity_5-45"], haralicks_A["dissimilarity_5-90"], haralicks_A["dissimilarity_5-135"])]  # ) / len(haralicks["dissimilarity_5-0"])
    A_dissimilarity_7 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_A["dissimilarity_7-0"], haralicks_A["dissimilarity_7-45"], haralicks_A["dissimilarity_7-90"], haralicks_A["dissimilarity_7-135"])]  # ) / len(haralicks["dissimilarity_7-0"])
    B_dissimilarity_1 = [(i + j + k + v)/4 for i, j, k, v in zip(haralicks_B["dissimilarity_1-0"], haralicks_B["dissimilarity_1-45"], haralicks_B["dissimilarity_1-90"], haralicks_B["dissimilarity_1-135"])]  # )/len(haralicks["dissimilarity_1-0"])
    B_dissimilarity_3 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_B["dissimilarity_3-0"], haralicks_B["dissimilarity_3-45"], haralicks_B["dissimilarity_3-90"], haralicks_B["dissimilarity_3-135"])]  # ) / len(haralicks["dissimilarity_3-0"])
    B_dissimilarity_5 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_B["dissimilarity_5-0"], haralicks_B["dissimilarity_5-45"], haralicks_B["dissimilarity_5-90"], haralicks_B["dissimilarity_5-135"])]  # ) / len(haralicks["dissimilarity_5-0"])
    B_dissimilarity_7 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_B["dissimilarity_7-0"], haralicks_B["dissimilarity_7-45"], haralicks_B["dissimilarity_7-90"], haralicks_B["dissimilarity_7-135"])]
    diff_dissimilarity_1 = [(i - j) for i, j in zip(A_dissimilarity_1, B_dissimilarity_1)]
    diff_dissimilarity_3 = [(i - j) for i, j in zip(A_dissimilarity_3, B_dissimilarity_3)]
    diff_dissimilarity_5 = [(i - j) for i, j in zip(A_dissimilarity_5, B_dissimilarity_5)]
    diff_dissimilarity_7 = [(i - j) for i, j in zip(A_dissimilarity_7, B_dissimilarity_7)]
    mean_diff_dissimilarity = (sum(diff_dissimilarity_1) / len(diff_dissimilarity_1) + sum(diff_dissimilarity_3) / len(diff_dissimilarity_3) + sum(diff_dissimilarity_5) / len(diff_dissimilarity_5) + sum(
        diff_dissimilarity_7) / len(diff_dissimilarity_7)) / 4
    print(mean_diff_dissimilarity)

    # dissimilarity_avg = (dissimilarity_1 + dissimilarity_3 + dissimilarity_5 + dissimilarity_7) / 4
    # plt.plot(sorted(dissimilarity_1))
    # plt.plot(sorted(dissimilarity_3))
    # plt.plot(sorted(dissimilarity_5))
    # plt.plot(sorted(dissimilarity_7))
    # plt.show()
    # print(dissimilarity_avg)
    A_homogeneity_1 = [(i + j + k + v)/4 for i, j, k, v in zip(haralicks_A["homogeneity_1-0"], haralicks_A["homogeneity_1-45"], haralicks_A["homogeneity_1-90"], haralicks_A["homogeneity_1-135"])]  # )/len(haralicks["homogeneity_1-0"])
    A_homogeneity_3 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_A["homogeneity_3-0"], haralicks_A["homogeneity_3-45"], haralicks_A["homogeneity_3-90"], haralicks_A["homogeneity_3-135"])]  # ) / len(haralicks["homogeneity_3-0"])
    A_homogeneity_5 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_A["homogeneity_5-0"], haralicks_A["homogeneity_5-45"], haralicks_A["homogeneity_5-90"], haralicks_A["homogeneity_5-135"])]  # ) / len(haralicks["homogeneity_5-0"])
    A_homogeneity_7 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_A["homogeneity_7-0"], haralicks_A["homogeneity_7-45"], haralicks_A["homogeneity_7-90"], haralicks_A["homogeneity_7-135"])] # ) / len(haralicks["homogeneity_7-0"])
    B_homogeneity_1 = [(i + j + k + v)/4 for i, j, k, v in zip(haralicks_B["homogeneity_1-0"], haralicks_B["homogeneity_1-45"], haralicks_B["homogeneity_1-90"], haralicks_B["homogeneity_1-135"])]  # )/len(haralicks["homogeneity_1-0"])
    B_homogeneity_3 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_B["homogeneity_3-0"], haralicks_B["homogeneity_3-45"], haralicks_B["homogeneity_3-90"], haralicks_B["homogeneity_3-135"])]  # ) / len(haralicks["homogeneity_3-0"])
    B_homogeneity_5 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_B["homogeneity_5-0"], haralicks_B["homogeneity_5-45"], haralicks_B["homogeneity_5-90"], haralicks_B["homogeneity_5-135"])]  # ) / len(haralicks["homogeneity_5-0"])
    B_homogeneity_7 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_B["homogeneity_7-0"], haralicks_B["homogeneity_7-45"], haralicks_B["homogeneity_7-90"], haralicks_B["homogeneity_7-135"])]
    diff_homogeneity_1 = [(i - j) for i, j in zip(A_homogeneity_1, B_homogeneity_1)]
    diff_homogeneity_3 = [(i - j) for i, j in zip(A_homogeneity_3, B_homogeneity_3)]
    diff_homogeneity_5 = [(i - j) for i, j in zip(A_homogeneity_5, B_homogeneity_5)]
    diff_homogeneity_7 = [(i - j) for i, j in zip(A_homogeneity_7, B_homogeneity_7)]
    mean_diff_homogeneity = (sum(diff_homogeneity_1) / len(diff_homogeneity_1) + sum(diff_homogeneity_3) / len(diff_homogeneity_3) + sum(diff_homogeneity_5) / len(diff_homogeneity_5) + sum(
        diff_homogeneity_7) / len(diff_homogeneity_7)) / 4
    print(mean_diff_homogeneity)

    # homogeneity_avg = (homogeneity_1 + homogeneity_3 + homogeneity_5 + homogeneity_7)/4
    # plt.plot(sorted(homogeneity_1))
    # plt.plot(sorted(homogeneity_3))
    # plt.plot(sorted(homogeneity_5))
    # plt.plot(sorted(homogeneity_7))
    # plt.show()
    # print(homogeneity_avg)
    A_correlation_1 = [(i + j + k + v)/4 for i, j, k, v in zip(haralicks_A["correlation_1-0"], haralicks_A["correlation_1-45"], haralicks_A["correlation_1-90"], haralicks_A["correlation_1-135"])]  # )/len(haralicks["correlation_1-0"])
    A_correlation_3 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_A["correlation_3-0"], haralicks_A["correlation_3-45"], haralicks_A["correlation_3-90"], haralicks_A["correlation_3-135"])]  # ) / len(haralicks["correlation_3-0"])
    A_correlation_5 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_A["correlation_5-0"], haralicks_A["correlation_5-45"], haralicks_A["correlation_5-90"], haralicks_A["correlation_5-135"])]  # ) / len(haralicks["correlation_5-0"])
    A_correlation_7 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_A["correlation_7-0"], haralicks_A["correlation_7-45"], haralicks_A["correlation_7-90"], haralicks_A["correlation_7-135"])]  # ) / len(haralicks["correlation_7-0"])
    B_correlation_1 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_B["correlation_1-0"], haralicks_B["correlation_1-45"], haralicks_B["correlation_1-90"],
                                                                 haralicks_B["correlation_1-135"])]  # )/len(haralicks["correlation_1-0"])
    B_correlation_3 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_B["correlation_3-0"], haralicks_B["correlation_3-45"], haralicks_B["correlation_3-90"],
                                                                 haralicks_B["correlation_3-135"])]  # ) / len(haralicks["correlation_3-0"])
    B_correlation_5 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_B["correlation_5-0"], haralicks_B["correlation_5-45"], haralicks_B["correlation_5-90"],
                                                                 haralicks_B["correlation_5-135"])]  # ) / len(haralicks["correlation_5-0"])
    B_correlation_7 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_B["correlation_7-0"], haralicks_B["correlation_7-45"], haralicks_B["correlation_7-90"],
                                                                 haralicks_B["correlation_7-135"])]  # ) / len(haralicks["correlation_7-0"])
    diff_correlation_1 = [(i - j) for i, j in zip(A_correlation_1, B_correlation_1)]
    diff_correlation_3 = [(i - j) for i, j in zip(A_correlation_3, B_correlation_3)]
    diff_correlation_5 = [(i - j) for i, j in zip(A_correlation_5, B_correlation_5)]
    diff_correlation_7 = [(i - j) for i, j in zip(A_correlation_7, B_correlation_7)]
    mean_diff_correlation = (sum(diff_correlation_1) / len(diff_correlation_1) + sum(diff_correlation_3) / len(diff_correlation_3) + sum(diff_correlation_5) / len(diff_correlation_5) + sum(
        diff_correlation_7) / len(diff_correlation_7)) / 4
    print(mean_diff_correlation)
    # correlation_avg = (correlation_1 + correlation_3 + correlation_5 + correlation_7)/4
    # plt.plot(sorted(correlation_1))
    # plt.plot(sorted(correlation_3))
    # plt.plot(sorted(correlation_5))
    # plt.plot(sorted(correlation_7))
    # plt.show()
    # print(correlation_avg)
    A_asm_1 = [(i + j + k + v)/4 for i, j, k, v in zip(haralicks_A["ASM_1-0"], haralicks_A["ASM_1-45"], haralicks_A["ASM_1-90"], haralicks_A["ASM_1-135"])]  # )/len(haralicks["ASM_1-0"])
    A_asm_3 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_A["ASM_3-0"], haralicks_A["ASM_3-45"], haralicks_A["ASM_3-90"], haralicks_A["ASM_3-135"])]  # ) / len(haralicks["ASM_3-0"])
    A_asm_5 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_A["ASM_5-0"], haralicks_A["ASM_5-45"], haralicks_A["ASM_5-90"], haralicks_A["ASM_5-135"])]  # ) / len(haralicks["ASM_5-0"])
    A_asm_7 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_A["ASM_7-0"], haralicks_A["ASM_7-45"], haralicks_A["ASM_7-90"], haralicks_A["ASM_7-135"])]  # ) / len(haralicks["ASM_7-0"])
    B_asm_1 = [(i + j + k + v)/4 for i, j, k, v in zip(haralicks_B["ASM_1-0"], haralicks_B["ASM_1-45"], haralicks_B["ASM_1-90"], haralicks_B["ASM_1-135"])]  # )/len(haralicks["ASM_1-0"])
    B_asm_3 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_B["ASM_3-0"], haralicks_B["ASM_3-45"], haralicks_B["ASM_3-90"], haralicks_B["ASM_3-135"])]  # ) / len(haralicks["ASM_3-0"])
    B_asm_5 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_B["ASM_5-0"], haralicks_B["ASM_5-45"], haralicks_B["ASM_5-90"], haralicks_B["ASM_5-135"])]  # ) / len(haralicks["ASM_5-0"])
    B_asm_7 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_B["ASM_7-0"], haralicks_B["ASM_7-45"], haralicks_B["ASM_7-90"], haralicks_B["ASM_7-135"])]  # ) / len(haralicks["ASM_7-0"])
    diff_asm_1 = [(i - j) for i, j in zip(A_asm_1, B_asm_1)]
    diff_asm_3 = [(i - j) for i, j in zip(A_asm_3, B_asm_3)]
    diff_asm_5 = [(i - j) for i, j in zip(A_asm_5, B_asm_5)]
    diff_asm_7 = [(i - j) for i, j in zip(A_asm_7, B_asm_7)]
    mean_diff_asm = (sum(diff_asm_1) / len(diff_asm_1) + sum(diff_asm_3) / len(diff_asm_3) + sum(diff_asm_5) / len(diff_asm_5) + sum(
        diff_asm_7) / len(diff_asm_7)) / 4
    print(mean_diff_asm)
    # asm_avg = (asm_1 + asm_3 + asm_5 + asm_7)/4
    # plt.plot(sorted(asm_1))
    # plt.plot(sorted(asm_3))
    # plt.plot(sorted(asm_5))
    # plt.plot(sorted(asm_7))
    # plt.show()
    # print(asm_avg)
    A_energy_1 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_A["energy_1-0"], haralicks_A["energy_1-45"], haralicks_A["energy_1-90"], haralicks_A["energy_1-135"])]  # ) / len(haralicks["energy_1-0"])
    A_energy_3 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_A["energy_3-0"], haralicks_A["energy_3-45"], haralicks_A["energy_3-90"], haralicks_A["energy_3-135"])]  # ) / len(haralicks["energy_3-0"])
    A_energy_5 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_A["energy_5-0"], haralicks_A["energy_5-45"], haralicks_A["energy_5-90"], haralicks_A["energy_5-135"])]  # ) / len(haralicks["energy_5-0"])
    A_energy_7 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_A["energy_7-0"], haralicks_A["energy_7-45"], haralicks_A["energy_7-90"], haralicks_A["energy_7-135"])]  # ) / len(haralicks["energy_7-0"])
    B_energy_1 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_B["energy_1-0"], haralicks_B["energy_1-45"], haralicks_B["energy_1-90"], haralicks_B["energy_1-135"])]  # ) / len(haralicks["energy_1-0"])
    B_energy_3 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_B["energy_3-0"], haralicks_B["energy_3-45"], haralicks_B["energy_3-90"], haralicks_B["energy_3-135"])]  # ) / len(haralicks["energy_3-0"])
    B_energy_5 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_B["energy_5-0"], haralicks_B["energy_5-45"], haralicks_B["energy_5-90"], haralicks_B["energy_5-135"])]  # ) / len(haralicks["energy_5-0"])
    B_energy_7 = [(i + j + k + v) / 4 for i, j, k, v in zip(haralicks_B["energy_7-0"], haralicks_B["energy_7-45"], haralicks_B["energy_7-90"], haralicks_B["energy_7-135"])]  # ) / len(haralicks["energy_7-0"])
    diff_energy_1 = [(i - j) for i, j in zip(A_energy_1, B_energy_1)]
    diff_energy_3 = [(i - j) for i, j in zip(A_energy_3, B_energy_3)]
    diff_energy_5 = [(i - j) for i, j in zip(A_energy_5, B_energy_5)]
    diff_energy_7 = [(i - j) for i, j in zip(A_energy_7, B_energy_7)]
    mean_diff_energy = (sum(diff_energy_1) / len(diff_energy_1) + sum(diff_energy_3) / len(diff_energy_3) + sum(diff_energy_5) / len(diff_energy_5) + sum(
        diff_energy_7) / len(diff_energy_7)) / 4
    print(mean_diff_energy)
    # energy_avg = (energy_1 + energy_3 + energy_5 + energy_7)/4
    # plt.plot(sorted(energy_1))
    # plt.plot(sorted(energy_3))
    # plt.plot(sorted(energy_5))
    # plt.plot(sorted(energy_7))
    # plt.show()

    # print(energy_avg)
    # values =  [contrast_1, contrast_3, contrast_5, contrast_7, dissimilarity_1, dissimilarity_3, dissimilarity_5, dissimilarity_7,
    #            homogeneity_1, homogeneity_3, homogeneity_5, homogeneity_7, correlation_1, correlation_3, correlation_5, correlation_7,
    #           asm_1, asm_3, asm_5, asm_7, energy_1, energy_3, energy_5, energy_7]
    # values = [contrast_1, dissimilarity_1, homogeneity_1, correlation_1, asm_1, energy_1, contrast_3, dissimilarity_3, homogeneity_3, correlation_3, asm_3, energy_3,
    #           contrast_5, dissimilarity_5, homogeneity_5, correlation_5, asm_5, energy_5, contrast_7, dissimilarity_7, homogeneity_7, correlation_7, asm_7, energy_7]
    # values = [contrast_avg, dissimilarity_avg, homogeneity_avg, correlation_avg, asm_avg, energy_avg]
    # keys = ["contrast d=1", "contrast_3", "contrast_5", "contrast_7", 'dissimilarity_1', 'dissimilarity_3', 'dissimilarity_5', 'dissimilarity_7',
    #         'homogeneity_1', 'homogeneity_3', 'homogeneity_5', 'homogeneity_7', 'correlation_1', 'correlation_3', 'correlation_5', 'correlation_7',
    #         'asm_1', 'asm_3', 'asm_5', 'asm_7', 'energy_1', 'energy_3', 'energy_5', 'energy_7']
    # keys = ["Contrast (d=1)", 'Dissimilarity (d=1)', 'Homogeneity (d=1)', 'Correlation (d=1)', 'ASM (d=1)', 'Energy (d=1)', "Contrast (d=3)", 'Dissimilarity (d=3)',
    #         'Homogeneity (d=3)', 'Correlation (d=3)', 'ASM (d=3)', 'Energy (d=3)', "Contrast (d=5)", 'Dissimilarity (d=5)', 'Homogeneity (d=5)', 'Correlation (d=5)', 'ASM (d=5)', 'Energy (d=5)',
    #         "Contrast (d=7)", 'Dissimilarity (d=7)', 'Homogeneity (d=7)', 'Correlation (d=7)', 'ASM (d=7)', 'Energy (d=7)']
    keys = ["Contrast", "Dissimilarity", "Homogeneity", "Correlation", "ASM", "Energy"]

    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.size'] = 15
    # mpl.rcParams['font.weight'] = 'bold'
    # Create a bar plot (you can choose a different type of plot as needed)
    plt.barh(keys, values)
    # Add labels and a title
    plt.xlabel("Magnitude", fontsize=15, fontweight='bold')
    plt.yticks(fontweight='bold')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()"""
