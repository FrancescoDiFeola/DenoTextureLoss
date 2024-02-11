import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from scipy.stats import wilcoxon
import os
import pylab as py
import json
from data.storage import load_from_json
from scipy.stats import wilcoxon
from data.storage import *
import copy
import cv2
from data.storage import load_from_json

# ATTENTION MAPS
# attention1 = np.load("/Volumes/sandisk/results_per_patient/cycleGAN/loss_texture_att_diff0001_4/attention_A.npy")
# attention2 = np.load("/Volumes/sandisk/results_per_patient/cycleGAN/loss_texture_att_diff0001_5/attention_A.npy")
# attention3 = np.load("/Volumes/sandisk/results_per_patient/cycleGAN/loss_texture_att_diff0001_6/attention_A.npy")
# attention4 = np.load("/Volumes/sandisk/results_per_patient/cycleGAN/loss_texture_att_diff0001_4/attention_B.npy")
# attention5 = np.load("/Volumes/sandisk/results_per_patient/cycleGAN/loss_texture_att_diff0001_5/attention_B.npy")
# attention6 = np.load("/Volumes/sandisk/results_per_patient/cycleGAN/loss_texture_att_diff0001_6/attention_B.npy")
# stacked_experiments = np.concatenate([attention1, attention2, attention3, attention4, attention5, attention6], axis=0)
attention1 = load_from_json("/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_texture_att_diff_7/attention_maps_ep50")
attention2 = load_from_json("/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_texture_att_diff_8/attention_maps_ep50")
attention3 = load_from_json("/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_texture_att_diff_9/attention_maps_ep50")

# print(np.expand_dims(attention1[0, :, :, :], axis=0).shape)
print(attention1)

rows, cols = 5, 10
num_heatmaps = rows * cols
fig, axs = plt.subplots(rows, cols, figsize=(12, 6))
for i in range(0, 50):
    stacked_experiments = np.concatenate([attention1[(i * 336):(336 * (i+1)), :, :, :], attention2[(i * 336):(336 * (i+1)), :, :, :], attention3[(i * 336):(336 * (i+1)), :, :, :]], axis=0)
    # stacked_experiments = np.concatenate([attention1, attention2, attention3], axis=0)
    print(stacked_experiments.shape)
    averaged_attention_map = np.mean(stacked_experiments, axis=(0, 1))
    # averaged_attention_map = cv2.resize(averaged_attention_map, (4, 4))


    # Plot the grid with black shades
    print(np.min(averaged_attention_map))
    print(np.max(averaged_attention_map))
    print(np.sum(averaged_attention_map, axis=1))
    print(np.sum(averaged_attention_map, axis=0))

    row, col = divmod(i, cols)  # Calculate the row and column for this heatmap

    # Clear the previous heatmap (if any) on the current subplot
    axs[row, col].cla()

    # Add the heatmap data to the current subplot
    heatmap = axs[row, col].imshow(averaged_attention_map, cmap='gray')
    axs[row, col].axis('off')
    # cax = ax.matshow(averaged_attention_map, cmap='gray')

    spatial_offsets = [1, 1, 1, 1, 3, 3, 3, 3, 5, 5, 5, 5, 7, 7, 7, 7]
    angular_offsets = [0, 45, 90, 135, 0, 45, 90, 135, 0, 45, 90, 135, 0, 45, 90, 135]
    # row_labels = [1, 3, 5, 7]
    # col_labels = ["0°", "45°", "90°", "135°"]

    # Add labels for rows and columns
    row_labels = [f'{i}-{j}°' for i, j in zip(spatial_offsets, angular_offsets)]
    col_labels = [f'{i}-{j}°' for i, j in zip(spatial_offsets, angular_offsets)]

    # axs.set_xticks(np.arange(len(col_labels)))
    # axs.set_yticks(np.arange(len(row_labels)))

    # axs.set_xticklabels(col_labels, fontsize=12)
    # axs.set_yticklabels(row_labels, fontsize=12)

    # Rotate the x-axis labels for better visibility
    # plt.setp(axs.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")  # , rotation=45

# Display the grid
plt.show()
