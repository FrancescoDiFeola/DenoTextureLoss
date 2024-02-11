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


# 1) for each patient in the set compute the average metric score
def calculate_patient_averages(data):
    patient_averages = {}

    for patient_id, metrics in data.items():
        patient_metrics = {}

        for metric, values in metrics.items():
            metric_average = sum(values) / len(values) if values else 0
            patient_metrics[metric] = metric_average

        patient_averages[patient_id] = patient_metrics

    return patient_averages


# 2) compute the average between experiments (before this, compute the per patient average for each experiment)
def calculate_overall_patient_average(patient_averages_list):
    overall_patient_averages = {}

    patient_ids = list(patient_averages_list[0].keys())

    for patient_id in patient_ids:
        patient_metrics = {}

        for metric in patient_averages_list[0][patient_id]:
            metric_sum = sum([averages[patient_id][metric] for averages in patient_averages_list])
            metric_average = metric_sum / len(patient_averages_list)
            patient_metrics[metric] = metric_average

        overall_patient_averages[patient_id] = patient_metrics

    return overall_patient_averages


# 3) compute the overall average among patients
def calculate_average_metrics(data):
    average_metrics = {"psnr": 0, "mse": 0, 'ssim': 0, 'vif': 0, 'paq2piq': 0, 'NIQE': 0, 'PIQE': 0, 'FID_ImNet': 0, 'FID_random': 0, 'brisque': 0}  # 'FID_ImNet': 0, 'FID_random': 0
    metrics_sets = {"psnr": [], "mse": [], 'ssim': [], 'vif': [], 'paq2piq': [], 'NIQE': [], 'PIQE': [], 'FID_ImNet': [], 'FID_random': [], 'brisque': []}

    total_patients = len(data)
    for patient_data in data.values():
        for metric, values in patient_data.items():
            average_metrics[metric] += values
            metrics_sets[metric].append(values)

    for metric in average_metrics:
        average_metrics[metric] /= total_patients

    return average_metrics, metrics_sets


def compute_wilcoxon_test(set_1, set_2, criterion):
    set1 = np.array(set_1)

    set2 = np.array(set_2)

    d = np.squeeze(np.subtract(set1, set2))
    # To test the null hypothesis that there
    # is no value difference between the two sets, we can apply the two-sided test
    # that is we want to verify that the distribution underlying d is not symmetric about zero.
    res = wilcoxon(d, alternative=criterion)
    print(res.statistic, res.pvalue)


if __name__ == "__main__":
    from itertools import product
    experiments_pix2pix = ["metrics_pix2pix_texture_att_diff_unpaired",
                           "metrics_pix2pix_texture_max_diff_unpaired",
                           "metrics_pix2pix_texture_Frob_diff_unpaired",
                           "metrics_pix2pix_texture_avg_diff_unpaired",
                           "metrics_pix2pix_perceptual_unpaired",
                           "metrics_pix2pix_baseline_diff_unpaired",
                           "metric_pix2pix_ssim",
                           "metric_pix2pix_autoencoder",
                           "metric_pix2pix_edge",
                           # "metrics_pix2pix_texture_att_diff_piqe",
                           # "metrics_pix2pix_perceptual_piqe",
                           # "metrics_pix2pix_baseline_diff_piqe",
                           # "metrics_pix2pix_texture_avg_diff_piqe",
                           # "metrics_pix2pix_texture_Frob_diff_piqe",
                           # "metrics_pix2pix_texture_max_diff_piqe",
                   ]
    #comparisons = list(product(experiments_pix2pix, experiments_pix2pix))
    experiments_unit = [
        "metrics_baseline_piqe_niqe",
        "metrics_perceptual_piqe_niqe",
        "metrics_texture_attention_diff_piqe_niqe",
        "metrics_texture_avg_diff_piqe_niqe",
        "metrics_texture_Frob_diff_piqe_niqe",
        "metrics_texture_max_diff_piqe_niqe",
        "metrics_edge",
        "metrics_ssim",
        "metrics_autoencoder",
    ]
    comparisons = list(product(experiments_unit, experiments_unit))
    """  # /Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_texture_att_diff_unpaired_1/metrics_{test_name}_epoch50
    # Per patient average metrics
    for test_name in ['test_2', 'test_3', 'elcap_complete']:  # , 'test_2', 'test_3', 'elcap_complete'
        data = load_from_json(f"/Volumes/sandisk/cycleGAN_emphysema/metrics_baseline_9pat_LUNAhwind_1/metrics_{test_name}_epoch50")
        data1 = load_from_json(f"/Volumes/sandisk/cycleGAN_emphysema/metrics_baseline_9pat_LUNAhwind_1/metrics_{test_name}_epoch50")
        data2 = load_from_json(f"/Volumes/sandisk/cycleGAN_emphysema/metrics_baseline_9pat_LUNAhwind_1/metrics_{test_name}_epoch50")

        # compute the average for each patient in each experiment
        patient_averages_data = calculate_patient_averages(data)
        patient_averages_data1 = calculate_patient_averages(data1)
        patient_averages_data2 = calculate_patient_averages(data2)

        # compute the average over the same patient between experiments
        average_of_averages = calculate_overall_patient_average([patient_averages_data, patient_averages_data1, patient_averages_data2])

        # print("Patient Averages in Data1:", patient_averages_data1)
        # print("Patient Averages in Data2:", patient_averages_data2)
        # print("Overall Averages:", overall_averages)

        # compute the overall average by averaging over the patients
        averaged_metrics, _ = calculate_average_metrics(average_of_averages)
        print(f"Overall average between 3 experiments {test_name}: {averaged_metrics}")"""

    # Wilcoxon test
    for i in comparisons:
        for test_name in ['test_2', 'test_3', 'elcap_complete']:  # , 'test_2', 'test_3', 'elcap_complete'
            data1 = load_from_json(f"/Volumes/Untitled/results_per_patient/UNIT/{i[0]}_1/metrics_{test_name}_epoch50")
            data2 = load_from_json(f"/Volumes/Untitled/results_per_patient/UNIT/{i[0]}_2/metrics_{test_name}_epoch50")
            data3 = load_from_json(f"/Volumes/Untitled/results_per_patient/UNIT/{i[0]}_3/metrics_{test_name}_epoch50")
            data4 = load_from_json(f"/Volumes/Untitled/results_per_patient/UNIT/{i[1]}_1/metrics_{test_name}_epoch50")
            data5 = load_from_json(f"/Volumes/Untitled/results_per_patient/UNIT/{i[1]}_2/metrics_{test_name}_epoch50")
            data6 = load_from_json(f"/Volumes/Untitled/results_per_patient/UNIT/{i[1]}_3/metrics_{test_name}_epoch50")

            # compute the average for each patient in each experiment
            patient_averages_data1 = calculate_patient_averages(data1)
            patient_averages_data2 = calculate_patient_averages(data2)
            patient_averages_data3 = calculate_patient_averages(data3)
            patient_averages_data4 = calculate_patient_averages(data4)
            patient_averages_data5 = calculate_patient_averages(data5)
            patient_averages_data6 = calculate_patient_averages(data6)

            # compute the average over the same patient between experiments
            average_of_averages_1 = calculate_overall_patient_average([patient_averages_data1, patient_averages_data2, patient_averages_data3])
            average_of_averages_2 = calculate_overall_patient_average([patient_averages_data4, patient_averages_data5, patient_averages_data6])

            # compute the overall average by averaging over the patients
            _, metric_sets_1 = calculate_average_metrics(average_of_averages_1)
            _, metric_sets_2 = calculate_average_metrics(average_of_averages_2)
            try:
                print(f"--------------------")
                print(f"TEST {test_name}, ({i[0]}-{i[1]})")
                print(f"--------------------")
                # PSNR
                print("Wilcoxon PSNR:")
                # compute_wilcoxon_test(metric_sets_1['psnr'], metric_sets_2['psnr'], "greater")

                # MSE
                print("Wilcoxon MSE:")
                # compute_wilcoxon_test(metric_sets_1['mse'], metric_sets_2['mse'], "less")
                # SSIM
                print("Wilcoxon SSIM:")
                # compute_wilcoxon_test(metric_sets_1['ssim'], metric_sets_2['ssim'], "greater")
                # VIF
                print("Wilcoxon VIF:")
                # compute_wilcoxon_test(metric_sets_1['vif'], metric_sets_2['vif'], "greater")
                # NIQE
                print("Wilcoxon NIQE:")
                compute_wilcoxon_test(metric_sets_1['NIQE'], metric_sets_2['NIQE'], "less")
                # PIQE
                print("Wilcoxon PIQE:")
                compute_wilcoxon_test(metric_sets_1['PIQE'], metric_sets_2['PIQE'], "less")
            except:
                  print(f"TEST {test_name}, ({i[0]}-{i[1]})")
                  pass

    # ATTENTION MAPS
    # attention1 = np.load("/Volumes/sandisk/results_per_patient/cycleGAN/loss_texture_att_diff0001_4/attention_A.npy")
    # attention2 = np.load("/Volumes/sandisk/results_per_patient/cycleGAN/loss_texture_att_diff0001_5/attention_A.npy")
    # attention3 = np.load("/Volumes/sandisk/results_per_patient/cycleGAN/loss_texture_att_diff0001_6/attention_A.npy")
    # attention4 = np.load("/Volumes/sandisk/results_per_patient/cycleGAN/loss_texture_att_diff0001_4/attention_B.npy")
    # attention5 = np.load("/Volumes/sandisk/results_per_patient/cycleGAN/loss_texture_att_diff0001_5/attention_B.npy")
    # attention6 = np.load("/Volumes/sandisk/results_per_patient/cycleGAN/loss_texture_att_diff0001_6/attention_B.npy")
    # stacked_experiments = np.concatenate([attention1, attention2, attention3, attention4, attention5, attention6], axis=0)
    # attention1 = np.load("/Volumes/sandisk/results_per_patient/pix2pix/loss_pix2pix_texture_att_diff_4/attention_B.npy")
    # attention2 = np.load("/Volumes/sandisk/results_per_patient/pix2pix/loss_pix2pix_texture_att_diff_5/attention_B.npy")
    # attention3 = np.load("/Volumes/sandisk/results_per_patient/pix2pix/loss_pix2pix_texture_att_diff_6/attention_B.npy")
    """attention = load_from_json("/Volumes/Untitled/results_per_patient/cycleGAN/metrics_texture_att_diff_7/attention_maps_B_ep50")
    texture_grids = load_from_json("/Volumes/Untitled/results_per_patient/cycleGAN/metrics_texture_att_diff_7/delta_grids_B_test_2_ep50")
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.size'] = 15
    for key in attention.keys():
        total = len(attention[key])
        middle = round(len(attention[key])/2)
        upper = round((len(attention[key])/3)/2)
        lower = round(((len(attention[key])/3)*2)+(len(attention[key])/3)/2)
        print(f"{key}, {len(attention[key])}, middle:{middle}, upper:{upper}, lower:{lower}")
        zone = [lower, middle, upper]
        for z in zone:

            averaged_attention_map = np.array(attention[key][z])[0, :, :]

            grid = np.array(texture_grids["C004"][z])[0, 0, :, :]

            fig, ax = plt.subplots()
            row_labels = [1, 3, 5, 7]
            col_labels = ["0°", "45°", "90°", "135°"]
            ax.matshow(grid, cmap='gray', vmin=np.min(grid), vmax=np.max(grid))
            ax.set_xticklabels(col_labels, fontsize=30)
            ax.set_yticklabels(row_labels, fontsize=30)
            ax.set_xticks(np.arange(len(col_labels)))
            ax.set_yticks(np.arange(len(row_labels)))
            plt.title(f"$\Delta\phi$ (slice n: {z}/{total})", fontsize=30)
            plt.show()

            # print(attention1.shape, attention2.shape, attention3.shape)
            # print(np.expand_dims(attention1[0, :, :, :], axis=0).shape)
            # stacked_experiments = np.concatenate([attention4[16536:, :, :, :], attention5[16536:, :, :, :], attention6[16536:, :, :, :]], axis=0)
            # stacked_experiments = np.concatenate([attention1, attention2, attention3], axis=0)
            # print(stacked_experiments.shape)
            # averaged_attention_map = np.mean(stacked_experiments, axis=(0, 1))
            # averaged_attention_map = cv2.resize(averaged_attention_map, (4, 4))
            # Create a figure and axis
            fig, ax = plt.subplots()

            # Plot the grid with black shades
            print(f"Min: {np.min(averaged_attention_map)}")
            print(f"Max: {np.max(averaged_attention_map)}")
            print(f"Row sum: {np.sum(averaged_attention_map, axis=1)}")
            print(f"Column sum: {np.sum(averaged_attention_map, axis=0)}")

            cax = ax.matshow(averaged_attention_map, cmap='gray', vmin=np.min(averaged_attention_map), vmax=np.max(averaged_attention_map))

            spatial_offsets = [1, 1, 1, 1, 3, 3, 3, 3, 5, 5, 5, 5, 7, 7, 7, 7]
            angular_offsets = [0, 45, 90, 135, 0, 45, 90, 135, 0, 45, 90, 135, 0, 45, 90, 135]
            # row_labels = [1, 3, 5, 7]
            # col_labels = ["0°", "45°", "90°", "135°"]

            # Add labels for rows and columns
            row_labels = [f'{i}-{j}°' for i, j in zip(spatial_offsets, angular_offsets)]
            col_labels = [f'{i}-{j}°' for i, j in zip(spatial_offsets, angular_offsets)]

            ax.set_xticks(np.arange(len(col_labels)))
            ax.set_yticks(np.arange(len(row_labels)))

            ax.set_xticklabels(col_labels, fontsize=12)
            ax.set_yticklabels(row_labels, fontsize=12)

            # Rotate the x-axis labels for better visibility
            plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")  # , rotation=45
            plt.title("cycleGAN attention map", fontsize=18)
            # Display the grid
            plt.show()"""

    # ATTENTION WEIGHT
    """w1 = np.load("/Volumes/sandisk/pix2pix_results/loss_pix2pix_texture_att1/weight.npy")
    w2 = np.load("/Volumes/sandisk/pix2pix_results/loss_pix2pix_texture_att2/weight.npy")
    w3 = np.load("/Volumes/sandisk/pix2pix_results/loss_pix2pix_texture_att3/weight.npy")
    print(len(w1), len(w2), len(w3))
    avg_weight_pix2pix = [(i+j+t)/3 for i, j, t in zip(w1, w2, w3)]
    w4 = np.load("/Volumes/sandisk/cycleGAN_results/loss_texture_att13/weight_A.npy")
    w5 = np.load("/Volumes/sandisk/cycleGAN_results/loss_texture_att14/weight_A.npy")
    w6 = np.load("/Volumes/sandisk/cycleGAN_results/loss_texture_att15/weight_A.npy")
    w7 = np.load("/Volumes/sandisk/cycleGAN_results/loss_texture_att13/weight_B.npy")
    w8 = np.load("/Volumes/sandisk/cycleGAN_results/loss_texture_att14/weight_B.npy")
    w9 = np.load("/Volumes/sandisk/cycleGAN_results/loss_texture_att15/weight_B.npy")
    print(len(w4), len(w5), len(w6), len(w7), len(w8), len(w9))
    avg_weight_cycle = [(a+b+c+d+e+f)/6 for a, b, c, d, e, f in zip(w4, w5, w6, w7, w8, w9)]
    w10 = np.load("/Volumes/sandisk/UNIT/loss_texture_att_early_1/weight_X1.npy")
    w11 = np.load("/Volumes/sandisk/UNIT/loss_texture_att_early_2/weight_X1.npy")
    w12 = np.load("/Volumes/sandisk/UNIT/loss_texture_att_early_3/weight_X1.npy")
    w13 = np.load("/Volumes/sandisk/UNIT/loss_texture_att_early_1/weight_X2.npy")
    w14 = np.load("/Volumes/sandisk/UNIT/loss_texture_att_early_2/weight_X2.npy")
    w15 = np.load("/Volumes/sandisk/UNIT/loss_texture_att_early_3/weight_X2.npy")
    print(len(w10), len(w10), len(w10), len(w10), len(w10), len(w10))
    avg_weight_unit = [(t1+t2+t3+t4+t5+t6)/6 for t1, t2, t3, t4, t5, t6 in zip(w10, w11, w12, w13, w14, w15)]
    plt.plot(avg_weight_cycle)
    plt.plot(avg_weight_pix2pix)
    plt.plot(avg_weight_unit)
    plt.legend(["cycleGAN", "Pix2Pix", "UNIT"])
    plt.title("Average attention loss weight")
    plt.grid()
    plt.show()"""

    # PERCEPTION-DISTORTION PLOT
    """plt.figure()
    # plt.scatter([31.9714, 31.8105, 31.8427, 32.1373, 31.8319], [0.9626, 0.9594, 0.9580, 0.9614, 0.9588], marker='o')
    mse_pix2pix = [0.02252, 0.01050, 0.01050, 0.01032, 0.01042, 0.01162, 0.01041]
    # Find the minimum and maximum values in the list
    # min_mse_value = min(mse)
    # max_mse_value = max(mse)
    # Normalize the list
    # mse = [(x - min_mse_value) / (max_mse_value - min_mse_value) for x in mse]

    niqe_pix2pix = [18.24939, 6.69768, 6.72725, 6.34454, 6.21355, 5.33557, 6.74511]
    fid_imnet_pix2pix = [80.31756, 45.27445, 44.75019, 47.12884, 45.64557, 69.52257, 44.42599]
    fid_random_pix2pix = [115.85072, 61.73528, 61.32585, 63.72904, 62.34899, 90.06185,  60.32384]
    # min_niqe_value = min(niqe)
    # max_niqe_value = max(niqe)
    # Normalize the list
    # niqe = [(x - min_niqe_value) / (max_niqe_value - min_niqe_value) for x in niqe]
    mse_cycleGAN = [0.02252, 0.01015, 0.00970, 0.00932, 0.00954, 0.00989, 0.01017]
    niqe_cycleGAN = [18.24939, 10.43972, 10.05958, 9.28210, 9.09292, 5.73334, 10.90776]
    fid_imnet_cycleGAN = [80.31756, 69.39279, 69.17404, 67.30076, 65.84454, 69.50455, 64.00218]
    fid_random_cycleGAN = [115.85072, 92.21178, 92.28362, 87.92113, 86.52458, 87.83419,  84.71191]

    mse_unit = [0.02252, 0.00885, 0.00732, 0.00770, 0.00741, 0.00739, 0.00854]
    niqe_unit = [18.24939, 6.63212, 7.05992, 6.97737, 7.47793, 5.98374, 6.58249]
    fid_imnet_unit = [80.31756, 66.08463, 54.15049, 57.00600, 56.91359, 55.71386, 57.22355]
    fid_random_unit = [115.85072, 84.50406,  66.93555, 70.92612, 71.06563, 70.40341, 71.40135]

    plt.scatter(fid_imnet_pix2pix[1:], niqe_pix2pix[1:], marker='o', color='orange')
    plt.scatter(fid_imnet_cycleGAN[1:], niqe_cycleGAN[1:], marker='o', color='red')
    plt.scatter(fid_imnet_unit[1:], niqe_unit[1:], marker='o', color='blue')
    # plt.annotate('LD', (mse_pix2pix[0], niqe_pix2pix[0]))
    plt.annotate('L_b', (fid_imnet_pix2pix[1], niqe_pix2pix[1]))
    plt.annotate('L_max', (fid_imnet_pix2pix[2], niqe_pix2pix[2]))
    plt.annotate('L_avg', (fid_imnet_pix2pix[3], niqe_pix2pix[3]))
    plt.annotate('L_Frob', (fid_imnet_pix2pix[4], niqe_pix2pix[4]))
    plt.annotate('L_att', (fid_imnet_pix2pix[5], niqe_pix2pix[5]))
    plt.annotate('L_vgg', (fid_imnet_pix2pix[6], niqe_pix2pix[6]))
    plt.annotate('L_b', (fid_imnet_cycleGAN[1], niqe_cycleGAN[1]))
    plt.annotate('L_max', (fid_imnet_cycleGAN[2], niqe_cycleGAN[2]))
    plt.annotate('L_avg', (fid_imnet_cycleGAN[3], niqe_cycleGAN[3]))
    plt.annotate('L_Frob', (fid_imnet_cycleGAN[4], niqe_cycleGAN[4]))
    plt.annotate('L_att', (fid_imnet_cycleGAN[5], niqe_cycleGAN[5]))
    plt.annotate('L_vgg', (fid_imnet_cycleGAN[6], niqe_cycleGAN[6]))
    plt.annotate('L_b', (fid_imnet_unit[1], niqe_unit[1]))
    plt.annotate('L_max', (fid_imnet_unit[2], niqe_unit[2]))
    plt.annotate('L_avg', (fid_imnet_unit[3], niqe_unit[3]))
    plt.annotate('L_Frob', (fid_imnet_unit[4], niqe_unit[4]))
    plt.annotate('L_att', (fid_imnet_unit[5], niqe_unit[5]))
    plt.annotate('L_vgg', (fid_imnet_unit[6], niqe_unit[6]))
    #plt.ylim([0.925, 1.02])
    plt.legend(['Pix2Pix', 'CycleGAN', 'UNIT'])
    plt.ylabel("NIQE")
    plt.xlabel("FID ImNet")
    plt.grid()
    plt.title('Perception-distorsion evaluation', fontsize=12)
    plt.show()"""
