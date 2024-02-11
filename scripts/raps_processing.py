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


def element_wise_average(lists):
    avg_sublist = [sum(values) / len(values) for values in zip(*lists)]

    return avg_sublist


def element_wise_average2(lists):
    list_lengths = [len(sublist) for sublist in lists[0]]

    if any(length != list_lengths[0] for length in list_lengths):
        raise ValueError("All lists must have the same length")

    result = []
    for sublists in zip(*lists):
        if any(len(sublist) != len(sublists[0]) for sublist in sublists):
            raise ValueError("Sublists must have the same length")

        avg_sublist = [sum(values) / len(values) for values in zip(*sublists)]
        result.append(avg_sublist)

    return [np.mean(k) for k in zip(*result)]


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    """categories = ["Precision", "Recall", "Specificity", "F-score", "Accuracy"]
    m1_m2 = [7.9, 3.8, 2.6, 4.9, 1.5]  # Replace with your actual data
    values_m2 = [0.6524, 0.6610, 0.8649, 0.6473, 0.7993]
    m3_m4 = [3.7, 6.1, 0.9, 5.2, 2.0]
    values_m4 = [0.6622, 0.6848, 0.8690, 0.6663, 0.8171]

    # Create the bar plot
    width = 0.4  # Width of each bar
    x = np.arange(len(categories))

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width / 2, values_m2, width, label='patch size 128x128', color='skyblue')
    bars2 = plt.bar(x + width / 2, values_m4, width, label='patch size 144x144', color='lightcoral')

    # Add percentages over each set of bars
    for bar, percentage1, percentage2 in zip(bars1, m1_m2, m3_m4):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'+{percentage1:.1f}%',
            ha='center',
            fontsize=15,
            fontweight='bold',
        )

    for bar, percentage1, percentage2 in zip(bars2, m1_m2, m3_m4):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'+{percentage2:.1f}%',
            ha='center',
            fontsize=15,
            fontweight='bold',
        )

    # Add labels and title
    plt.title('Emphysema classification: increase in performance due to denoising', fontsize=18)
    plt.xticks(x, categories, fontsize=18)
    plt.legend(fontsize=18)
    plt.yticks(fontsize=18)
    # Show the plot
    plt.tight_layout()
    plt.show()"""

    # RAPS (Radially Averaged Power Spectrum) overall per patient
    """for test_name in ['test_3', "elcap_complete"]:  # 'test_3', 'elcap_complete'
        data = load_from_json(f"/Volumes/sandisk/results_per_patient/UNIT/metrics_perceptual_diff_1/raps_{test_name}_epoch50")
        data1 = load_from_json(f"/Volumes/sandisk/results_per_patient/UNIT/metrics_perceptual_diff_2/raps_{test_name}_epoch50")
        data2 = load_from_json(f"/Volumes/sandisk/results_per_patient/UNIT/metrics_perceptual_diff_3/raps_{test_name}_epoch50")

        data_new = copy.deepcopy(data)
        empty_dictionary(data_new, nesting=1)

        data_new_1 = copy.deepcopy(data1)
        empty_dictionary(data_new_1, nesting=1)

        data_new_2 = copy.deepcopy(data2)
        empty_dictionary(data_new_2, nesting=1)

        for patient in data.keys():
            if patient != "W0048":
                data_new[patient] = element_wise_average(data[patient]['raps'])

        for patient1 in data.keys():
            if patient1 != "W0048":
                data_new_1[patient1] = element_wise_average(data1[patient1]['raps'])

        for patient2 in data.keys():
            if patient2 != "W0048":
                data_new_2[patient2] = element_wise_average(data2[patient2]['raps'])

        data_new_3 = copy.deepcopy(data)
        empty_dictionary(data_new_3, nesting=1)

        for p in data.keys():
            if p != "W0048":
                data_new_3[p] = element_wise_average([data_new[p], data_new_1[p], data_new_2[p]])

        raps_overall = []
        for t in data.keys():
            if t != "W0048":
                raps_overall.append(data_new_3[t])

        raps_average = element_wise_average(raps_overall)

        save_to_json(raps_average, f"/Volumes/sandisk/results_per_patient/UNIT/metrics_perceptual_diff_1/raps_overall_{test_name}_ep50")"""

    profile_elcap_baseline = load_from_json("/Volumes/sandisk/results_per_patient/cycleGAN/metrics_baseline_diff01_1/raps_overall_elcap_complete_ep50")
    profile_elcap_texture_max = load_from_json("/Volumes/sandisk/results_per_patient/cycleGAN/metrics_texture_max_diff0001_1/raps_overall_elcap_complete_ep50")
    profile_elcap_texture_avg = load_from_json("/Volumes/sandisk/results_per_patient/cycleGAN/metrics_texture_avg_diff0001_1/raps_overall_elcap_complete_ep50")
    profile_elcap_texture_Frob = load_from_json("/Volumes/sandisk/results_per_patient/cycleGAN/metrics_texture_Frob_diff0001_1/raps_overall_elcap_complete_ep50")
    profile_elcap_texture_att = load_from_json("/Volumes/sandisk/results_per_patient/cycleGAN/metrics_texture_att_diff0001_1/raps_overall_elcap_complete_ep50")
    profile_elcap_perceptual = load_from_json("/Volumes/sandisk/results_per_patient/cycleGAN/metrics_perceptual_diff_4/raps_overall_elcap_complete_ep50")

    profile_test_3_baseline = load_from_json("/Volumes/sandisk/results_per_patient/cycleGAN/metrics_baseline_diff01_1/raps_overall_test_3_ep50")
    profile_test_3_texture_max = load_from_json("/Volumes/sandisk/results_per_patient/cycleGAN/metrics_texture_max_diff0001_1/raps_overall_test_3_ep50")
    profile_test_3_texture_avg = load_from_json("/Volumes/sandisk/results_per_patient/cycleGAN/metrics_texture_avg_diff0001_1/raps_overall_test_3_ep50")
    profile_test_3_texture_Frob = load_from_json("/Volumes/sandisk/results_per_patient/cycleGAN/metrics_texture_Frob_diff0001_1/raps_overall_test_3_ep50")
    profile_test_3_texture_att = load_from_json("/Volumes/sandisk/results_per_patient/cycleGAN/metrics_texture_att_diff0001_1/raps_overall_test_3_ep50")
    profile_test_3_perceptual = load_from_json("/Volumes/sandisk/results_per_patient/cycleGAN/metrics_perceptual_diff_4/raps_overall_test_3_ep50")

    # ELCAP low-dose profile
    raps_ld = open(f'/Volumes/sandisk/metrics_low_dose/raps_ELCAP_ld.json')
    raps_ld = json.load(raps_ld)
    profile_elcap_low_dose = element_wise_average2([raps_ld])

    # Test 3 (LIDC/IDRI) low-dose profile
    raps_ld = open(f'/Volumes/sandisk/metrics_low_dose/raps_test_3_ld.json')
    raps_ld = json.load(raps_ld)
    profile_test_3_low_dose = element_wise_average2([raps_ld])

    # ELCAP test
    py.semilogy(profile_elcap_baseline)
    py.semilogy(profile_elcap_texture_max)
    py.semilogy(profile_elcap_texture_avg)
    py.semilogy(profile_elcap_texture_Frob)
    py.semilogy(profile_elcap_texture_att)
    py.semilogy(profile_elcap_perceptual)
    py.semilogy(profile_elcap_low_dose)
    py.legend(["baseline", "texture_max", "texture_avg",  "texture_Frob", "texture_att", "perceptual_loss", 'Low-Dose reference'])
    # py.xticks(fontsize=18)
    # py.yticks(fontsize=18)
    py.grid()
    py.title("Real test set 2 at epoch 50 (UNIT)")
    py.show()

    # Test 3
    py.semilogy(profile_test_3_baseline)
    py.semilogy(profile_test_3_texture_max)
    py.semilogy(profile_test_3_texture_avg)
    py.semilogy(profile_test_3_texture_Frob)
    py.semilogy(profile_test_3_texture_att)
    py.semilogy(profile_test_3_perceptual)
    py.semilogy(profile_test_3_low_dose)
    py.legend(["baseline_loss", "texture_loss_1", "texture_loss_2", "texture_loss_3", "texture_loss_4", "perceptual_loss", "Low-Dose reference"], fontsize=18)
    py.xticks(fontsize=18)
    py.yticks(fontsize=18)
    py.xlabel("Radius",fontsize=18 )
    py.grid()
    py.title("Radially averaged power spectrum: before and after denoising \n using different model configurations", fontsize=18)
    py.show()
