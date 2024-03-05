from matplotlib import rc
import seaborn as sns
import matplotlib.pyplot as plt
import pylab as py
from data.storage import *
import numpy as np
import copy
from data.storage import load_from_json

#########################################################################
# Latex preamble
# For aspect ratio 4:3.
column_width_pt = 516.0
pt_to_inch = 1 / 72.27
column_width_inches = column_width_pt * pt_to_inch
aspect_ratio = 4 / 3
sns.set(style="white", font_scale=1.8, rc={"figure.figsize": (column_width_inches, column_width_inches / aspect_ratio)})

# For Latex.
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
#########################################################################
# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #
# img = np.load("/Users/francescodifeola/Desktop/10011_clean_deno.npy")
# print(img.shape)
# plt.imshow(img[0, 200, :, :], cmap='gray')
# plt.show()
# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #
# PERCEPTION-DISTORTION PLOT

plt.figure(figsize=(column_width_inches, column_width_inches / aspect_ratio))

plt.figure(figsize=(column_width_inches, column_width_inches / aspect_ratio))

# plt.scatter([31.9714, 31.8105, 31.8427, 32.1373, 31.8319], [0.9626, 0.9594, 0.9580, 0.9614, 0.9588], marker='o')

###### mse_pix2pix = [0.02252, 0.01050, 0.01050, 0.01032, 0.01042, 0.01162, 0.01041, 0.01055, 0.01037, 0.01063]
mse_pix2pix = [2.25, 10.50, 10.50, 10.32, 10.42, 11.62, 10.41, 10.55, 10.37, 10.63]
# Find the minimum and maximum values in the list
# min_mse_value = min(mse)
# max_mse_value = max(mse)
# Normalize the list
# mse = [(x - min_mse_value) / (max_mse_value - min_mse_value) for x in mse]

#########niqe_pix2pix = [17.02715, 6.52625, 6.50243, 6.23657, 6.06699, 5.19338, 6.56729, 6.16399, 6.06902, 6.44128]
niqe_pix2pix = [17.0272, 6.5263, 6.5024, 6.2366, 6.0670, 5.1934, 6.5673, 6.1640, 6.0690, 6.4413]
#########piqe_pix2pix = [23.54954, 8.60171, 8.79549, 8.54729, 8.80961, 7.35751, 8.71142, 8.91563, 8.29945, 8.74611]
piqe_pix2pix = [23.5495, 8.6017, 8.7950, 8.5473, 8.8096, 7.3575, 8.7114, 8.9156, 8.2995, 8.7461]
# fid_imnet_pix2pix = [80.31756, 45.27445, 44.75019, 47.12884, 45.64557, 69.52257, 44.42599]
# fid_random_pix2pix = [115.85072, 61.73528, 61.32585, 63.72904, 62.34899, 90.06185,  60.32384]
# min_niqe_value = min(niqe)
# max_niqe_value = max(niqe)
# Normalize the list
# niqe = [(x - min_niqe_value) / (max_niqe_value - min_niqe_value) for x in niqe]
###########mse_cycleGAN = [0.02252, 0.01015, 0.00970, 0.00932, 0.00954, 0.00989, 0.01017, 0.00969, 0.01050, 0.01084]
mse_cycleGAN = [2.25, 10.15, 9.70, 9.32, 9.54, 9.89, 10.17, 9.69, 10.50, 10.84]
###########niqe_cycleGAN = [17.02715, 9.62032, 9.28434, 8.66423, 8.51001, 5.32091, 10.17277, 8.03170, 10.27325, 10.70923]
niqe_cycleGAN = [17.0272, 9.6203, 9.2843, 8.6642, 8.5100, 5.3209, 10.1728, 8.0317, 10.2733, 10.7092]
############piqe_cycleGAN = [23.54948, 12.68929, 11.93226, 10.97366, 11.12554, 5.44317, 13.32563, 8.71662, 13.01870, 13.58824]
piqe_cycleGAN = [23.5495, 12.6893, 11.9323, 10.9737, 11.1255, 5.4432, 13.3256, 8.7166, 13.0187, 13.5882]
# fid_imnet_cycleGAN = [80.31756, 69.39279, 69.17404, 67.30076, 65.84454, 69.50455, 64.00218]
# fid_random_cycleGAN = [115.85072, 92.21178, 92.28362, 87.92113, 86.52458, 87.83419,  84.71191]

############mse_unit = [0.02252, 0.00885, 0.00732, 0.00770, 0.00741, 0.00739, 0.00854, 0.00768, 0.00764, 0.00783]
mse_unit = [2.25, 8.85, 7.32, 7.70, 7.41, 7.39, 8.54, 7.68, 7.64, 7.83]
############niqe_unit = [17.02715, 6.45745, 6.84344, 6.75063, 7.16556, 6.02491, 6.50301, 7.25239, 7.41838, 6.68971]
niqe_unit = [17.0272, 6.4581, 6.8441, 6.7520, 7.1659, 6.0238, 6.5030, 7.2524, 7.4184, 6.6897]
##########piqe_unit = [23.54948, 8.31790, 7.5657, 7.88673, 7.78715, 7.41100, 7.79082, 7.63876, 7.56228, 7.22717]
piqe_unit = [23.5495, 8.3179, 7.5657, 7.8877, 7.7813, 7.4110, 7.7908, 7.6388, 7.5623, 7.2272]
# fid_imnet_unit = [80.31756, 66.08463, 54.15049, 57.00600, 56.91359, 55.71386, 57.22355]
# fid_random_unit = [115.85072, 84.50406,  66.93555, 70.92612, 71.06563, 70.40341, 71.40135]

plt.scatter(mse_pix2pix[1:], piqe_pix2pix[1:], marker='o', color='orange')
plt.scatter(mse_cycleGAN[1:], piqe_cycleGAN[1:], marker='o', color='red')
plt.scatter(mse_unit[1:], piqe_unit[1:], marker='o', color='blue')

plt.scatter(mse_unit[1:], piqe_unit[1:], marker='o', color='blue')

# plt.annotate('LD', (mse_pix2pix[0], niqe_pix2pix[0]))
"""plt.annotate('Baseline', (mse_pix2pix[1], niqe_pix2pix[1]+0.3), fontsize=12)
plt.annotate('MSTLF-max', (mse_pix2pix[2]+0.00008, niqe_pix2pix[2]+0.11), fontsize=12)
plt.annotate('MSTLF-average', (mse_pix2pix[3]-0.5, niqe_pix2pix[3]+0.0), fontsize=12, ha="center")
plt.annotate('MSTLF-Frobenius', (mse_pix2pix[4]+0.04, niqe_pix2pix[4]-0.25), fontsize=12)
plt.annotate('MSTLF-attention', (mse_pix2pix[5]+0.04, niqe_pix2pix[5]+0.1), fontsize=12, ha="right")
plt.annotate('VGG-16', (mse_pix2pix[6], niqe_pix2pix[6]+0.1), fontsize=12, ha='right')
plt.annotate('AE-CT', (mse_pix2pix[7]+0.48, niqe_pix2pix[7]-0.05), fontsize=12, ha='right')
plt.annotate('SSIM-L', (mse_pix2pix[8]-0.03, niqe_pix2pix[8]-0.2), fontsize=12, ha='right')
plt.annotate('EDGE', (mse_pix2pix[9]+0.44, niqe_pix2pix[9]-0.06), fontsize=12, ha='right')
plt.annotate('Baseline', (mse_cycleGAN[1], niqe_cycleGAN[1]+0.1), fontsize=12, ha="center")
plt.annotate('MSTLF-max', (mse_cycleGAN[2], niqe_cycleGAN[2]+0.1), fontsize=12, ha="center")
plt.annotate('MSTLF-average', (mse_cycleGAN[3]+0.00016, niqe_cycleGAN[3]+0.1), fontsize=12, ha="center")
plt.annotate('MSTLF-Frobenius', (mse_cycleGAN[4]+0.05, niqe_cycleGAN[4]), fontsize=12)
plt.annotate('MSTLF-attention', (mse_cycleGAN[5], niqe_cycleGAN[5]+0.1), fontsize=12, ha="center")
plt.annotate('VGG-16', (mse_cycleGAN[6]-0.0002, niqe_cycleGAN[6]+0.08), fontsize=12, ha="center")
plt.annotate('AE-CT', (mse_cycleGAN[7], niqe_cycleGAN[7]+0.08), fontsize=12, ha="center")
plt.annotate('SSIM-L', (mse_cycleGAN[8]+0.2, niqe_cycleGAN[8]+0.09), fontsize=12, ha="center")
plt.annotate('EDGE', (mse_cycleGAN[9], niqe_cycleGAN[9]+0.08), fontsize=12, ha="center")
plt.annotate('Baseline', (mse_unit[1]-0.2, niqe_unit[1]-0.3), fontsize=12, ha="left")
plt.annotate('MSTLF-max', (mse_unit[2]-0.2, niqe_unit[2]+0.1), fontsize=12)
plt.annotate('MSTLF-average', (mse_unit[3], niqe_unit[3]+0.1), fontsize=12)
plt.annotate('MSTLF-Frobenius', (mse_unit[4]+0.32, niqe_unit[4]+0.02), fontsize=12)
plt.annotate('MSTLF-attention', (mse_unit[5]-0.0007, niqe_unit[5]+0.1), fontsize=12)
plt.annotate('VGG-16', (mse_unit[6], niqe_unit[6]+0.1), fontsize=12, ha="center")
plt.annotate('AE-CT', (mse_unit[7]-0.3, niqe_unit[7]), fontsize=12, ha="center")
plt.annotate('SSIM-L', (mse_unit[8], niqe_unit[8]+0.1), fontsize=12, ha="center")
plt.annotate('EDGE', (mse_unit[9]-0.00006, niqe_unit[9]-0.28), fontsize=12, ha="center")"""
##########
plt.annotate('Baseline', (mse_pix2pix[1]+0.04, piqe_pix2pix[1]-0.32), fontsize=12)
plt.annotate('MSTLF-max', (mse_pix2pix[2]+0.1, piqe_pix2pix[2]+0.1), fontsize=12)
plt.annotate('MSTLF-average', (mse_pix2pix[3]-0.45, piqe_pix2pix[3]-0.28), fontsize=12, ha="center")
plt.annotate('MSTLF-Frobenius', (mse_pix2pix[4]-1, piqe_pix2pix[4]+0.28), fontsize=12)
plt.annotate('MSTLF-attention', (mse_pix2pix[5], piqe_pix2pix[5]+0.15), fontsize=12, ha="right")
plt.annotate('VGG-16', (mse_pix2pix[6]-0.05, piqe_pix2pix[6]), fontsize=12, ha='right')
plt.annotate('AE-CT', (mse_pix2pix[7]+0.4, piqe_pix2pix[7]+0.28), fontsize=12, ha='right')
plt.annotate('SSIM-L', (mse_pix2pix[8]+0.0002, piqe_pix2pix[8]-0.4), fontsize=12, ha='right')
plt.annotate('EDGE', (mse_pix2pix[9]+0.45, piqe_pix2pix[9]-0.18), fontsize=12, ha='right')
plt.annotate('Baseline', (mse_cycleGAN[1]-0.00008, piqe_cycleGAN[1]+0.15), fontsize=12, ha="center")
plt.annotate('MSTLF-max', (mse_cycleGAN[2], piqe_cycleGAN[2]+0.15), fontsize=12, ha="center")
plt.annotate('MSTLF-average', (mse_cycleGAN[3]+0.00016, piqe_cycleGAN[3]-0.4), fontsize=12, ha="center")
plt.annotate('MSTLF-Frobenius', (mse_cycleGAN[4]+0.08, piqe_cycleGAN[4]), fontsize=12)
plt.annotate('MSTLF-attention', (mse_cycleGAN[5], piqe_cycleGAN[5]+0.2), fontsize=12, ha="center")
plt.annotate('VGG-16', (mse_cycleGAN[6], piqe_cycleGAN[6]+0.15), fontsize=12, ha="center")
plt.annotate('AE-CT', (mse_cycleGAN[7], piqe_cycleGAN[7]+0.05), fontsize=12, ha='right')
plt.annotate('SSIM-L', (mse_cycleGAN[8]+0.5, piqe_cycleGAN[8]-0.2), fontsize=12, ha='right')
plt.annotate('EDGE', (mse_cycleGAN[9]+0.45, piqe_cycleGAN[9]-0.06), fontsize=12, ha='right')
plt.annotate('Baseline', (mse_unit[1]-0.2, piqe_unit[1]+0.15), fontsize=12, ha="left")
plt.annotate('MSTLF-max', (mse_unit[2]-0.2, piqe_unit[2]-0.6), fontsize=12)
plt.annotate('MSTLF-average', (mse_unit[3]-0.0005, piqe_unit[3]+0.05), fontsize=12)
plt.annotate('MSTLF-Frobenius', (mse_unit[4]-0.2, piqe_unit[4]+0.6), fontsize=12)
plt.annotate('MSTLF-attention', (mse_unit[5]-0.2, piqe_unit[5]-0.8), fontsize=12)
plt.annotate('VGG-16', (mse_unit[6]+0.2, piqe_unit[6]-0.4), fontsize=12, ha="center")
plt.annotate('AE-CT', (mse_unit[7]+0.5, piqe_unit[7]-0.015), fontsize=12, ha='right')
plt.annotate('SSIM-L', (mse_unit[8]+0.4, piqe_unit[8]-0.2), fontsize=12, ha='right')
plt.annotate('EDGE', (mse_unit[9]+0.43, piqe_unit[9]-0.15), fontsize=12, ha='right')
# plt.ylim([0.925, 1.02])
plt.legend(['Pix2Pix', 'CycleGAN', 'UNIT'])
# plt.xlim([0, 0.012])
plt.ylabel("PIQUE")
# plt.ylabel("PIQE")
plt.xlabel("MSE $(1e-3)$")
plt.tight_layout()
# plt.title('Perception-distorsion evaluation', fontsize=9)
plt.savefig('/Users/francescodifeola/Desktop/PIQE-MSE.pdf', format='pdf')
plt.show()


# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #

# Radially Averaged Power Spectrum (RAPS)

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


# RAPS (Radially Averaged Power Spectrum) overall per patient
"""for test_name in ['test_2']:  # 'test_3', 'elcap_complete'
    data = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_perceptual_diff_4/raps_{test_name}_epoch50")
    data1 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_perceptual_diff_5/raps_{test_name}_epoch50")
    data2 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_perceptual_diff_6/raps_{test_name}_epoch50")

    data_new = copy.deepcopy(data)
    empty_Dictionary(data_new, nesting=1)

    data_new_1 = copy.deepcopy(data1)
    empty_Dictionary(data_new_1, nesting=1)

    data_new_2 = copy.deepcopy(data2)
    empty_Dictionary(data_new_2, nesting=1)

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
    empty_Dictionary(data_new_3, nesting=1)

    for p in data.keys():
        if p != "W0048":
            data_new_3[p] = element_wise_average([data_new[p], data_new_1[p], data_new_2[p]])

    raps_overall = []
    for t in data.keys():
        if t != "W0048":
            raps_overall.append(data_new_3[t])

    raps_average = element_wise_average(raps_overall)

    save_to_json(raps_average, f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_perceptual_diff_4/raps_overall_{test_name}_ep50")"""
# -------------------------------------------------------------------------#
"""profile_test_2_baseline = load_from_json("/Volumes/Untitled/results_per_patient/cycleGAN/metrics_baseline_diff01_1/raps_overall_test_2_ep50")
profile_test_2_texture_max = load_from_json("/Volumes/Untitled/results_per_patient/cycleGAN/metrics_texture_max_diff0001_1/raps_overall_test_2_ep50")
profile_test_2_texture_avg = load_from_json("/Volumes/Untitled/results_per_patient/cycleGAN/metrics_texture_avg_diff0001_1/raps_overall_test_2_ep50")
profile_test_2_texture_Frob = load_from_json("/Volumes/Untitled/results_per_patient/cycleGAN/metrics_texture_Frob_diff0001_1/raps_overall_test_2_ep50")
profile_test_2_texture_att = load_from_json("/Volumes/Untitled/results_per_patient/cycleGAN/metrics_texture_att_diff0001_1/raps_overall_test_2_ep50")
profile_test_2_perceptual = load_from_json("/Volumes/Untitled/results_per_patient/cycleGAN/metrics_perceptual_diff_4/raps_overall_test_2_ep50")

# profile_elcap_baseline = load_from_json("/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_baseline_diff_1/raps_overall_elcap_ep50")
# profile_elcap_texture_max = load_from_json("/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_texture_max_diff0001_1/raps_overall_test_2_ep50")
# profile_elcap_texture_avg = load_from_json("/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_texture_avg_diff0001_1/raps_overall_test_2_ep50")
# profile_elcap_texture_Frob = load_from_json("/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_texture_Frob_diff0001_1/raps_overall_test_2_ep50")
# profile_elcap_texture_att = load_from_json("/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_texture_att_diff0001_1/raps_overall_test_2_ep50")
# profile_elcap_perceptual = load_from_json("/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_perceptual_s_1/raps_overall_test_2_ep50")

# profile_test_3_baseline = load_from_json("/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_baseline_diff_1/raps_overall_test_2_ep50")
# profile_test_3_texture_max = load_from_json("/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_texture_max_diff0001_1/raps_overall_test_2_ep50")
# profile_test_3_texture_avg = load_from_json("/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_texture_avg_diff0001_1/raps_overall_test_2_ep50")
# profile_test_3_texture_Frob = load_from_json("/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_texture_Frob_diff0001_1/raps_overall_test_2_ep50")
# profile_test_3_texture_att = load_from_json("/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_texture_att_diff0001_1/raps_overall_test_2_ep50")
# profile_test_3_perceptual = load_from_json("/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_perceptual_s_1/raps_overall_test_2_ep50")

# Test 2
raps_hd = open(f'/Volumes/Untitled/results_per_patient/metrics_low_dose_reference/HD_raps_test_2_ep50.json')
raps_hd = json.load(raps_hd)
profile_test_2_high_dose = raps_hd

raps_ld = open(f'/Volumes/Untitled/results_per_patient/metrics_low_dose_reference/raps_test_2_ep50.json')
raps_ld = json.load(raps_ld)
profile_test_2_low_dose = raps_ld

# ELCAP low-dose profile
# raps_ld = open(f'/Volumes/Untitled/results_per_patient/metrics_low_dose_reference/raps_ELCAP_ld.json')
# raps_ld = json.load(raps_ld)
# profile_elcap_low_dose = element_wise_average2([raps_ld])

# Test 3 (LIDC/IDRI) low-dose profile
# raps_ld = open(f'/Volumes/Untitled/results_per_patient/metrics_low_dose_reference/raps_test_3_ld.json')
# raps_ld = json.load(raps_ld)
# profile_test_3_low_dose = element_wise_average2([raps_ld])

# Test 2
py.semilogy(profile_test_2_baseline)
py.semilogy(profile_test_2_texture_max)
py.semilogy(profile_test_2_texture_avg)
py.semilogy(profile_test_2_texture_Frob)
py.semilogy(profile_test_2_texture_att)
py.semilogy(profile_test_2_perceptual)
py.semilogy(profile_test_2_low_dose)
py.semilogy(profile_test_2_high_dose)
py.legend(["baseline_loss", "texture_loss_1", "texture_loss_2", "texture_loss_3", "texture_loss_4", "perceptual_loss", "Low-Dose reference", "High-Dose reference"], fontsize=18)
py.xticks(fontsize=18)
py.yticks(fontsize=18)
py.xlabel("Radius", fontsize=18)
py.grid()
py.title("Radially averaged power spectrum: before and after denoising \n using different model configurations", fontsize=18)
py.show()"""

# ELCAP test
"""py.semilogy(profile_elcap_baseline)
py.semilogy(profile_elcap_texture_max)
py.semilogy(profile_elcap_texture_avg)
py.semilogy(profile_elcap_texture_Frob)
py.semilogy(profile_elcap_texture_att)
py.semilogy(profile_elcap_perceptual)
py.semilogy(profile_elcap_low_dose)
py.legend(["baseline", "texture_max", "texture_avg", "texture_Frob", "texture_att", "perceptual_loss", 'Low-Dose reference'])
# py.xticks(fontsize=18)
# py.yticks(fontsize=18)
py.grid()
py.title("Real test set 2 at epoch 50 (UNIT)")
py.show()"""

# Test 3
"""py.semilogy(profile_test_3_baseline)
py.semilogy(profile_test_3_texture_max)
py.semilogy(profile_test_3_texture_avg)
py.semilogy(profile_test_3_texture_Frob)
py.semilogy(profile_test_3_texture_att)
py.semilogy(profile_test_3_perceptual)
py.semilogy(profile_test_3_low_dose)
py.legend(["baseline_loss", "texture_loss_1", "texture_loss_2", "texture_loss_3", "texture_loss_4", "perceptual_loss", "Low-Dose reference"], fontsize=18)
py.xticks(fontsize=18)
py.yticks(fontsize=18)
py.xlabel("Radius", fontsize=18)
py.grid()
py.title("Radially averaged power spectrum: before and after denoising \n using different model configurations", fontsize=18)
py.show()"""


# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #

# Template Matching

def kernel_density_estimate(d):  # , img3

    sns.kdeplot(d["baseline"]["hd"], color='#40E0D0')

    # sns.kdeplot(d["baseline"]["hd"], color='#33a02c')
    sns.kdeplot(d["baseline"]["ld"], color='#e41a1c')
    sns.kdeplot(d["baseline"]["deno"], color='#1f78b4')
    sns.kdeplot(d["perceptual"]["deno"], color='#ff7f00')
    sns.kdeplot(d["autoencoder"]["deno"], color='#6a3d9a')
    sns.kdeplot(d["ssim"]["deno"], color='#b15928')
    sns.kdeplot(d["edge"]["deno"], color='#ff00ff')
    sns.kdeplot(d["texture_max"]["deno"], color='#4daf4a')
    sns.kdeplot(d["texture_avg"]["deno"], color='#ffd700')
    sns.kdeplot(d["texture_Frob"]["deno"], color='gray')
    sns.kdeplot(d["texture_att"]["deno"], color='black')
    # sns.kdeplot(d["perceptual"]["deno"], color='#00ced1')  # Different color for the second appearance of 'perceptual'

    # plt.grid()

    # Create a KDE plot
    # sns.histplot(data1, color='blue')  # , shade=True
    # sns.kdeplot(data2, shade=True, color='red')
    # sns.histplot(data3, color='orange')
    # Customize the plot
    plt.legend(["LDCT", "Baseline", "VGG-16", "AE-CT", "SSIM", "EDGE", "MSTLF-max", "MSTLF-average", "MSLTF-Frobenius", "MSTLF-attention"], fontsize=15)
    # plt.title("Kernel Density Estimate (KDE) Plot (Test set 4, UNIT)")
    plt.ylim(0, 25)
    plt.xlim(0.875, 1.025)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig('/Users/francescodifeola/Desktop/pix2pix_test_4_tm.pdf', format='pdf')
    plt.legend(["LDCT", "baseline", "VGG-16", "AE-CT", "SSIM", "EDGE", "MSTLF-max", "MSTLF-average", "MSLTF-Frobenius", "MSTLF-attention"], fontsize=15)
    # plt.title("Kernel Density Estimate (KDE) Plot (Test set 4, UNIT)")
    plt.ylim(0, 75)
    plt.xlim(0.875, 1.01)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig('/Users/francescodifeola/Desktop/UNIT_test_4_tm.pdf', format='pdf')
    # Display the plot
    plt.show()


# Plot template matching KDE
# model = "cycleGAN"
"""data1 = load_from_json(f"/Volumes/Untitled/results_per_patient/metrics_baseline_1/tm_elcap_complete_epoch50")
model = "cycleGAN"
data1 = load_from_json(f"/Volumes/Untitled/results_per_patient/metrics_baseline_1/tm_elcap_complete_epoch50")
data2 = load_from_json(f"/Volumes/Untitled/results_per_patient/metrics_baseline_2/tm_elcap_complete_epoch50")
data3 = load_from_json(f"/Volumes/Untitled/results_per_patient/metrics_baseline_3/tm_elcap_complete_epoch50")
data4 = load_from_json(f"/Volumes/Untitled/results_per_patient/metrics_perceptual_1/tm_elcap_complete_epoch50")
data5 = load_from_json(f"/Volumes/Untitled/results_per_patient/metrics_perceptual_2/tm_elcap_complete_epoch50")
data6 = load_from_json(f"/Volumes/Untitled/results_per_patient/metrics_perceptual_3/tm_elcap_complete_epoch50")
data7 = load_from_json(f"/Volumes/Untitled/results_per_patient/metrics_texture_max_diff_1/tm_elcap_complete_epoch50")
data8 = load_from_json(f"/Volumes/Untitled/results_per_patient/metrics_texture_max_diff_2/tm_elcap_complete_epoch50")
data9 = load_from_json(f"/Volumes/Untitled/results_per_patient/metrics_texture_max_diff_3/tm_elcap_complete_epoch50")
data10 = load_from_json(f"/Volumes/Untitled/results_per_patient/metrics_texture_avg_diff_1/tm_elcap_complete_epoch50")
data11 = load_from_json(f"/Volumes/Untitled/results_per_patient/metrics_texture_avg_diff_2/tm_elcap_complete_epoch50")
data12 = load_from_json(f"/Volumes/Untitled/results_per_patient/metrics_texture_avg_diff_3/tm_elcap_complete_epoch50")
data13 = load_from_json(f"/Volumes/Untitled/results_per_patient/metrics_texture_Frob_diff_1/tm_elcap_complete_epoch50")
data14 = load_from_json(f"/Volumes/Untitled/results_per_patient/metrics_texture_Frob_diff_2/tm_elcap_complete_epoch50")
data15 = load_from_json(f"/Volumes/Untitled/results_per_patient/metrics_texture_Frob_diff_3/tm_elcap_complete_epoch50")
data16 = load_from_json(f"/Volumes/Untitled/results_per_patient/metrics_texture_attention_diff_1/tm_elcap_complete_epoch50")
data17 = load_from_json(f"/Volumes/Untitled/results_per_patient/metrics_texture_attention_diff_2/tm_elcap_complete_epoch50")
data18 = load_from_json(f"/Volumes/Untitled/results_per_patient/metrics_texture_attention_diff_3/tm_elcap_complete_epoch50")
data19 = load_from_json(f"/Volumes/Untitled/results_per_patient/UNIT/metrics_autoencoder_1/tm_elcap_complete_epoch50")
data20 = load_from_json(f"/Volumes/Untitled/results_per_patient/UNIT/metrics_autoencoder_2/tm_elcap_complete_epoch50")
data21 = load_from_json(f"/Volumes/Untitled/results_per_patient/UNIT/metrics_autoencoder_3/tm_elcap_complete_epoch50")
data22 = load_from_json(f"/Volumes/Untitled/results_per_patient/UNIT/metrics_ssim_1/tm_elcap_complete_epoch50")
data23 = load_from_json(f"/Volumes/Untitled/results_per_patient/UNIT/metrics_ssim_2/tm_elcap_complete_epoch50")
data24 = load_from_json(f"/Volumes/Untitled/results_per_patient/UNIT/metrics_ssim_3/tm_elcap_complete_epoch50")
data25 = load_from_json(f"/Volumes/Untitled/results_per_patient/UNIT/metrics_edge_1/tm_elcap_complete_epoch50")
data26 = load_from_json(f"/Volumes/Untitled/results_per_patient/UNIT/metrics_edge_2/tm_elcap_complete_epoch50")
data27 = load_from_json(f"/Volumes/Untitled/results_per_patient/UNIT/metrics_edge_3/tm_elcap_complete_epoch50")"""

data1 = load_from_json(f"/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_baseline_tm_1/tm_elcap_complete_epoch50")
data2 = load_from_json(f"/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_baseline_tm_2/tm_elcap_complete_epoch50")
data3 = load_from_json(f"/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_baseline_tm_3/tm_elcap_complete_epoch50")
data4 = load_from_json(f"/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_perceptual_tm_1/tm_elcap_complete_epoch50")
data5 = load_from_json(f"/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_perceptual_tm_2/tm_elcap_complete_epoch50")
data6 = load_from_json(f"/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_perceptual_tm_3/tm_elcap_complete_epoch50")
data7 = load_from_json(f"/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_texture_max_tm_1/tm_elcap_complete_epoch50")
data8 = load_from_json(f"/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_texture_max_tm_2/tm_elcap_complete_epoch50")
data9 = load_from_json(f"/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_texture_max_tm_3/tm_elcap_complete_epoch50")
data10 = load_from_json(f"/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_texture_avg_tm_1/tm_elcap_complete_epoch50")
data11 = load_from_json(f"/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_texture_avg_tm_2/tm_elcap_complete_epoch50")
data12 = load_from_json(f"/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_texture_avg_tm_3/tm_elcap_complete_epoch50")
data13 = load_from_json(f"/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_texture_Frob_tm_1/tm_elcap_complete_epoch50")
data14 = load_from_json(f"/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_texture_Frob_tm_2/tm_elcap_complete_epoch50")
data15 = load_from_json(f"/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_texture_Frob_tm_3/tm_elcap_complete_epoch50")
data16 = load_from_json(f"/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_texture_att_tm_1/tm_elcap_complete_epoch50")
data17 = load_from_json(f"/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_texture_att_tm_2/tm_elcap_complete_epoch50")
data18 = load_from_json(f"/Volumes/Untitled/results_per_patient/pix2pix/metrics_pix2pix_texture_att_tm_3/tm_elcap_complete_epoch50")
data19 = load_from_json(f"/Volumes/Untitled/results_per_patient/pix2pix/metric_pix2pix_autoencoder_1/tm_elcap_complete_epoch50")
data20 = load_from_json(f"/Volumes/Untitled/results_per_patient/pix2pix/metric_pix2pix_autoencoder_2/tm_elcap_complete_epoch50")
data21 = load_from_json(f"/Volumes/Untitled/results_per_patient/pix2pix/metric_pix2pix_autoencoder_3/tm_elcap_complete_epoch50")
data22 = load_from_json(f"/Volumes/Untitled/results_per_patient/pix2pix/metric_pix2pix_ssim_1/tm_elcap_complete_epoch50")
data23 = load_from_json(f"/Volumes/Untitled/results_per_patient/pix2pix/metric_pix2pix_ssim_1/tm_elcap_complete_epoch50")
data24 = load_from_json(f"/Volumes/Untitled/results_per_patient/pix2pix/metric_pix2pix_ssim_1/tm_elcap_complete_epoch50")
data25 = load_from_json(f"/Volumes/Untitled/results_per_patient/pix2pix/metric_pix2pix_edge_1/tm_elcap_complete_epoch50")
data26 = load_from_json(f"/Volumes/Untitled/results_per_patient/pix2pix/metric_pix2pix_edge_2/tm_elcap_complete_epoch50")
data27 = load_from_json(f"/Volumes/Untitled/results_per_patient/pix2pix/metric_pix2pix_edge_3/tm_elcap_complete_epoch50")

"""data1 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_baseline_tm_1/tm_elcap_complete_epoch50")
data2 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_baseline_tm_2/tm_elcap_complete_epoch50")
data3 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_baseline_tm_3/tm_elcap_complete_epoch50")
data4 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_perceptual_tm_4/tm_elcap_complete_epoch50")
data5 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_perceptual_tm_5/tm_elcap_complete_epoch50")
data6 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_perceptual_tm_6/tm_elcap_complete_epoch50")
data7 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_texture_max_tm_1/tm_elcap_complete_epoch50")
data8 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_texture_max_tm_2/tm_elcap_complete_epoch50")
data9 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_texture_max_tm_3/tm_elcap_complete_epoch50")
data10 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_texture_avg_tm_1/tm_elcap_complete_epoch50")
data11 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_texture_avg_tm_2/tm_elcap_complete_epoch50")
data12 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_texture_avg_tm_3/tm_elcap_complete_epoch50")
data13 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_texture_Frob_tm_1/tm_elcap_complete_epoch50")
data14 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_texture_Frob_tm_2/tm_elcap_complete_epoch50")
data15 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_texture_Frob_tm_3/tm_elcap_complete_epoch50")
data16 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_texture_att_tm_1/tm_elcap_complete_epoch50")
data17 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_texture_att_tm_2/tm_elcap_complete_epoch50")
data18 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_texture_att_tm_3/tm_elcap_complete_epoch50")
data19 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_autoencoder_1/tm_elcap_complete_epoch50")
data20 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_autoencoder_2/tm_elcap_complete_epoch50")
data21 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_autoencoder_3/tm_elcap_complete_epoch50")
data22 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_ssim_1/tm_elcap_complete_epoch50")
data23 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_ssim_1/tm_elcap_complete_epoch50")
data24 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_ssim_1/tm_elcap_complete_epoch50")
data25 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_edge_1/tm_elcap_complete_epoch50")
data26 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_edge_2/tm_elcap_complete_epoch50")
data27 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_edge_3/tm_elcap_complete_epoch50")"""
data27 = load_from_json(f"/Volumes/Untitled/results_per_patient/UNIT/metrics_edge_3/tm_elcap_complete_epoch50")

d_ = {"baseline": {"deno": [], "hd": [], "ld": []},
      "texture_max": {"deno": [], "hd": [], "ld": []},
      "texture_avg": {"deno": [], "hd": [], "ld": []},
      "texture_Frob": {"deno": [], "hd": [], "ld": []},
      "texture_att": {"deno": [], "hd": [], "ld": []},
      "perceptual": {"deno": [], "hd": [], "ld": []},
      "autoencoder": {"deno": [], "hd": [], "ld": []},
      "ssim": {"deno": [], "hd": [], "ld": []},
      "edge": {"deno": [], "hd": [], "ld": []},
      }

ld = []
hd = []
deno = []
for j in test_4_ids:
    for i in data1[j]["images"].keys():
        ld.append(data1[j]["images"][str(i)]['computed_values']['ld'][1])
        # hd.append(data1[j]["images"][str(i)]['computed_values']['deno'][1])
        # deno.append(data1[j]["images"][str(i)]['computed_values']['hd'][1])
        data_avg = (data1[j]["images"][str(i)]['computed_values']['deno'][1] + data2[j]["images"][str(i)]['computed_values']['deno'][1] +
                    data3[j]["images"][str(i)]['computed_values']['deno'][1]) / 3
        deno.append(data_avg)
        # hd.append(data2[j]["images"][str(i)]['computed_values']['hd'][1])
        # deno.append(data2[j]["images"][str(i)]['computed_values']['deno'][1])
        # hd.append(data3[j]["images"][str(i)]["computed_values"]['hd'])
        # deno.append(data3[j]["images"][str(i)]["computed_values"]['deno'])
# print(sum(ld)/len(ld), sum(deno)/len(deno), sum(hd)/len(hd))
# print(sum(deno)/len(deno), sum(ld)/len(ld))
d_["baseline"]["deno"] = deno
d_["baseline"]["hd"] = hd
d_["baseline"]["ld"] = ld

ld = []
hd = []
deno = []
for j in test_4_ids:
    for i in data1[j]["images"].keys():
        # ld.append(data1[j]["images"][str(i)]['computed_values']['ld'][1])
        # hd.append(data1[j]["images"][str(i)]['computed_values']['hd'][1])
        # deno.append(data4[j]["images"][str(i)]['computed_values']['hd'][1])
        data_avg = (data4[j]["images"][str(i)]['computed_values']['deno'][1] + data5[j]["images"][str(i)]['computed_values']['deno'][1] +
                    data6[j]["images"][str(i)]['computed_values']['deno'][1]) / 3
        deno.append(data_avg)
        # hd.append(data5[j]["images"][str(i)]['computed_values']['hd'][1])
        # deno.append(data5[j]["images"][str(i)]['computed_values']['deno'][1])
        # hd.append(data6[j]["images"][str(i)]['computed_values']['hd'][1])
        # deno.append(data6[j]["images"][str(i)]['computed_values']['deno'][1])

d_["perceptual"]["deno"] = deno
d_["perceptual"]["hd"] = hd
d_["perceptual"]["ld"] = ld

ld = []
hd = []
deno = []
for j in test_4_ids:
    for i in data1[j]["images"].keys():
        # ld.append(data1[j]["images"][str(i)]['computed_values']['ld'][1])
        # hd.append(data1[j]["images"][str(i)]['computed_values']['hd'][1])
        # deno.append(data6[j]["images"][str(i)]['computed_values']['hd'][1])
        data_avg = (data7[j]["images"][str(i)]['computed_values']['deno'][1] + data8[j]["images"][str(i)]['computed_values']['deno'][1] +
                    data9[j]["images"][str(i)]['computed_values']['deno'][1]) / 3
        deno.append(data_avg)
        # hd.append(data8[j]["images"][str(i)]['computed_values']['hd'][1])
        # deno.append(data8[j]["images"][str(i)]['computed_values']['deno'][1])
        # hd.append(data9[j]["images"][str(i)]['computed_values']['hd'])
        # deno.append(data9[j]["images"][str(i)]['computed_values']['deno'])

d_["texture_max"]["deno"] = deno
d_["texture_max"]["hd"] = hd
d_["texture_max"]["ld"] = ld

ld = []
hd = []
deno = []
for j in test_4_ids:
    for i in data1[j]["images"].keys():
        # ld.append(data1[j]["images"][str(i)]['computed_values']['ld'][1])
        # hd.append(data1[j]["images"][str(i)]['computed_values']['hd'][1])
        # deno.append(data11[j]["images"][str(i)]['computed_values']['hd'][1])
        data_avg = (data10[j]["images"][str(i)]['computed_values']['deno'][1] + data11[j]["images"][str(i)]['computed_values']['deno'][1] +
                    data12[j]["images"][str(i)]['computed_values']['deno'][1]) / 3
        deno.append(data_avg)
        # hd.append(data11[j]["images"][str(i)]['computed_values']['hd'][1])
        # deno.append(data11[j]["images"][str(i)]['computed_values']['deno'][1])
        # hd.append(data12[j]["images"][str(i)]['computed_values']['hd'])
        # deno.append(data12[j]["images"][str(i)]['computed_values']['deno'])

d_["texture_avg"]["deno"] = deno
d_["texture_avg"]["hd"] = hd
d_["texture_avg"]["ld"] = ld

ld = []
hd = []
deno = []
for j in test_4_ids:
    for i in data1[j]["images"].keys():
        # ld.append(data1[j]["images"][str(i)]['computed_values']['ld'][1])
        # hd.append(data1[j]["images"][str(i)]['computed_values']['hd'][1])
        # deno.append(data14[j]["images"][str(i)]['computed_values']['hd'][1])
        data_avg = (data13[j]["images"][str(i)]['computed_values']['deno'][1] + data14[j]["images"][str(i)]['computed_values']['deno'][1] +
                    data15[j]["images"][str(i)]['computed_values']['deno'][1]) / 3
        deno.append(data_avg)
        # hd.append(data14[j]["images"][str(i)]['computed_values']['hd'][1])
        # deno.append(data14[j]["images"][str(i)]['computed_values']['deno'][1])
        # hd.append(data15[j]["images"][str(i)]['computed_values']['hd'])
        # deno.append(data15[j]["images"][str(i)]['computed_values']['deno'])

d_["texture_Frob"]["deno"] = deno
d_["texture_Frob"]["hd"] = hd
d_["texture_Frob"]["ld"] = ld

ld = []
hd = []
deno = []
for j in test_4_ids:
    for i in data1[j]["images"].keys():
        # ld.append(data1[j]["images"][str(i)]['computed_values']['ld'][1])
        # hd.append(data1[j]["images"][str(i)]['computed_values']['hd'][1])
        # deno.append(data17[j]["images"][str(i)]['computed_values']['hd'][1])
        data_avg = (data16[j]["images"][str(i)]['computed_values']['deno'][1] + data17[j]["images"][str(i)]['computed_values']['deno'][1] +
                    data18[j]["images"][str(i)]['computed_values']['deno'][1]) / 3
        deno.append(data_avg)
        # hd.append(data17[j]["images"][str(i)]['computed_values']['hd'][1])
        # deno.append(data17[j]["images"][str(i)]['computed_values']['deno'][1])
        # hd.append(data18[j]["images"][str(i)]['computed_values']['hd'][1])
        # deno.append(data18[j]["images"][str(i)]['computed_values']['deno'][1])

d_["texture_att"]["deno"] = deno
d_["texture_att"]["hd"] = hd
d_["texture_att"]["ld"] = ld

ld = []
hd = []
deno = []
for j in test_4_ids:
    for i in data1[j]["images"].keys():
        # ld.append(data1[j]["images"][str(i)]['computed_values']['ld'][1])
        # hd.append(data1[j]["images"][str(i)]['computed_values']['hd'][1])
        # deno.append(data17[j]["images"][str(i)]['computed_values']['hd'][1])
        print(j)
        print(data19[j]["images"][str(i)]['computed_values']['deno'][1])
        print(data20[j]["images"][str(i)]['computed_values']['deno'][1])
        print(data21[j]["images"][str(i)]['computed_values']['deno'][1])
        data_avg = (data19[j]["images"][str(i)]['computed_values']['deno'][1] + data20[j]["images"][str(i)]['computed_values']['deno'][1] +
                    data21[j]["images"][str(i)]['computed_values']['deno'][1]) / 3
        deno.append(data_avg)
        # hd.append(data17[j]["images"][str(i)]['computed_values']['hd'][1])
        # deno.append(data17[j]["images"][str(i)]['computed_values']['deno'][1])
        # hd.append(data18[j]["images"][str(i)]['computed_values']['hd'][1])
        # deno.append(data18[j]["images"][str(i)]['computed_values']['deno'][1])

d_["autoencoder"]["deno"] = deno
d_["autoencoder"]["hd"] = hd
d_["autoencoder"]["ld"] = ld

ld = []
hd = []
deno = []
for j in test_4_ids:
    for i in data1[j]["images"].keys():
        # ld.append(data1[j]["images"][str(i)]['computed_values']['ld'][1])
        # hd.append(data1[j]["images"][str(i)]['computed_values']['hd'][1])
        # deno.append(data17[j]["images"][str(i)]['computed_values']['hd'][1])
        data_avg = (data22[j]["images"][str(i)]['computed_values']['deno'][1] + data23[j]["images"][str(i)]['computed_values']['deno'][1] +
                    data24[j]["images"][str(i)]['computed_values']['deno'][1]) / 3
        deno.append(data_avg)
        # hd.append(data17[j]["images"][str(i)]['computed_values']['hd'][1])
        # deno.append(data17[j]["images"][str(i)]['computed_values']['deno'][1])
        # hd.append(data18[j]["images"][str(i)]['computed_values']['hd'][1])
        # deno.append(data18[j]["images"][str(i)]['computed_values']['deno'][1])

d_["ssim"]["deno"] = deno
d_["ssim"]["hd"] = hd
d_["ssim"]["ld"] = ld

ld = []
hd = []
deno = []
for j in test_4_ids:
    for i in data1[j]["images"].keys():
        # ld.append(data1[j]["images"][str(i)]['computed_values']['ld'][1])
        # hd.append(data1[j]["images"][str(i)]['computed_values']['hd'][1])
        # deno.append(data17[j]["images"][str(i)]['computed_values']['hd'][1])
        data_avg = (data25[j]["images"][str(i)]['computed_values']['deno'][1] + data26[j]["images"][str(i)]['computed_values']['deno'][1] +
                    data27[j]["images"][str(i)]['computed_values']['deno'][1]) / 3
        deno.append(data_avg)
        # hd.append(data17[j]["images"][str(i)]['computed_values']['hd'][1])
        # deno.append(data17[j]["images"][str(i)]['computed_values']['deno'][1])
        # hd.append(data18[j]["images"][str(i)]['computed_values']['hd'][1])
        # deno.append(data18[j]["images"][str(i)]['computed_values']['deno'][1])

d_["edge"]["deno"] = deno
d_["edge"]["hd"] = hd
d_["edge"]["ld"] = ld

kernel_density_estimate(d_)
