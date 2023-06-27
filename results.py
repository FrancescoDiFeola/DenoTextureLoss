import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from scipy.stats import wilcoxon


def convert_string_to_floats(long_string_with_commas):
    long_string_without_brackets = long_string_with_commas.replace('[', '').replace(']', '')
    number_strings = long_string_without_brackets.split(',')
    float_numbers = [float(number.replace(' ', '')) for number in number_strings]
    return float_numbers


def compute_wilcoxon_test(set1, set2):
    # difference between two set of metrics
    set1 = np.array(set1)

    set2 = np.array(set2)

    d = np.squeeze(np.subtract(set1, set2))
    # To test the null hypothesis that there
    # is no value difference between the two sets, we can apply the two-sided test
    # that is we want to verify that the distribution underlying d is not symmetric about zero.
    res = wilcoxon(d, alternative='greater')
    print(res.statistic, res.pvalue)


if __name__ == '__main__':
    baseline_1 = pd.read_csv(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_baseline_wind_1/metrics_test_1.csv', header=None)
    baseline_2 = pd.read_csv(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_baseline_wind_2/metrics_test_1.csv', header=None)
    baseline_3 = pd.read_csv(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_baseline_wind_3/metrics_test_1.csv', header=None)

    baseline_1 = convert_string_to_floats(baseline_1.iloc[0][1])
    baseline_2 = convert_string_to_floats(baseline_2.iloc[0][1])
    baseline_3 = convert_string_to_floats(baseline_3.iloc[0][1])

    average_baseline = [(x + y) / 2 for x, y in zip(baseline_1, baseline_3)]
    print(f'Baseline: {average_baseline[-1]}')

    texture_max_1 = pd.read_csv(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_texture_max_wind_1/metrics_test_1.csv',
        header=None)
    texture_max_2 = pd.read_csv(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_texture_max_wind_2/metrics_test_1.csv',
        header=None)
    texture_max_3 = pd.read_csv(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_texture_max_wind_3/metrics_test_1.csv',
        header=None)

    texture_max_1 = convert_string_to_floats(texture_max_1.iloc[0][1])
    texture_max_2 = convert_string_to_floats(texture_max_2.iloc[0][1])
    texture_max_3 = convert_string_to_floats(texture_max_3.iloc[0][1])


    average_texture_max = [(x + y + z) / 3 for x, y, z in zip(texture_max_1, texture_max_2, texture_max_3)]
    print(baseline_3)
    print(f'Texture max: {average_texture_max[-1]}')

    plt.figure()
    plt.plot(range(1, 201), baseline_1)
    plt.plot(range(1, 201), average_texture_max)
    plt.legend(['baseline', 'texture_max'])
    plt.title('PSNR over epochs on the Mayo test (1 patient)')
    plt.show()

    # texture_norm_1 = pd.read_csv(
    #    '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_normalized_1/metrics_epoch200.csv',
    #    header=None)
    # texture_norm_2 = pd.read_csv(
    #     '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_normalized_2/metrics_epoch200.csv',
    #     header=None)
    # texture_norm_3 = pd.read_csv(
    #     '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_normalized_3/metrics_epoch200.csv',
    #     header=None)

    # texture_norm_1 = convert_string_to_floats(texture_norm_1.iloc[0][1])
    # texture_norm_2 = convert_string_to_floats(texture_norm_2.iloc[0][1])
    # texture_norm_3 = convert_string_to_floats(texture_norm_3.iloc[0][1])

    # average_texture_norm = [(x + y + z) / 3 for x, y, z in zip(texture_norm_1, texture_norm_2, texture_norm_3)]

    # print(f'Texture norm: {average_texture_norm[-1]}')

    # texture_averaged_d5_1 = pd.read_csv(
    #     '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_offset5_1/metrics_epoch200.csv',
    #     header=None)
    # texture_averaged_d5_2 = pd.read_csv(
    #     '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_offset5_2/metrics_epoch200.csv',
    #     header=None)
    # texture_averaged_d5_3 = pd.read_csv(
    #     '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_offset5_3/metrics_epoch200.csv',
    #     header=None)

    # texture_averaged_d5_1 = convert_string_to_floats(texture_averaged_d5_1.iloc[0][1])
    # texture_averaged_d5_2 = convert_string_to_floats(texture_averaged_d5_2.iloc[0][1])
    # texture_averaged_d5_3 = convert_string_to_floats(texture_averaged_d5_3.iloc[0][1])

    # average_texture_averaged_d5 = [(x + y + z) / 3 for x, y, z in
    #                               zip(texture_averaged_d5_1, texture_averaged_d5_2, texture_averaged_d5_3)]

    # print(f'Texture avg d5: {average_texture_averaged_d5[-1]}')

    # perceptual_1 = pd.read_csv(
    #     '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_perceptual_1/metrics_epoch200.csv',
    #     header=None)
    # perceptual_2 = pd.read_csv(
    #     '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_perceptual_2/metrics_epoch200.csv',
    #     header=None)
    # perceptual_3 = pd.read_csv(
    #     '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_perceptual_3/metrics_epoch200.csv',
    #     header=None)

    # perceptual_1 = convert_string_to_floats(perceptual_1.iloc[0][1])
    # perceptual_2 = convert_string_to_floats(perceptual_2.iloc[0][1])
    # perceptual_3 = convert_string_to_floats(perceptual_3.iloc[0][1])

    # average_perceptual = [(x + y + z) / 3 for x, y, z in zip(perceptual_1, perceptual_2, perceptual_3)]


    # print(f'Perceptual: {average_perceptual[-1]}')

    '''perceptual_l45_1 = pd.read_csv(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_perceptual_l45_1/metrics_test.csv',
        header=None)
    perceptual_l45_2 = pd.read_csv(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_perceptual_l45_2/metrics_test.csv',
        header=None)
    perceptual_l45_3 = pd.read_csv(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_perceptual_l45_3/metrics_test.csv',
        header=None)

    perceptual_l45_1 = convert_string_to_floats(perceptual_l45_1.iloc[2][1])
    perceptual_l45_2 = convert_string_to_floats(perceptual_l45_2.iloc[2][1])
    perceptual_l45_3 = convert_string_to_floats(perceptual_l45_3.iloc[2][1])

    average_perceptual_l45 = [(x + y + z) / 3 for x, y, z in zip(perceptual_l45_1, perceptual_l45_2, perceptual_l45_3)]

    print(f'Perceptual_l45: {average_perceptual_l45[-1]}')

    # compute_wilcoxon_test(average_texture_norm, average_texture_max)
    plt.figure()
    # plt.scatter([31.9714, 31.8105, 31.8427, 32.1373, 31.8319], [0.9626, 0.9594, 0.9580, 0.9614, 0.9588], marker='o')
    plt.scatter([29.2322, 30.6947, 29.3731, 30.7706, 29.7606], [1.0021, 1.0044, 0.9393, 1.0165, 0.9564], marker='o', color='orange')
    plt.annotate('baseline', (31.9714, 0.9626))
    plt.annotate('Texture max', (31.8105, 0.9594))
    plt.annotate('Texture norm', (31.8427, 0.9580))
    plt.annotate('Texture avg d5', (32.1373, 0.9614))
    plt.annotate('Perceptual ImNet', (31.8319, 0.9588))
    plt.annotate('baseline', (29.2322, 1.0021))
    plt.annotate('Texture max', (30.6947, 1.0044))
    plt.annotate('Texture norm', (29.3731, 0.9393))
    plt.annotate('Texture avg d5', (30.7706, 1.0165))
    plt.annotate('Perceptual ImNet', (29.7606, 0.9564))
    #plt.ylim([0.925, 1.02])
    plt.legend(['CycleGAN'])
    plt.ylabel("VIF")
    plt.xlabel("PSNR")
    plt.grid()
    plt.title('Average metrics at epoch 200 over 1 test patient (Mayo)')
    plt.show()
    plt.figure()
    plt.plot(range(1, 201), average_baseline)
    plt.plot(range(1, 201), average_texture_max)
    plt.plot(range(1, 201), average_texture_norm)
    plt.plot(range(1, 201), average_texture_averaged_d5)
    plt.plot(range(1, 201), average_perceptual)
    # plt.plot(range(1, 201), average_perceptual_l45)
    # plt.text(110, average_baseline[0], f'Baseline epoch 200: {round(average_baseline[-1], 4)}', fontsize=10, color='black')
    # plt.text(110, average_baseline[0]-0.01, f'Texture max epoch 200: {round(average_texture_max[-1], 4)}', fontsize=10, color='black')
    # plt.text(110, average_baseline[0] - 0.02, f'Texture norm epoch 200: {round(average_texture_norm[-1], 4)}', fontsize=10, color='black')
    # plt.text(110, average_baseline[0] - 0.03, f'Texture norm epoch 200: {round(average_texture_averaged_d5[-1], 4)}', fontsize=10, color='black')
    # plt.text(110, average_baseline[0]-0.04, f'Perceptual epoch 200: {round(average_perceptual[-1], 4)}', fontsize=10, color='black')
    plt.legend(['baseline', 'texture_max', 'texture_norm', 'texture_averaged_d5', 'perceptual_ImNet'])
    plt.title('VIF over epochs on the Mayo test (1 patient)')
    plt.show()

    # Define a function to convert string to tensor
    def convert_to_tensor(string):
        if string.startswith('tensor(') and string.endswith(')'):
            tensor_str = string.replace('tensor(', '').replace(')', '')
            return torch.tensor(eval(tensor_str))
        else:
            return string

    exp_1 = pd.read_csv('/Users/francescodifeola/Desktop/downloads_alvis/loss_texture_8/idx_texture_B.csv')
    id_d_1 = exp_1['id_d'].map(convert_to_tensor)
    id_theta_1 = exp_1['id_theta'].map(convert_to_tensor)



    id_d = list()
    id_theta = list()
    for i in id_d_1:
        if len(i) > 1:
            id_d.append(i[0].item())
        else:
            id_d.append(i.item())

    for j in id_theta_1:
        if len(j) > 1:
            id_theta.append(j[0].item())
        else:
            id_theta.append(j.item())

    histogram = {
        '1-0°': 0,
        '1-45°': 0,
        '1-90°': 0,
        '1-135°': 0,
         '3-0°': 0,
        '3-45°': 0,
        '3-90°': 0,
        '3-135°': 0,
        '5-0°': 0,
        '5-45°': 0,
        '5-90°': 0,
        '5-135°': 0,
        '7-0°': 0,
        '7-45°': 0,
        '7-90°': 0,
        '7-135°': 0,
    }
    for k in zip(id_d, id_theta):
          if k == (0, 0):
              histogram['1-0°'] += 1
          elif  k == (0, 1):
              histogram['1-45°'] += 1
          elif  k == (0, 2):
              histogram['1-90°'] += 1
          elif  k == (0, 3):
              histogram['1-135°'] += 1
          elif  k == (1, 0):
              histogram['3-0°'] += 1
          elif  k == (1, 1):
              histogram['3-45°'] += 1
          elif  k == (1, 2):
              histogram['3-90°'] += 1
          elif  k == (1, 3):
              histogram['3-135°'] += 1
          elif  k == (2, 0):
              histogram['5-0°'] += 1
          elif  k == (2, 1):
              histogram['5-45°'] += 1
          elif  k == (2, 2):
              histogram['5-90°'] += 1
          elif  k == (2, 3):
              histogram['5-135°'] += 1
          elif  k == (3, 0):
              histogram['7-0°'] += 1
          elif  k == (3, 1):
              histogram['7-45°'] += 1
          elif  k == (3, 2):
              histogram['7-90°'] += 1
          elif  k == (3, 3):
             histogram['7-135°'] += 1

    plt.figure()
    plt.bar(histogram.keys(), histogram.values())
    plt.show()'''
